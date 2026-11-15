# -*- coding: utf-8 -*-
"""LSTM-VAE：LSTM-based Variational Autoencoder 异常检测。

参考实现：https://github.com/TimyadNyda/Variational-Lstm-Autoencoder
论文：Park, Hoshi & Kemp, "A Multimodal Anomaly Detector for
Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder",
IEEE RA-L 2018（SWaT/WADI 上的 LSTM-VAE 基线通常引用该实现）。

参考实现（TensorFlow/Keras）的结构：
    Encoder: 单层 LSTMCell(intermediate_dim, activation=ReLU, forget_bias=1)
             取最后时刻输出 -> Linear -> z_mean / z_logvar(softplus)
    Decoder: z 复制 seq_len 次 -> 两层 LSTMCell(intermediate_dim, n_dim)
             默认 tanh 输出
    损失:    MSE(recon, x) + kl_coef * KL(q(z|x) || N(0,I))
             + reg_lambda * L2(Encoder)
    预处理:  MinMaxScaler(feature_range=(-1,1))
    训练:    Adam(lr=1e-3), batch=100, epochs=200, grad_clip=10

本实现用 PyTorch 逐项复刻上述行为：
- 自定义 ReLULSTMCell（TF 的 `activation=tf.nn.relu` 不能直接用 nn.LSTM 代替）；
- 解码器两层 LSTM 的 forget gate bias 初始化为 1（对应 TF 的 forget_bias=1）；
- `z_logvar = softplus(...)`、采样 `mean + exp(0.5*logvar)*eps`、KL 公式
  与参考实现一致；
- 推理时默认使用 z 的均值（确定性评分），训练时重参数采样；
- 可选 `--clip-input`，把输入裁剪到 [-1,1]（WADI 上建议开启）。

用法::

    # SWaT / WADI，统一两阶段协议
    python -m models.reconstruction.lstm_vae --stage all \
        --csv 08_SWaT/SWaT_all_1s.csv

    python -m models.reconstruction.lstm_vae --stage all \
        --csv 09_WADI/WADI_all_1s.csv --clip-input

    # 关键超参（参考实现/论文基准）
    # --intermediate-dim 15 --z-dim 3 --kl-coef 0.1 --reg-lambda 0.01
    # --batch-size 256 --epochs 100 --lr 1e-3 --grad-clip 10
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.common.config import RunConfig, build_parser
from models.common.detection import DetectionConfig, run_detection_stage
from models.common.recon_runner import run_recon_stage

MODEL_NAME = "LSTMVAE"
INTERMEDIATE_DIM = 15
Z_DIM = 3
KL_COEF = 0.1
REG_LAMBDA = 0.01
GRAD_CLIP = 10.0


class ReLULSTMCell(nn.Module):
    """与 TF ``LSTMCell(activation=tf.nn.relu, forget_bias=1)`` 对齐的单元。

    gates: i, f, g, o（PyTorch 顺序）
    - i/f/o 用 sigmoid；
    - g 用 ReLU（TF activation=relu 作用于候选状态 j）；
    - c = f*c + i*g；
    - h = o * ReLU(c)（TF 的 activation 也作用于 cell 输出）。
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        h = self.hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * h, self.input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * h, h))
        self.bias_ih = nn.Parameter(torch.zeros(4 * h))
        self.bias_hh = nn.Parameter(torch.zeros(4 * h))
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        # forget_bias=1：加到 f 的门前偏置上
        with torch.no_grad():
            self.bias_ih[h:2 * h].fill_(1.0)

    def forward(self, x, state):
        h_prev, c_prev = state
        gates = (F.linear(x, self.weight_ih, self.bias_ih)
                 + F.linear(h_prev, self.weight_hh, self.bias_hh))
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = F.relu(g)
        c = f * c_prev + i * g
        h = o * F.relu(c)
        return h, c


def _set_forget_bias_one(lstm: nn.LSTM) -> None:
    """对齐 TF ``forget_bias=1``：所有 bias 初始化为 0，再给 forget 门 +1。

    PyTorch LSTM 的 bias 拆成 ``bias_ih`` 与 ``bias_hh`` 两部分；
    若两部分都 +1，实际 forget 偏置会变成 +2。这里只给 ``bias_ih``
    的 forget 门加 1，``bias_hh`` 保持 0，与 TF 的单一 bias 等价。
    """
    for name, param in lstm.named_parameters():
        if not name.startswith("bias"):
            continue
        with torch.no_grad():
            param.zero_()
        if "bias_ih" in name and param.ndim == 1 and param.numel() % 4 == 0:
            h = param.numel() // 4
            with torch.no_grad():
                param[h:2 * h].fill_(1.0)


class LSTMVAE(nn.Module):
    """LSTM 编码器 + 潜变量头 + 两层 LSTM 解码器的 VAE。"""

    def __init__(self, n_channels: int, seq_len: int,
                 intermediate_dim: int = INTERMEDIATE_DIM,
                 z_dim: int = Z_DIM, kl_coef: float = KL_COEF,
                 reg_lambda: float = REG_LAMBDA, grad_clip: float = GRAD_CLIP,
                 clip_input: bool = False):
        super().__init__()
        self.n_channels = int(n_channels)
        self.seq_len = int(seq_len)
        self.intermediate_dim = int(intermediate_dim)
        self.z_dim = int(z_dim)
        self.kl_coef = float(kl_coef)
        self.reg_lambda = float(reg_lambda)
        self.grad_clip = float(grad_clip)
        self.clip_input = bool(clip_input)

        self.encoder = ReLULSTMCell(self.n_channels, self.intermediate_dim)
        self.z_mean_layer = nn.Linear(self.intermediate_dim, self.z_dim)
        self.z_logvar_layer = nn.Linear(self.intermediate_dim, self.z_dim)

        # 解码器：z -> intermediate_dim -> n_channels，默认 tanh
        self.decoder1 = nn.LSTM(self.z_dim, self.intermediate_dim,
                                batch_first=True)
        self.decoder2 = nn.LSTM(self.intermediate_dim, self.n_channels,
                                batch_first=True)
        _set_forget_bias_one(self.decoder1)
        _set_forget_bias_one(self.decoder2)

    # ------------------------------------------------------------------
    # 前向 / 损失
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor):
        b, t, _ = x.shape
        h = x.new_zeros(b, self.intermediate_dim)
        c = x.new_zeros(b, self.intermediate_dim)
        for k in range(t):
            h, c = self.encoder(x[:, k, :], (h, c))
        mu = self.z_mean_layer(h)
        # 参考实现用 softplus 产生 z_sigma，再按 log-variance 使用
        logvar = F.softplus(self.z_logvar_layer(h))
        return mu, logvar

    def reparameterize(self, mu, logvar, sample: bool):
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        h1, _ = self.decoder1(z_rep)
        out, _ = self.decoder2(h1)
        return out

    def forward(self, x: torch.Tensor, sample: bool | None = None):
        if sample is None:
            sample = self.training
        x_in = x.clamp(-1.0, 1.0) if self.clip_input else x
        mu, logvar = self.encode(x_in)
        z = self.reparameterize(mu, logvar, sample)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def kl_divergence(mu, logvar) -> torch.Tensor:
        return -0.5 * torch.sum(
            1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)

    def _l2_encoder(self) -> torch.Tensor:
        vals = [0.5 * p.pow(2).sum()
                for n, p in self.encoder.named_parameters() if p.dim() > 1]
        if not vals:
            return torch.zeros((), device=self.z_mean_layer.weight.device)
        return torch.stack(vals).mean()

    def training_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.clamp(-1.0, 1.0) if self.clip_input else x
        recon, mu, logvar = self(x_in, sample=True)
        recon_loss = F.mse_loss(recon, x_in)
        kl = self.kl_divergence(mu, logvar).mean()
        return (recon_loss + self.kl_coef * kl
                + self.reg_lambda * self._l2_encoder())

    # ------------------------------------------------------------------
    # 自定义训练循环（参考实现：Adam + clip 10 + 无 scheduler）
    # ------------------------------------------------------------------
    def fit(self, loader, epochs: int, lr: float, device,
            name: str = "LSTMVAE", val_loader=None,
            history: list[dict] | None = None) -> float:
        self.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        start = time.time()

        for epoch in range(1, int(epochs) + 1):
            self.train()
            total = 0.0
            seen = 0
            pbar = tqdm(loader, desc=f"[{name}] Epoch {epoch:03d}/{epochs}",
                        unit="batch", ncols=110, dynamic_ncols=True)
            for x, _ in pbar:
                x = x.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = self.training_loss(x)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), self.grad_clip)
                opt.step()
                total += float(loss.item()) * x.size(0)
                seen += x.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.5f}",
                                  "avg": f"{total / max(seen, 1):.5f}"})
            pbar.close()

            row = {"epoch": epoch, "loss": total / max(seen, 1)}
            if val_loader is not None:
                self.eval()
                v_total = 0.0
                v_seen = 0
                with torch.no_grad():
                    for xv, _ in val_loader:
                        xv = xv.to(device, non_blocking=True)
                        v_total += float(self.training_loss(xv).item()) * xv.size(0)
                        v_seen += xv.size(0)
                row["val_loss"] = v_total / max(v_seen, 1)
                print(f"[{name}] Epoch {epoch:03d} | "
                      f"loss {row['loss']:.6f} | val {row['val_loss']:.6f}")
            else:
                print(f"[{name}] Epoch {epoch:03d} | loss {row['loss']:.6f}")
            if history is not None:
                history.append(row)
        return time.time() - start

    # ------------------------------------------------------------------
    # 异常分数
    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_batch(self, x: torch.Tensor) -> dict:
        self.eval()
        x_in = x.clamp(-1.0, 1.0) if self.clip_input else x
        recon, mu, logvar = self(x_in, sample=False)
        diff = recon - x_in
        abs_err = diff.abs()
        sq_err = diff ** 2
        sq_ts = sq_err.mean(dim=-1)     # [B,L]
        abs_ts = abs_err.mean(dim=-1)   # [B,L]
        kl = self.kl_divergence(mu, logvar)
        k = min(3, self.seq_len)
        return {
            "window": {
                "recon_mean": sq_ts.mean(dim=1),
                "recon_max": sq_ts.max(dim=1).values,
                "recon_top3_mean": torch.sort(sq_ts, dim=1).values[:, -k:].mean(dim=1),
                "kl": kl,
            },
            "point": {
                "point_recon_mse_last": sq_ts[:, -1],
                "point_recon_mae_last": abs_ts[:, -1],
                "point_kl": kl,
            },
            "ts": {
                "sq_ts": sq_ts,
                "abs_ts": abs_ts,
            },
            "channel": {
                "abs_last": abs_err[:, -1, :],
                "sq_last": sq_err[:, -1, :],
                "recon_last": recon[:, -1, :],
                # 原始真值用于反变换绘图；误差按裁剪后的输入计算
                "true_last": x[:, -1, :],
            },
        }


def build_model(n_channels: int, seq_len: int, data=None,
                intermediate_dim: int = INTERMEDIATE_DIM,
                z_dim: int = Z_DIM, kl_coef: float = KL_COEF,
                reg_lambda: float = REG_LAMBDA,
                grad_clip: float = GRAD_CLIP,
                clip_input: bool = False) -> LSTMVAE:
    return LSTMVAE(
        n_channels, seq_len, intermediate_dim=intermediate_dim,
        z_dim=z_dim, kl_coef=kl_coef, reg_lambda=reg_lambda,
        grad_clip=grad_clip, clip_input=clip_input)


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def _cli_has(flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])


def _prefix_from_csv(csv: str) -> str:
    name = str(csv).lower()
    if "wadi" in name:
        return "lstmvae_wadi"
    return "lstmvae_swat"


def _parse_lstmvae_config():
    p = build_parser(
        "LSTM-VAE 重构异常检测（SWaT / WADI）",
        batch_size=256, epochs=100, lr=1e-3)
    p.add_argument("--intermediate-dim", type=int, default=INTERMEDIATE_DIM,
                   help="编码器 LSTM / 解码器第一层维度（参考实现 15）")
    p.add_argument("--z-dim", type=int, default=Z_DIM,
                   help="潜变量维度（参考实现 3）")
    p.add_argument("--kl-coef", type=float, default=KL_COEF,
                   help="KL 项权重（参考实现 0.1）")
    p.add_argument("--reg-lambda", type=float, default=REG_LAMBDA,
                   help="Encoder L2 权重（参考实现 0.01）")
    p.add_argument("--grad-clip", type=float, default=GRAD_CLIP,
                   help="梯度裁剪范数（参考实现 10）")
    p.add_argument("--clip-input", action="store_true",
                   help="把输入裁剪到 [-1,1]（MinMax11 范围；WADI 建议开启）")
    args = p.parse_args()

    fields = set(RunConfig.__dataclass_fields__)
    cfg = RunConfig(**{k: v for k, v in vars(args).items() if k in fields})
    cfg.out_dir = Path(cfg.out_dir)
    if not _cli_has("--seq-len"):
        cfg.seq_len = 12
    if not _cli_has("--scaler"):
        cfg.scaler = "minmax11"
    if cfg.smoke and not _cli_has("--epochs"):
        cfg.epochs = 1
    return cfg, args


def main() -> None:
    cfg, args = _parse_lstmvae_config()
    prefix = _prefix_from_csv(cfg.csv)
    artifact_dir = Path(cfg.artifact_dir) if cfg.artifact_dir else (
        Path(cfg.out_dir) / "runs" / prefix)
    print(f"[{MODEL_NAME}] hidden={args.intermediate_dim} | z={args.z_dim} | "
          f"kl={args.kl_coef} | reg={args.reg_lambda} | "
          f"clip_input={args.clip_input} | scaler={cfg.scaler}")
    summary = {"model": MODEL_NAME, "prefix": prefix,
               "artifact_dir": str(artifact_dir), "stage": cfg.stage}

    if cfg.stage in ("train", "all"):
        build = lambda n_channels, seq_len, data=None: build_model(
            n_channels, seq_len, data,
            intermediate_dim=args.intermediate_dim,
            z_dim=args.z_dim, kl_coef=args.kl_coef,
            reg_lambda=args.reg_lambda, grad_clip=args.grad_clip,
            clip_input=args.clip_input)
        artifact_dir = run_recon_stage(
            MODEL_NAME, prefix, build, cfg, save_model=True)
        summary["artifact_dir"] = str(artifact_dir)

    if cfg.stage in ("detect", "all"):
        run_detection_stage(
            artifact_dir, DetectionConfig.from_run_config(cfg))

    print(f"\n[{MODEL_NAME}] 完成: {summary}")


if __name__ == "__main__":
    main()
