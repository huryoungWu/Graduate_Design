# -*- coding: utf-8 -*-
"""USAD：UnSupervised Anomaly Detection on Multivariate Time Series。

论文：Audibert et al., KDD 2020, https://doi.org/10.1145/3394486.3403392
官方实现：https://github.com/manigalati/usad

网络结构与官方实现一致：
    Encoder:  w -> Linear(w/2) -> ReLU -> Linear(w/4) -> ReLU -> Linear(z) -> ReLU
    Decoder:  z -> Linear(w/4) -> ReLU -> Linear(w/2) -> ReLU -> Linear(w) -> Sigmoid
    AE1 = Encoder + Decoder1，AE2 = Encoder + Decoder2（共享编码器）

其中 ``w = seq_len * n_channels`` 是展平后的窗口维度，
``z = seq_len * hidden_size``（官方 SWaT notebook 取 hidden_size=100）。

训练采用两个 Adam 优化器和 epoch 相关的对抗损失（官方 training_step）::

    L1 = (1/n) * MSE(W, AE1(W)) + (1 - 1/n) * MSE(W, AE2(AE1(W)))
    L2 = (1/n) * MSE(W, AE2(W)) - (1 - 1/n) * MSE(W, AE2(AE1(W)))

优化器 1 更新 Encoder+Decoder1（最小化 L1），优化器 2 更新
Encoder+Decoder2（最小化 L2）。n 为从 1 开始的 epoch 序号。

测试异常分数（官方 testing）::

    score(W) = alpha * MSE(W, AE1(W)) + beta * MSE(W, AE2(AE1(W)))

默认 alpha = beta = 0.5；使用 MinMaxScaler 将输入缩放到 [0, 1]
（decoder 末端是 Sigmoid，与官方实现一致）。

用法::

    # SWaT（官方超参：window=12, hidden=100, batch=7919, epochs=100, lr=1e-3）
    python -m models.reconstruction.usad --stage train \
        --csv 08_SWaT/SWaT_all_1s.csv

    python -m models.detect --artifact-dir runs/usad_swat

    # WADI（prefix 自动变为 usad_wadi）
    python -m models.reconstruction.usad --stage all \
        --csv 09_WADI/WADI_all_1s.csv
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

MODEL_NAME = "USAD"
ALPHA = 0.5   # 测试分数中 AE1 重构误差的权重
BETA = 0.5    # 测试分数中 AE2(AE1(W)) 重构误差的权重
HIDDEN_SIZE = 100  # 官方 notebook: z_size = window_size * hidden_size


class Encoder(nn.Module):
    """官方 Encoder：三层 Linear + ReLU。"""

    def __init__(self, in_size: int, latent_size: int):
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size / 2))
        self.linear2 = nn.Linear(int(in_size / 2), int(in_size / 4))
        self.linear3 = nn.Linear(int(in_size / 4), latent_size)
        self.relu = nn.ReLU(True)

    def forward(self, w):
        out = self.relu(self.linear1(w))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear3(out))
        return out


class Decoder(nn.Module):
    """官方 Decoder：三层 Linear + ReLU，末端 Sigmoid 输出 [0,1]。"""

    def __init__(self, latent_size: int, out_size: int):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size / 4))
        self.linear2 = nn.Linear(int(out_size / 4), int(out_size / 2))
        self.linear3 = nn.Linear(int(out_size / 2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.relu(self.linear1(z))
        out = self.relu(self.linear2(out))
        return self.sigmoid(self.linear3(out))


class UsadModel(nn.Module):
    """共享 Encoder + 两个 Decoder 的对抗自编码器。"""

    def __init__(self, seq_len: int, n_channels: int,
                 hidden_size: int = HIDDEN_SIZE, clip_input: bool = False):
        super().__init__()
        self.seq_len = int(seq_len)
        self.n_channels = int(n_channels)
        self.w_size = self.seq_len * self.n_channels
        self.z_size = self.seq_len * int(hidden_size)
        # 官方 MinMaxScaler 输出理论上在 [0,1]，但 WADI 攻击期存在大量
        # 超出 Normal 范围的值；Decoder 末端是 Sigmoid，若不裁剪会产生
        # 数百~数千的极端重构误差。裁剪只影响推理/评分（Normal 训练数据
        # 本身就在 [0,1]）。
        self.clip_input = bool(clip_input)
        self.encoder = Encoder(self.w_size, self.z_size)
        self.decoder1 = Decoder(self.z_size, self.w_size)
        self.decoder2 = Decoder(self.z_size, self.w_size)
        self._epoch = 1

    # ------------------------------------------------------------------
    # 基础前向
    # ------------------------------------------------------------------
    def _flat(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.size(0), -1)

    def _unflat(self, w: torch.Tensor) -> torch.Tensor:
        return w.view(w.size(0), self.seq_len, self.n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """AE1 的窗口重构（供共享框架的默认接口使用）。"""
        if self.clip_input:
            x = x.clamp(0.0, 1.0)
        return self._unflat(self.decoder1(self.encoder(self._flat(x))))

    # ------------------------------------------------------------------
    # 损失
    # ------------------------------------------------------------------
    def _losses(self, x: torch.Tensor, n: float):
        if self.clip_input:
            x = x.clamp(0.0, 1.0)
        batch = self._flat(x)
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))

        loss1 = ((1.0 / n) * F.mse_loss(w1, batch)
                 + (1.0 - 1.0 / n) * F.mse_loss(w3, batch))
        loss2 = ((1.0 / n) * F.mse_loss(w2, batch)
                 - (1.0 - 1.0 / n) * F.mse_loss(w3, batch))
        return loss1, loss2

    def training_loss(self, x: torch.Tensor) -> torch.Tensor:
        """单优化器兼容接口（共享框架默认训练循环用）。

        正式训练请使用 :meth:`fit`：USAD 需要两个优化器分别优化
        AE1/AE2，单优化器无法复现论文。
        """
        loss1, _ = self._losses(x, float(max(1, self._epoch)))
        return loss1

    def fit(self, loader, epochs: int, lr: float, device,
            name: str = "USAD", val_loader=None,
            history: list[dict] | None = None,
            zeroing: str = "official",
            val_every: int = 1) -> float:
        """官方双优化器训练循环。

        每一步依次做两次前向/反向（与官方 ``training()`` 一致）：
          1. 优化 Encoder+Decoder1，最小化 loss1；
          2. 用更新后的 Encoder 重新前向，优化 Encoder+Decoder2，
             最小化 loss2。

        ``zeroing="official"``（默认）严格复刻官方代码：
        ``optimizer1.zero_grad()`` 只清 Encoder+Decoder1，
        ``optimizer2.zero_grad()`` 只清 Encoder+Decoder2，
        因此两个 Decoder 会通过 :math:`AE_2(AE_1(W))` 互相保留一部分
        梯度。官方 notebook 的训练曲线依赖这一行为；改成完全清零
        （``zeroing="clean"``）会让对抗项发散更快。

        Parameters
        ----------
        zeroing:
            ``official``：与 manigalati/usad 的 ``training()`` 完全一致；
            ``clean``：每个子步骤前清空全部梯度（消融用）。
        """
        if zeroing not in ("official", "clean"):
            raise ValueError(f"未知 zeroing 模式: {zeroing}")
        self.to(device)
        opt1 = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder1.parameters()),
            lr=lr)
        opt2 = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder2.parameters()),
            lr=lr)
        start = time.time()

        for epoch in range(1, int(epochs) + 1):
            self.train()
            self._epoch = epoch
            n = float(epoch)
            total1 = total2 = 0.0
            seen = 0
            t0 = time.time()
            pbar = tqdm(loader, desc=f"[{name}] Epoch {epoch:03d}/{epochs}",
                        unit="batch", ncols=110, dynamic_ncols=True)
            for x, _ in pbar:
                x = x.to(device, non_blocking=True)

                # ---- 优化 AE1 (Encoder + Decoder1) ----
                if zeroing == "clean":
                    self.zero_grad(set_to_none=True)
                loss1, _ = self._losses(x, n)
                loss1.backward()
                opt1.step()
                opt1.zero_grad()      # 官方：只清 Encoder+Decoder1

                # ---- 优化 AE2 (Encoder + Decoder2) ----
                # 重新前向：Encoder 已被 opt1 更新
                if zeroing == "clean":
                    self.zero_grad(set_to_none=True)
                _, loss2 = self._losses(x, n)
                loss2.backward()
                opt2.step()
                opt2.zero_grad()      # 官方：只清 Encoder+Decoder2

                total1 += float(loss1.item()) * x.size(0)
                total2 += float(loss2.item()) * x.size(0)
                seen += x.size(0)
                pbar.set_postfix({
                    "L1": f"{loss1.item():.4f}",
                    "L2": f"{loss2.item():.4f}",
                })
            pbar.close()

            row = {
                "epoch": epoch,
                "loss1": total1 / max(seen, 1),
                "loss2": total2 / max(seen, 1),
                "epoch_time_s": time.time() - t0,
            }
            # 官方的 evaluate()：每个 epoch 在正常验证集上计算两个损失。
            # 验证不参与权重更新，val_every>1 只影响日志频率。
            if (val_loader is not None
                    and (val_every <= 1 or epoch % val_every == 0
                         or epoch == epochs)):
                v1, v2 = self._validate_losses(val_loader, n, device)
                row["val_loss1"] = v1
                row["val_loss2"] = v2
                print(f"[{name}] Epoch {epoch:03d} | "
                      f"L1 {row['loss1']:.6f} | L2 {row['loss2']:.6f} | "
                      f"val_L1 {v1:.6f} | val_L2 {v2:.6f}")
            else:
                print(f"[{name}] Epoch {epoch:03d} | "
                      f"L1 {row['loss1']:.6f} | L2 {row['loss2']:.6f}")
            if history is not None:
                history.append(row)

        return time.time() - start

    @torch.no_grad()
    def _validate_losses(self, loader, n: float, device) -> tuple[float, float]:
        self.eval()
        total1 = total2 = 0.0
        seen = 0
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            l1, l2 = self._losses(x, n)
            total1 += float(l1.item()) * x.size(0)
            total2 += float(l2.item()) * x.size(0)
            seen += x.size(0)
        return total1 / max(seen, 1), total2 / max(seen, 1)

    # ------------------------------------------------------------------
    # 异常分数
    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_batch(self, x: torch.Tensor) -> dict:
        """返回 USAD 官方异常分数及逐时刻/逐通道误差。

        - ``usad_score``：官方 testing() 的窗口分数
          ``alpha*MSE(W,AE1(W)) + beta*MSE(W,AE2(AE1(W)))``
        - ``usad_score_paper``：论文式 (5) 的窗口分数
          ``alpha*MSE(W,AE1(W)) + beta*MSE(W,AE2(W))``
        - ``usad_max``：逐时刻官方分数的窗口最大值
        - ``point_usad_last`` / ``point_usad_paper_last``：最后一点的两种分数
        - ``sq_ts``/``abs_ts``：AE1 的逐时刻平方/绝对误差
        - ``recon_last``/``true_last``：AE1 最后一点重构值/真值
        """
        x_in = x.clamp(0.0, 1.0) if self.clip_input else x
        batch = self._flat(x_in)
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)                 # 论文公式: AE2(W)
        w3 = self.decoder2(self.encoder(w1))  # 官方 testing: AE2(AE1(W))

        err1 = (w1 - batch).pow(2).view(x.size(0), self.seq_len,
                                        self.n_channels)
        err2 = (w2 - batch).pow(2).view(x.size(0), self.seq_len,
                                        self.n_channels)
        err3 = (w3 - batch).pow(2).view(x.size(0), self.seq_len,
                                        self.n_channels)
        # 官方 testing() 使用的分数
        ts = ALPHA * err1.mean(dim=-1) + BETA * err3.mean(dim=-1)  # [B,L]
        # 论文式 (5) 的分数 alpha*||W-AE1(W)|| + beta*||W-AE2(W)||
        ts_paper = ALPHA * err1.mean(dim=-1) + BETA * err2.mean(dim=-1)

        recon1 = self._unflat(w1)
        diff1 = recon1 - x_in
        abs1 = diff1.abs()
        sq1 = diff1 ** 2
        return {
            "window": {
                "usad_score": ts.mean(dim=1),
                "usad_max": ts.max(dim=1).values,
                "usad_score_paper": ts_paper.mean(dim=1),
            },
            "point": {
                "point_usad_last": ts[:, -1],
                "point_usad_max": ts.max(dim=1).values,
                "point_usad_paper_last": ts_paper[:, -1],
            },
            "ts": {
                "sq_ts": sq1.mean(dim=-1),
                "abs_ts": abs1.mean(dim=-1),
                "usad_ts": ts,
            },
            "channel": {
                "abs_last": abs1[:, -1, :],
                "sq_last": sq1[:, -1, :],
                "recon_last": recon1[:, -1, :],
                # 保留原始真值用于反变换绘图；误差本身按裁剪后的输入计算
                "true_last": x[:, -1, :],
            },
        }


def build_model(n_channels: int, seq_len: int, data=None,
                hidden_size: int = HIDDEN_SIZE,
                clip_input: bool = False) -> UsadModel:
    return UsadModel(seq_len, n_channels, hidden_size=hidden_size,
                     clip_input=clip_input)


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def _cli_has(flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])


def _prefix_from_csv(csv: str) -> str:
    name = str(csv).lower()
    if "wadi" in name:
        return "usad_wadi"
    return "usad_swat"


def _parse_usad_config() -> tuple[RunConfig, int, bool]:
    """在共享参数之外增加 USAD 特有的 --hidden-size / --clip-input。"""
    p = build_parser(
        "USAD 重构异常检测（SWaT / WADI）",
        batch_size=7919, epochs=100, lr=1e-3)
    p.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE,
                   help="z_size = seq_len * hidden_size；"
                        "官方 notebook 100，仓库 model.pth 对应 10（z=120）")
    p.add_argument("--clip-input", action="store_true",
                   help="推理时把 MinMax 后的输入裁剪到 [0,1]；"
                        "WADI 建议开启（Sigmoid Decoder 无法输出越界值）")
    args = p.parse_args()
    fields = set(RunConfig.__dataclass_fields__)
    cfg = RunConfig(**{k: v for k, v in vars(args).items() if k in fields})
    cfg.out_dir = Path(cfg.out_dir)
    if not _cli_has("--seq-len"):
        cfg.seq_len = 12          # 官方 window_size=12
    if not _cli_has("--scaler"):
        cfg.scaler = "minmax"     # 官方 MinMaxScaler
    if cfg.smoke:
        cfg.epochs = 1
    return cfg, int(args.hidden_size), bool(args.clip_input)


def main() -> None:
    # 官方 SWaT notebook 默认超参：batch=7919, epochs=100, Adam lr=1e-3
    cfg, hidden_size, clip_input = _parse_usad_config()
    prefix = _prefix_from_csv(cfg.csv)
    artifact_dir = Path(cfg.artifact_dir) if cfg.artifact_dir else (
        Path(cfg.out_dir) / "runs" / prefix)
    summary = {"model": MODEL_NAME, "prefix": prefix,
               "artifact_dir": str(artifact_dir), "stage": cfg.stage}
    print(f"[USAD] hidden_size={hidden_size} -> z_size="
          f"{cfg.seq_len * hidden_size} | scaler={cfg.scaler} | "
          f"clip_input={clip_input}")

    if cfg.stage in ("train", "all"):
        build = lambda n_channels, seq_len, data=None: build_model(
            n_channels, seq_len, data, hidden_size=hidden_size,
            clip_input=clip_input)
        artifact_dir = run_recon_stage(
            MODEL_NAME, prefix, build, cfg, save_model=True)
        summary["artifact_dir"] = str(artifact_dir)

    if cfg.stage in ("detect", "all"):
        run_detection_stage(
            artifact_dir, DetectionConfig.from_run_config(cfg))

    print(f"\n[{MODEL_NAME}] 完成: {summary}")


if __name__ == "__main__":
    main()
