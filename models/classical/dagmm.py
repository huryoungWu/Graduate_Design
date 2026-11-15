# -*- coding: utf-8 -*-
"""DAGMM：Deep Autoencoding Gaussian Mixture Model（PyTorch 复现）。

参考实现：https://github.com/tnakae/DAGMM
论文：Zong et al., *Deep Autoencoding Gaussian Mixture Model for
Unsupervised Anomaly Detection*, ICLR 2018.

参考实现是 TensorFlow 1.x；本文件用 PyTorch 逐项复刻：

- Compression Network：编码器 MLP（tanh）+ 解码器 MLP（tanh），
  隐变量 ``z_c`` 与两个重构误差特征
  ``[relative Euclidean distance, cosine distance]`` 拼接成
  ``z = [z_c, loss_E, loss_C]``；
- Estimation Network：MLP + dropout + softmax，输出 GMM 责任度 gamma；
- GMM：由当前 batch 的 z/gamma 估计 phi、mu、sigma（协方差）；
  energy 用 Cholesky 分解稳定计算，``-logsumexp`` 得到样本能量；
- 损失：``reconstruction + lambda1 * mean(energy)
  + lambda2 * sum(1/diag(sigma))``；
- 训练结束后用完整训练集重新估计并固定 GMM 参数，再计算 anomaly energy
  （能量越大越异常）。

原实现是逐样本（非时序）模型；本复现把每个时刻的传感器向量作为一个
样本，逐点计算 energy，再用 12 步滑动窗口聚合成窗口分数，与项目统一
评估协议对齐。

默认超参（KDDCup 示例/参考实现）::

    comp_hiddens=[60,30,10,1]
    est_hiddens=[10,4]        # 最后一维 = GMM 组件数
    dropout=0.5
    lr=1e-4, batch=1024, epochs=100
    lambda1=0.1, lambda2=1e-4

用法::

    python -m models.classical.dagmm --protocol framework \
        --csv 08_SWaT/SWaT_all_1s.csv
    python -m models.classical.dagmm --protocol framework \
        --csv 09_WADI/WADI_all_1s.csv

    python -m models.classical.dagmm --protocol paper \
        --csv 08_SWaT/SWaT_all_1s.csv
    python -m models.classical.dagmm --protocol paper \
        --csv 09_WADI/WADI_all_1s.csv
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from models.classical.isolation_forest import (
    _dataset_name, _window_outputs)
from models.common.artifacts import (ensure_dir, make_meta, save_meta,
                                     save_recon_split)
from models.common.config import (PROJECT_ROOT, RunConfig, build_parser,
                                  resolve_device, set_seed)
from models.common.data import load_swat
from models.common.detection import DetectionConfig, run_detection_stage
from models.reconstruction.usad_paper import (
    _plot_roc, _plot_score_hist, evaluate_scores, load_periods)

MODEL_NAME = "DAGMM"


# ----------------------------------------------------------------------
# 网络
# ----------------------------------------------------------------------
class CompressionNet(nn.Module):
    """编码器 + 解码器，输出 z_c 与重构误差特征。"""

    def __init__(self, n_features: int, comp_hiddens: list[int]):
        super().__init__()
        self.n_features = int(n_features)
        self.comp_hiddens = [int(h) for h in comp_hiddens]
        if len(self.comp_hiddens) < 2:
            raise ValueError("comp_hiddens 至少需要 2 层（含 latent）")

        enc_dims = [self.n_features] + self.comp_hiddens
        enc = []
        for i in range(len(enc_dims) - 1):
            enc.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:
                enc.append(nn.Tanh())
        self.encoder = nn.Sequential(*enc)

        dec_dims = [self.comp_hiddens[-1]] + self.comp_hiddens[-2::-1] + [
            self.n_features]
        dec = []
        for i in range(len(dec_dims) - 1):
            dec.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec.append(nn.Tanh())
        self.decoder = nn.Sequential(*dec)

    @staticmethod
    def _recon_features(x, x_dash):
        norm_x = x.norm(dim=1)
        norm_xd = x_dash.norm(dim=1)
        dist = (x - x_dash).norm(dim=1)
        dot = (x * x_dash).sum(dim=1)
        min_val = 1e-3
        loss_e = dist / (norm_x + min_val)
        loss_c = 0.5 * (1.0 - dot / (norm_x * norm_xd + min_val))
        return torch.stack([loss_e, loss_c], dim=1)

    def forward(self, x):
        z_c = self.encoder(x)
        x_dash = self.decoder(z_c)
        z_r = self._recon_features(x, x_dash)
        return torch.cat([z_c, z_r], dim=1), x_dash


class EstimationNet(nn.Module):
    """MLP + dropout + softmax，输出 GMM 责任度。"""

    def __init__(self, in_dim: int, est_hiddens: list[int],
                 dropout: float = 0.5):
        super().__init__()
        self.est_hiddens = [int(h) for h in est_hiddens]
        if not self.est_hiddens:
            raise ValueError("est_hiddens 不能为空")
        layers = []
        dim = int(in_dim)
        for h in self.est_hiddens[:-1]:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.Tanh())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            dim = h
        layers.append(nn.Linear(dim, self.est_hiddens[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        logits = self.net(z)
        return F.softmax(logits, dim=1)


class GMM:
    """DAGMM 的 GMM：phi/mu/sigma 由 z/gamma 估计，energy 用 Cholesky。"""

    def __init__(self, n_comp: int, eps: float = 1e-6):
        self.n_comp = int(n_comp)
        self.eps = float(eps)

    def estimate(self, z, gamma):
        # z: [B,D], gamma: [B,K]
        gamma_sum = gamma.sum(dim=0).clamp_min(self.eps)
        phi = gamma.mean(dim=0).clamp_min(self.eps)
        mu = (gamma.t() @ z) / gamma_sum[:, None]                 # [K,D]
        z_centered = z[:, None, :] - mu[None, :, :]                # [B,K,D]
        weighted = z_centered * gamma[:, :, None]                  # [B,K,D]
        sigma = torch.einsum("bkd,bke->kde", weighted, z_centered)
        sigma = sigma / gamma_sum[:, None, None]                   # [K,D,D]
        d = z.shape[1]
        eye = torch.eye(d, device=z.device, dtype=z.dtype)
        L = torch.linalg.cholesky(sigma + self.eps * eye[None, :, :])
        return phi, mu, sigma, L

    def energy(self, z, phi, mu, L):
        # z: [B,D]; phi [K], mu [K,D], L [K,D,D]
        B, D = z.shape
        K = phi.shape[0]
        z_centered = z[:, None, :] - mu[None, :, :]                # [B,K,D]
        v = []
        for k in range(K):
            rhs = z_centered[:, k, :].t()                          # [D,B]
            vk = torch.linalg.solve_triangular(
                L[k], rhs, upper=False)                            # [D,B]
            v.append(vk.t())                                       # [B,D]
        v = torch.stack(v, dim=1)                                  # [B,K,D]
        log_det = 2.0 * torch.log(
            torch.diagonal(L, dim1=1, dim2=2)).sum(dim=1)          # [K]
        logits = (torch.log(phi)[None, :]
                  - 0.5 * (v.pow(2).sum(dim=2)
                           + D * np.log(2.0 * np.pi)
                           + log_det[None, :]))                     # [B,K]
        return -torch.logsumexp(logits, dim=1)

    def cov_diag_loss(self, sigma):
        diag = torch.diagonal(sigma, dim1=1, dim2=2)               # [K,D]
        return (1.0 / diag).sum()


class DAGMM(nn.Module):
    """PyTorch DAGMM。"""

    def __init__(self, n_features: int,
                 comp_hiddens=(60, 30, 10, 1),
                 est_hiddens=(10, 4),
                 dropout: float = 0.5,
                 lambda1: float = 0.1,
                 lambda2: float = 1e-4):
        super().__init__()
        self.comp = CompressionNet(n_features, list(comp_hiddens))
        z_dim = int(comp_hiddens[-1]) + 2
        self.est = EstimationNet(z_dim, list(est_hiddens), dropout)
        self.gmm = GMM(int(est_hiddens[-1]))
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.z_dim = z_dim
        self.n_comp = int(est_hiddens[-1])
        self.register_buffer("phi", torch.full((self.n_comp,),
                                               1.0 / self.n_comp))
        self.register_buffer("mu", torch.zeros(self.n_comp, z_dim))
        self.register_buffer("sigma", torch.eye(z_dim).repeat(
            self.n_comp, 1, 1))
        self.register_buffer("L", torch.eye(z_dim).repeat(
            self.n_comp, 1, 1))
        self.fixed = False

    # ------------------------------------------------------------------
    def training_step(self, x):
        z, x_dash = self.comp(x)
        gamma = self.est(z)
        phi, mu, sigma, L = self.gmm.estimate(z, gamma)
        energy = self.gmm.energy(z, phi, mu, L)
        recon = ((x - x_dash) ** 2).sum(dim=1).mean()
        loss = (recon + self.lambda1 * energy.mean()
                + self.lambda2 * self.gmm.cov_diag_loss(sigma))
        return loss, recon.detach(), energy.detach()

    @torch.no_grad()
    def _collect_z_gamma(self, X: torch.Tensor, batch_size: int):
        zs, gs = [], []
        for s in range(0, X.size(0), batch_size):
            xb = X[s:s + batch_size]
            z, _ = self.comp(xb)
            gamma = self.est(z)
            zs.append(z)
            gs.append(gamma)
        return torch.cat(zs, dim=0), torch.cat(gs, dim=0)

    @torch.no_grad()
    def fix_gmm(self, X: torch.Tensor, batch_size: int) -> None:
        was_train = self.training
        self.eval()
        z, gamma = self._collect_z_gamma(X, batch_size)
        phi, mu, sigma, L = self.gmm.estimate(z, gamma)
        self.phi.copy_(phi)
        self.mu.copy_(mu)
        self.sigma.copy_(sigma)
        self.L.copy_(L)
        self.fixed = True
        if was_train:
            self.train()

    def fit(self, X: np.ndarray, epochs: int, batch_size: int, lr: float,
            device, name: str = "DAGMM",
            history: list[dict] | None = None) -> float:
        X_t = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32,
                              device=device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        start = time.time()
        n = X_t.size(0)
        for epoch in range(1, int(epochs) + 1):
            self.train()
            perm = torch.randperm(n, device=device)
            total = 0.0
            seen = 0
            pbar = tqdm(range(0, n, batch_size),
                        desc=f"[{name}] Epoch {epoch:03d}/{epochs}",
                        unit="batch", ncols=110, dynamic_ncols=True)
            for s in pbar:
                idx = perm[s:s + batch_size]
                xb = X_t[idx]
                opt.zero_grad(set_to_none=True)
                loss, recon, energy = self.training_step(xb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * xb.size(0)
                seen += xb.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.4f}",
                                  "recon": f"{recon.item():.4f}",
                                  "avg": f"{total / max(seen, 1):.4f}"})
            pbar.close()
            row = {"epoch": epoch, "loss": total / max(seen, 1)}
            if history is not None:
                history.append(row)
            if epoch <= 3 or epoch % 10 == 0 or epoch == epochs:
                print(f"[{name}] Epoch {epoch:03d} | "
                      f"loss {row['loss']:.5f}")
        self.fix_gmm(X_t, batch_size)
        return time.time() - start

    @torch.no_grad()
    def score_points(self, X, batch_size: int = 4096) -> np.ndarray:
        if not self.fixed:
            raise RuntimeError("GMM 尚未固定，请先训练或 fix_gmm")
        self.eval()
        if not torch.is_tensor(X):
            X = torch.as_tensor(np.ascontiguousarray(X),
                                dtype=torch.float32)
        device = self.phi.device
        energies = []
        for s in range(0, X.size(0), batch_size):
            xb = X[s:s + batch_size].to(device)
            z, _ = self.comp(xb)
            e = self.gmm.energy(z, self.phi, self.mu, self.L)
            energies.append(e.cpu())
        return torch.cat(energies, dim=0).numpy().astype(np.float64)


# ----------------------------------------------------------------------
# 统一框架协议
# ----------------------------------------------------------------------
def run_framework(cfg: RunConfig, args) -> None:
    prefix = f"dagmm_{_dataset_name(cfg.csv)}"
    artifact_dir = (Path(cfg.artifact_dir) if cfg.artifact_dir
                    else Path(cfg.out_dir) / "runs" / prefix)
    if cfg.stage == "detect":
        run_detection_stage(
            artifact_dir, DetectionConfig.from_run_config(cfg))
        return

    device = resolve_device(cfg.device)
    print(f"[{MODEL_NAME}] device={device} | 读取数据...")
    bundle = load_swat(cfg.csv, val_frac=cfg.val_frac, scaler="standard")
    n_train, n_val, n_test = (len(bundle.train_X), len(bundle.val_X),
                              len(bundle.test_X))
    print(f"[{MODEL_NAME}] 传感器数={bundle.n_channels} | "
          f"train={n_train:,} | val={n_val:,} | test={n_test:,}")
    val_ts_all = bundle.val_df["timestamp"].to_numpy()
    test_ts_all = bundle.test_df["timestamp"].to_numpy()
    sensor_cols = bundle.sensor_cols
    split_info = {
        "split_idx": int(bundle.split_idx),
        "val_end": int(bundle.val_end),
        "train_rows": n_train,
        "val_rows": n_val,
        "test_rows": n_test,
    }
    bundle.train_df = bundle.val_df = bundle.test_df = None

    model = DAGMM(
        n_features=bundle.n_channels,
        comp_hiddens=args.comp_hiddens,
        est_hiddens=args.est_hiddens,
        dropout=args.dropout,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{MODEL_NAME}] 参数量={n_params:,} | z_dim={model.z_dim} | "
          f"n_comp={model.n_comp}")

    history: list[dict] = []
    fit_time = model.fit(
        bundle.train_X, cfg.epochs, cfg.batch_size, cfg.lr, device,
        name=MODEL_NAME, history=history)
    print(f"[{MODEL_NAME}] 训练 + GMM 固定化耗时 {fit_time:.1f}s")
    bundle.train_X = None

    artifact_dir = ensure_dir(artifact_dir)
    meta = make_meta(
        model=MODEL_NAME, prefix=prefix, task="score",
        channels=sensor_cols, seq_len=cfg.seq_len,
        pred_len=1, horizon=1, val_frac=cfg.val_frac,
        split=split_info,
        train={
            "fit_time_s": float(fit_time),
            "epochs": int(cfg.epochs),
            "batch_size": int(cfg.batch_size),
            "lr": float(cfg.lr),
            "params": int(n_params),
            "comp_hiddens": list(args.comp_hiddens),
            "est_hiddens": list(args.est_hiddens),
            "dropout": float(args.dropout),
            "lambda1": float(args.lambda1),
            "lambda2": float(args.lambda2),
            "feature_mode": "point",
            "score_aggregation": ["mean", "max", "top3_mean"],
        },
        scaler_mean=bundle.scaler.mean_,
        scaler_scale=bundle.scaler.scale_,
        scaler_type="standard",
        extra={"score_aggregation": "point energy -> sliding window"},
    )
    save_meta(artifact_dir, meta)
    torch.save(model.state_dict(), artifact_dir / "model.pt")

    for split, X, labels, ts in (
            ("val", bundle.val_X, bundle.val_label, val_ts_all),
            ("test", bundle.test_X, bundle.test_label, test_ts_all)):
        ps = model.score_points(X, cfg.batch_size)
        win, wl, wts, pts, pl = _window_outputs(
            ps, labels, ts, cfg.seq_len, prefix="dagmm")
        save_recon_split(
            artifact_dir, split,
            label_window=wl, label_point=pl, timestamp=wts,
            window_scores=win, point_scores=pts)
        print(f"[{MODEL_NAME}] {split}: {len(wl):,} 个窗口，"
              f"正例率 {float(wl.mean()):.4f}")

    if history:
        pd.DataFrame(history).to_csv(
            artifact_dir / "training_history.csv", index=False,
            encoding="utf-8-sig")
    print(f"[{MODEL_NAME}] artifact 已保存: {artifact_dir}")

    if cfg.stage in ("all", "detect"):
        run_detection_stage(
            artifact_dir, DetectionConfig.from_run_config(cfg))


# ----------------------------------------------------------------------
# 全 Attack 段协议
# ----------------------------------------------------------------------
def run_paper(cfg: RunConfig, args) -> None:
    device = resolve_device(cfg.device)
    dataset = _dataset_name(cfg.csv)
    out_arg = Path(args.out_dir) if args.out_dir else None
    if out_arg is None or out_arg.resolve() == PROJECT_ROOT.resolve():
        out_dir = Path("runs") / f"dagmm_paper_{dataset}"
    else:
        out_dir = out_arg
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{MODEL_NAME}-paper] device={device} | dataset={dataset}")
    normal, attack, sensor_cols = load_periods(cfg.csv)
    print(f"[{MODEL_NAME}-paper] Normal {len(normal):,} | "
          f"Attack {len(attack):,} | 变量 {len(sensor_cols)}")

    scaler = StandardScaler()
    normal_X = scaler.fit_transform(
        normal[sensor_cols].to_numpy(np.float64)).astype(np.float32)
    attack_X = scaler.transform(
        attack[sensor_cols].to_numpy(np.float64)).astype(np.float32)
    attack_y = attack["label"].to_numpy(np.int8)
    attack_ts = attack["timestamp"].to_numpy()
    del normal, attack

    model = DAGMM(
        n_features=len(sensor_cols),
        comp_hiddens=args.comp_hiddens,
        est_hiddens=args.est_hiddens,
        dropout=args.dropout,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
    ).to(device)
    history: list[dict] = []
    if cfg.load_model:
        state = torch.load(cfg.load_model, map_location=device,
                           weights_only=True)
        model.load_state_dict(state)
        model.fixed = True
        fit_time = 0.0
        print(f"[{MODEL_NAME}-paper] 已加载权重: {cfg.load_model}")
    else:
        fit_time = model.fit(
            normal_X, cfg.epochs, cfg.batch_size, cfg.lr, device,
            name=f"{MODEL_NAME}-paper", history=history)
        print(f"[{MODEL_NAME}-paper] 训练 + GMM 固定化耗时 "
              f"{fit_time:.1f}s")
    del normal_X

    point_scores = model.score_points(attack_X, cfg.batch_size)
    del attack_X
    win, wl, wts, pts, pl = _window_outputs(
        point_scores, attack_y, attack_ts, cfg.seq_len, prefix="dagmm")
    results = {
        name: evaluate_scores(name, arr, wl, attack_y, cfg.seq_len)
        for name, arr in win.items()
    }

    metrics = {
        "dataset": dataset,
        "protocol": "full attack segment (oracle thresholds)",
        "model": MODEL_NAME,
        "seq_len": int(cfg.seq_len),
        "epochs": int(cfg.epochs),
        "batch_size": int(cfg.batch_size),
        "lr": float(cfg.lr),
        "comp_hiddens": list(args.comp_hiddens),
        "est_hiddens": list(args.est_hiddens),
        "dropout": float(args.dropout),
        "lambda1": float(args.lambda1),
        "lambda2": float(args.lambda2),
        "fit_time_s": float(fit_time),
        "n_test_windows": int(len(wl)),
        "scores": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(history).to_csv(
        out_dir / "training_history.csv", index=False,
        encoding="utf-8-sig")
    pd.DataFrame({
        "timestamp": pd.to_datetime(wts),
        **{f"score_{k}": v for k, v in win.items()},
        "score_point_dagmm": pts["point_dagmm"],
        "label_window": wl.astype(int),
    }).to_csv(out_dir / "window_scores.csv", index=False,
              encoding="utf-8-sig")

    if not args.no_plot:
        try:
            _plot_roc(wl, win["dagmm_mean"], out_dir / "roc.png",
                      f"[{MODEL_NAME}] {dataset.upper()} window-level ROC")
            _plot_score_hist(wl, win["dagmm_mean"],
                             out_dir / "score_hist.png",
                             f"[{MODEL_NAME}] {dataset.upper()} "
                             f"energy distribution")
        except Exception as exc:
            print(f"[{MODEL_NAME}-paper] 绘图跳过: {exc}")

    for name, res in results.items():
        w = res["window_level"]
        pa = res["point_level_PA"]
        print(f"[{MODEL_NAME}-paper] {name}: AUC {w['AUC']:.4f} | "
              f"F1@EER {w['F1_eer']:.4f} | "
              f"F1@oracle {w['F1_best_f1_oracle']:.4f} | "
              f"PA-F1@oracle {pa['F1_oracle']:.4f}")
    print(f"[{MODEL_NAME}-paper] 结果已保存: {out_dir}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_args():
    p = build_parser(
        "DAGMM 异常检测（SWaT / WADI）",
        batch_size=1024, epochs=100, lr=1e-4)
    p.add_argument("--protocol", choices=["framework", "paper"],
                   default="framework")
    p.add_argument("--comp-hiddens", default="60,30,10,1",
                   help="压缩网络隐藏层，最后一维为 latent 维度")
    p.add_argument("--est-hiddens", default="10,4",
                   help="估计网络隐藏层，最后一维为 GMM 组件数")
    p.add_argument("--dropout", type=float, default=0.5,
                   help="估计网络 dropout")
    p.add_argument("--lambda1", type=float, default=0.1,
                   help="energy 项权重")
    p.add_argument("--lambda2", type=float, default=1e-4,
                   help="协方差对角项权重")
    args = p.parse_args()

    args.comp_hiddens = [int(v) for v in args.comp_hiddens.split(",")
                         if v.strip()]
    args.est_hiddens = [int(v) for v in args.est_hiddens.split(",")
                        if v.strip()]

    fields = set(RunConfig.__dataclass_fields__)
    cfg = RunConfig(**{k: v for k, v in vars(args).items() if k in fields})
    cfg.out_dir = Path(cfg.out_dir)
    if not any(a == "--seq-len" or a.startswith("--seq-len=")
               for a in sys.argv[1:]):
        cfg.seq_len = 12
    if not any(a == "--scaler" or a.startswith("--scaler=")
               for a in sys.argv[1:]):
        cfg.scaler = "standard"
    if cfg.smoke:
        cfg.epochs = 1
    return cfg, args


def main() -> None:
    cfg, args = _parse_args()
    set_seed(int(args.seed))
    if args.protocol == "paper":
        run_paper(cfg, args)
    else:
        run_framework(cfg, args)


if __name__ == "__main__":
    main()
