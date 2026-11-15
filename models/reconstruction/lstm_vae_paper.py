# -*- coding: utf-8 -*-
"""LSTM-VAE 的全 Attack 段评估（复用统一框架训练好的权重）。

该脚本不重新训练，只加载 ``runs/lstmvae_<dataset>/model.pt``，
按参考实现的 MinMax(-1,1) 预处理在整个 Attack 段上打分，输出：

- 窗口级 AUC / PR-AUC / EER 阈值 P-R-F1 / oracle 最优 F1；
- 逐点 raw F1 / point-adjusted F1；
- 每个窗口的分数 CSV 与 ROC/分数分布图。

用途是与文献中 LSTM-VAE 的 SWaT/WADI 指标对照；项目统一协议
（验证集选阈值）请直接看 ``runs/lstmvae_<dataset>/detection``。

用法::

    python -m models.reconstruction.lstm_vae_paper \
        --csv 08_SWaT/SWaT_all_1s.csv

    python -m models.reconstruction.lstm_vae_paper \
        --csv 09_WADI/WADI_all_1s.csv --clip-input
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from models.common.config import resolve_device, set_seed
from models.common.data import ReconstructionDataset
from models.reconstruction.lstm_vae import MODEL_NAME, LSTMVAE
from models.reconstruction.usad_paper import (
    _make_loader, _plot_roc, _plot_score_hist, evaluate_scores,
    load_periods)


def main() -> None:
    p = argparse.ArgumentParser(description="LSTM-VAE 全 Attack 段评估")
    p.add_argument("--csv", required=True)
    p.add_argument("--out-dir", default="",
                   help="为空时默认 runs/lstmvae_paper_<swat|wadi>")
    p.add_argument("--model", default="",
                   help="为空时默认 runs/lstmvae_<dataset>/model.pt")
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--intermediate-dim", type=int, default=15)
    p.add_argument("--z-dim", type=int, default=3)
    p.add_argument("--clip-input", action="store_true",
                   help="把输入裁剪到 [-1,1]（必须与训练/推理配置一致）")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = "wadi" if "wadi" in str(args.csv).lower() else "swat"
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path("runs") / f"lstmvae_paper_{dataset}")
    model_path = Path(args.model) if args.model else (
        Path("runs") / f"lstmvae_{dataset}" / "model.pt")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LSTMVAE-paper] dataset={dataset} | device={device}")
    normal, attack, sensor_cols = load_periods(args.csv)
    print(f"[LSTMVAE-paper] Normal {len(normal):,} | "
          f"Attack {len(attack):,} | 变量 {len(sensor_cols)}")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    normal_X = scaler.fit_transform(
        normal[sensor_cols].to_numpy(np.float64)).astype(np.float32)
    attack_X = scaler.transform(
        attack[sensor_cols].to_numpy(np.float64)).astype(np.float32)
    if args.clip_input:
        normal_X = np.clip(normal_X, -1.0, 1.0)
        attack_X = np.clip(attack_X, -1.0, 1.0)
        print("[LSTMVAE-paper] 输入已裁剪到 [-1,1]")

    normal_y = normal["label"].to_numpy(np.int8)
    attack_y = attack["label"].to_numpy(np.int8)
    attack_ts = attack["timestamp"].to_numpy()
    del normal, attack, normal_X

    test_ds = ReconstructionDataset(attack_X, attack_y, args.seq_len)
    test_loader = _make_loader(test_ds, args.batch_size, device)
    del attack_X

    model = LSTMVAE(
        len(sensor_cols), args.seq_len,
        intermediate_dim=args.intermediate_dim, z_dim=args.z_dim,
        clip_input=args.clip_input).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"[LSTMVAE-paper] 已加载权重: {model_path}")

    parts = {k: [] for k in ("recon_mean", "recon_max",
                             "point_recon_mse_last")}
    labels = []
    with torch.no_grad():
        for x, lab in test_loader:
            out = model.score_batch(x.to(device, non_blocking=True))
            parts["recon_mean"].append(
                out["window"]["recon_mean"].cpu().numpy())
            parts["recon_max"].append(
                out["window"]["recon_max"].cpu().numpy())
            parts["point_recon_mse_last"].append(
                out["point"]["point_recon_mse_last"].cpu().numpy())
            labels.append(np.asarray(lab))
    scores = {k: np.concatenate(v).astype(np.float64)
              for k, v in parts.items()}
    win_labels = np.concatenate(labels).astype(np.int8)
    n = len(test_ds)
    for k, v in scores.items():
        if len(v) != n:
            raise RuntimeError(f"{k}: 分数长度 {len(v)} != {n}")

    results = {name: evaluate_scores(name, arr, win_labels,
                                     attack_y, args.seq_len)
               for name, arr in scores.items()}
    primary = "recon_mean"
    metrics = {
        "dataset": dataset,
        "protocol": "full attack segment (oracle thresholds)",
        "model": MODEL_NAME,
        "seq_len": int(args.seq_len),
        "intermediate_dim": int(args.intermediate_dim),
        "z_dim": int(args.z_dim),
        "clip_input": bool(args.clip_input),
        "n_test_windows": int(n),
        "scores": results,
        "primary": primary,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    timestamps = attack_ts[args.seq_len - 1:]
    pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "score_recon_mean": scores["recon_mean"],
        "score_recon_max": scores["recon_max"],
        "score_point_recon_mse_last": scores["point_recon_mse_last"],
        "label_window": win_labels.astype(int),
    }).to_csv(out_dir / "window_scores.csv", index=False,
              encoding="utf-8-sig")

    if not args.no_plot:
        try:
            _plot_roc(win_labels, scores["recon_mean"],
                      out_dir / "roc.png",
                      f"[LSTM-VAE] {dataset.upper()} window-level ROC")
            _plot_score_hist(win_labels, scores["recon_mean"],
                             out_dir / "score_hist.png",
                             f"[LSTM-VAE] {dataset.upper()} score distribution")
        except Exception as exc:
            print(f"[LSTMVAE-paper] 绘图跳过: {exc}")

    for name, res in results.items():
        w = res["window_level"]
        pa = res["point_level_PA"]
        print(f"\n[LSTMVAE-paper] {name}: AUC {w['AUC']:.4f} | "
              f"winF1@EER {w['F1_eer']:.4f} | "
              f"winF1@oracle {w['F1_best_f1_oracle']:.4f} | "
              f"PA-F1@oracle {pa['F1_oracle']:.4f}")
    print(f"[LSTMVAE-paper] 结果已保存: {out_dir}")


if __name__ == "__main__":
    main()
