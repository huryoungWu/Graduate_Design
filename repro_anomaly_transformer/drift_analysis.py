# -*- coding: utf-8 -*-
"""SWaT 数据漂移与模型分数漂移分析（20/80 协议）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "08_SWaT" / "SWaT_all_1s.csv"
OUT_CSV = ROOT / "runs" / "recon_comparison" / "DRIFT_ANALYSIS.csv"
OUT_MD = ROOT / "runs" / "recon_comparison" / "DRIFT_ANALYSIS.md"


def _raw_drift():
    df = pd.read_csv(CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feats = [c for c in df.columns
             if c not in ("timestamp", "label", "period")]
    period = df["period"].astype(str).str.lower()
    atk = df[period == "attack"].reset_index(drop=True)
    n_val = int(len(atk) * 0.2)
    val = atk.iloc[:n_val]
    test = atk.iloc[n_val:]
    normal = df[period == "normal"]

    def normal_rows(x):
        return x[x["label"].astype(int) == 0]

    segs = {
        "val_normal": normal_rows(val),
        "test_normal": normal_rows(test),
    }
    base = normal[feats].to_numpy(np.float64)
    base_mean = base.mean(axis=0)
    base_std = base.std(axis=0)
    base_std[base_std < 1e-8] = 1.0
    # KS 用抽样，避免大数组过慢
    rng = np.random.default_rng(42)
    base_sample = base[rng.choice(len(base), size=min(50000, len(base)),
                                  replace=False)]
    rows = []
    for name, seg in segs.items():
        x = seg[feats].to_numpy(np.float64)
        smd = (x.mean(axis=0) - base_mean) / base_std
        ks = np.array([ks_2samp(base_sample[:, j],
                                x[:, j]).statistic
                       for j in range(len(feats))])
        top = np.argsort(np.abs(smd))[-5:][::-1]
        rows.append({
            "segment": name, "n_normal_rows": len(x),
            "mean_abs_SMD": float(np.mean(np.abs(smd))),
            "median_abs_SMD": float(np.median(np.abs(smd))),
            "channels_abs_SMD_gt_0.5": int((np.abs(smd) > 0.5).sum()),
            "channels_abs_SMD_gt_1": int((np.abs(smd) > 1.0).sum()),
            "median_KS": float(np.median(ks)),
            "max_KS": float(np.max(ks)),
            "channels_KS_gt_0.1": int((ks > 0.1).sum()),
            "top_shifted_channels": ", ".join(
                f"{feats[j]}(SMD={smd[j]:.2f},KS={ks[j]:.3f})"
                for j in top),
        })
    return rows


def _score_drift():
    th = pd.read_csv(ROOT / "runs/normal99_full_attack/comparison.csv")
    th = th[(th.dataset == "SWaT") & (th.method == "q99_all")]
    th = th.set_index("model")["threshold"].to_dict()
    specs = [
        ("IsolationForest", ROOT / "runs/iforest_swat", "win_iforest_mean",
         float(th["IsolationForest"])),
        ("USAD", ROOT / "runs/usad_swat", "win_usad_score",
         float(th["USAD"])),
        ("LSTM-VAE", ROOT / "runs/lstmvae_swat", "win_recon_mean",
         float(th["LSTM-VAE"])),
    ]
    rows = []
    for name, path, key, thr in specs:
        out = {}
        for split in ("val", "test"):
            with np.load(Path(path) / f"{split}.npz",
                        allow_pickle=False) as z:
                s = np.asarray(z[key], dtype=np.float64)
                y = np.asarray(z["label_window"]).astype(int)
            normal = s[y == 0]
            out[split] = {
                "normal_scores": normal,
                "fpr": float((normal > thr).mean()),
            }
        rows.append({
            "model": name, "level": "window12", "threshold": thr,
            "val_normal_FPR": out["val"]["fpr"],
            "test_normal_FPR": out["test"]["fpr"],
            "val_normal_median": float(np.median(
                out["val"]["normal_scores"])),
            "test_normal_median": float(np.median(
                out["test"]["normal_scores"])),
            "test_normal_q90": float(np.quantile(
                out["test"]["normal_scores"], 0.90)),
            "test_normal_q99": float(np.quantile(
                out["test"]["normal_scores"], 0.99)),
            "test_q99_over_thr": float(np.quantile(
                out["test"]["normal_scores"], 0.99) / max(thr, 1e-12)),
        })

    # AT 12-step
    at_dir = ROOT / "repro_anomaly_transformer/runs/swat_20_80"
    va = np.load(at_dir / "val_energy.npy")
    te = np.load(at_dir / "test_energy.npy")
    ts = pd.read_csv(at_dir / "test_scores.csv")
    yp = ts["label"].to_numpy(np.int8)[:len(te)]
    w = 12
    va_w = np.lib.stride_tricks.sliding_window_view(va, w).mean(axis=1)
    te_w = np.lib.stride_tricks.sliding_window_view(te, w).mean(axis=1)
    yv_w = np.lib.stride_tricks.sliding_window_view(
        yp[:len(va)], w).max(axis=1)
    yt_w = np.lib.stride_tricks.sliding_window_view(yp, w).max(axis=1)
    nv = min(len(va_w), len(yv_w))
    nt = min(len(te_w), len(yt_w))
    va_w, yv_w = va_w[:nv], yv_w[:nv]
    te_w, yt_w = te_w[:nt], yt_w[:nt]
    tr = np.load(ROOT / "repro_anomaly_transformer/runs/swat/train_energy.npy")
    tr_w = np.lib.stride_tricks.sliding_window_view(tr, w).mean(axis=1)
    thr = float(np.percentile(tr_w, 99))
    vn = va_w[yv_w == 0]
    tn = te_w[yt_w == 0]
    rows.append({
        "model": "AnomalyTransformer(paper)", "level": "window12",
        "threshold": thr,
        "val_normal_FPR": float((vn > thr).mean()),
        "test_normal_FPR": float((tn > thr).mean()),
        "val_normal_median": float(np.median(vn)),
        "test_normal_median": float(np.median(tn)),
        "test_normal_q90": float(np.quantile(tn, 0.90)),
        "test_normal_q99": float(np.quantile(tn, 0.99)),
        "test_q99_over_thr": float(np.quantile(tn, 0.99) / max(thr, 1e-12)),
    })
    return rows


def main():
    raw = _raw_drift()
    score = _score_drift()
    raw_df = pd.DataFrame(raw)
    score_df = pd.DataFrame(score)
    raw_df.to_csv(ROOT / "runs/recon_comparison/DRIFT_RAW.csv",
                  index=False, encoding="utf-8-sig")
    score_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    lines = ["# SWaT 漂移分析", "",
             "## 1. 原始传感器特征漂移（相对 Normal 训练段）", ""]
    cols = ["segment", "n_normal_rows", "mean_abs_SMD", "median_abs_SMD",
            "channels_abs_SMD_gt_0.5", "channels_abs_SMD_gt_1",
            "median_KS", "max_KS", "channels_KS_gt_0.1"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, x in raw_df.iterrows():
        vals = []
        for c in cols:
            v = x[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "### 漂移最大的通道", ""]
    for _, x in raw_df.iterrows():
        lines.append(f"- {x['segment']}: {x['top_shifted_channels']}")

    lines += ["", "## 2. 模型分数的漂移（Normal-99 阈值）", ""]
    cols2 = ["model", "level", "threshold", "val_normal_FPR",
             "test_normal_FPR", "val_normal_median", "test_normal_median",
             "test_normal_q90", "test_normal_q99", "test_q99_over_thr"]
    lines.append("| " + " | ".join(cols2) + " |")
    lines.append("|" + "|".join(["---"] * len(cols2)) + "|")
    for _, x in score_df.iterrows():
        vals = []
        for c in cols2:
            v = x[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(raw_df[cols].to_string(index=False,
          float_format=lambda x: f"{x:.4f}"))
    print("\n", score_df[cols2].to_string(index=False,
          float_format=lambda x: f"{x:.6g}"))
    print(f"已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")


if __name__ == "__main__":
    main()
