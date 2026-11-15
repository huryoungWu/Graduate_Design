# -*- coding: utf-8 -*-
"""IF 与其它模型在 20/80 冻结测试 + Normal-99 下的融合实验。

只使用已保存的 12 步窗口分数，不重新训练、不碰测试集阈值：
  IF / USAD / LSTM-VAE / Anomaly-Transformer(paper)

融合方式：
  - 决策级：AND / OR / 多数投票
  - 分数级：按各自 Normal-99 阈值归一化后 mean / max / min
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)

from models.reconstruction.usad_extended_metrics import point_adjust

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "runs" / "recon_comparison" / "FUSION_IFOREST.csv"
OUT_MD = ROOT / "runs" / "recon_comparison" / "FUSION_IFOREST.md"


def _metrics(y, pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    neg = y == 0
    fpr = float(((pred == 1) & neg).sum() / max(neg.sum(), 1))
    pa = point_adjust(y, pred)
    pp, pr, pf1, _ = precision_recall_fscore_support(
        y, pa, average="binary", zero_division=0)
    return float(p), float(r), float(f1), fpr, float(pp), float(pr), float(pf1)


def _load_frames():
    specs = [
        ("IF", ROOT / "runs/iforest_swat/test.npz", "win_iforest_mean"),
        ("USAD", ROOT / "runs/usad_swat/test.npz", "win_usad_score"),
        ("LSTMVAE", ROOT / "runs/lstmvae_swat/test.npz", "win_recon_mean"),
    ]
    frames = []
    for name, path, key in specs:
        with np.load(path, allow_pickle=False) as z:
            ts = pd.to_datetime(z["timestamp"])
            df = pd.DataFrame({
                name: np.asarray(z[key], dtype=np.float64),
                "label": np.asarray(z["label_window"]).astype(int),
            }, index=ts)
        frames.append(df)

    # AT: 12 步窗口
    te = np.load(ROOT / "repro_anomaly_transformer/runs/swat_20_80/test_energy.npy")
    ts_df = pd.read_csv(
        ROOT / "repro_anomaly_transformer/runs/swat_20_80/test_scores.csv")
    ts = pd.to_datetime(ts_df["timestamp"])
    y = ts_df["label"].to_numpy(np.int8)
    w = 12
    te_w = np.lib.stride_tricks.sliding_window_view(te, w).mean(axis=1)
    y_w = np.lib.stride_tricks.sliding_window_view(y, w).max(axis=1)
    frames.append(pd.DataFrame({"AT": te_w, "label": y_w},
                               index=ts[w - 1:]))
    return frames


def main():
    frames = _load_frames()
    common = frames[0].index
    for f in frames[1:]:
        common = common.intersection(f.index)
    common = pd.DatetimeIndex(sorted(common))
    data = {}
    labels = None
    for f in frames:
        sub = f.loc[common]
        data[f.columns[0]] = sub.iloc[:, 0].to_numpy(float)
        lab = sub["label"].to_numpy(int)
        if labels is None:
            labels = lab
        elif not np.array_equal(labels, lab):
            print("[warning] labels differ; using first frame labels")
    print(f"common windows={len(common):,} | positives={labels.sum():,}")

    # Normal-99 阈值
    th = pd.read_csv(ROOT / "runs/normal99_full_attack/comparison.csv")
    th = th[(th.dataset == "SWaT") & (th.method == "q99_all")]
    th = th.set_index("model")["threshold"].to_dict()
    tr = np.load(ROOT / "repro_anomaly_transformer/runs/swat/train_energy.npy")
    at_thr = float(np.percentile(
        np.lib.stride_tricks.sliding_window_view(tr, 12).mean(axis=1), 99))
    thresholds = {
        "IF": float(th["IsolationForest"]),
        "USAD": float(th["USAD"]),
        "LSTMVAE": float(th["LSTM-VAE"]),
        "AT": at_thr,
    }
    print("thresholds", thresholds)

    preds = {k: (data[k] > thresholds[k]).astype(int) for k in data}
    rules = {}
    for k in ["IF", "USAD", "LSTMVAE", "AT"]:
        rules[k] = preds[k]
    for base in ["USAD", "LSTMVAE", "AT"]:
        rules[f"IF_AND_{base}"] = preds["IF"] & preds[base]
        rules[f"IF_OR_{base}"] = preds["IF"] | preds[base]
    rules["IF_AND_ANY"] = preds["IF"] & (
        preds["USAD"] | preds["LSTMVAE"] | preds["AT"])
    rules["MAJORITY_2"] = (
        preds["IF"] + preds["USAD"] + preds["LSTMVAE"] + preds["AT"]) >= 2
    rules["MAJORITY_3"] = (
        preds["IF"] + preds["USAD"] + preds["LSTMVAE"] + preds["AT"]) >= 3

    # 两阶段：IF 为主；基础模型只在“非常强”或“IF 接近阈值”时补充
    for base in ["USAD", "LSTMVAE", "AT"]:
        for c in [2, 5, 10]:
            rules[f"IF_OR_{base}_x{c}"] = (
                preds["IF"] | (data[base] > c * thresholds[base]))
        for a in [0.8, 0.9, 0.95]:
            rules[f"IF_OR_{base}_if{a}"] = (
                preds["IF"] | (preds[base]
                               & (data["IF"] > a * thresholds["IF"])))
    strong = ((data["USAD"] > 5 * thresholds["USAD"])
              | (data["LSTMVAE"] > 5 * thresholds["LSTMVAE"])
              | (data["AT"] > 5 * thresholds["AT"]))
    rules["IF_OR_ANY_x5"] = preds["IF"] | strong

    rows = []
    for name, pred in rules.items():
        p, r, f1, fpr, pp, pr, pf1 = _metrics(labels, pred.astype(int))
        rows.append({
            "rule": name, "type": "decision",
            "AUC": np.nan, "PR_AUC": np.nan,
            "Precision": p, "Recall": r, "F1": f1, "FPR": fpr,
            "PA_Precision_win": pp, "PA_Recall_win": pr,
            "PA_F1_win": pf1,
        })

    # 分数级：按各自阈值归一化
    norm = {k: data[k] / max(thresholds[k], 1e-12) for k in data}
    fused = {
        "IF_USAD_mean": 0.5 * (norm["IF"] + norm["USAD"]),
        "IF_USAD_max": np.maximum(norm["IF"], norm["USAD"]),
        "IF_USAD_min": np.minimum(norm["IF"], norm["USAD"]),
        "IF_LSTMVAE_mean": 0.5 * (norm["IF"] + norm["LSTMVAE"]),
        "IF_LSTMVAE_max": np.maximum(norm["IF"], norm["LSTMVAE"]),
        "IF_AT_mean": 0.5 * (norm["IF"] + norm["AT"]),
        "IF_AT_max": np.maximum(norm["IF"], norm["AT"]),
        "ALL_mean": 0.25 * (norm["IF"] + norm["USAD"]
                            + norm["LSTMVAE"] + norm["AT"]),
        "ALL_max": np.maximum.reduce(
            [norm["IF"], norm["USAD"], norm["LSTMVAE"], norm["AT"]]),
    }
    for name, s in fused.items():
        auc = (float(roc_auc_score(labels, s))
               if len(np.unique(labels)) > 1 else np.nan)
        ap = (float(average_precision_score(labels, s))
              if len(np.unique(labels)) > 1 else np.nan)
        # 用 1.0 作为归一化分数的启发式阈值（各模型 Normal-99 对应 1）
        pred = (s > 1.0).astype(int)
        p, r, f1, fpr, pp, pr, pf1 = _metrics(labels, pred)
        rows.append({
            "rule": name, "type": "score",
            "AUC": auc, "PR_AUC": ap,
            "Precision": p, "Recall": r, "F1": f1, "FPR": fpr,
            "PA_Precision_win": pp, "PA_Recall_win": pr,
            "PA_F1_win": pf1,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["type", "F1"], ascending=[True, False])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    cols = ["rule", "type", "AUC", "PR_AUC", "Precision", "Recall",
            "F1", "FPR", "PA_F1_win"]
    lines = ["# IF 融合实验（Normal-99 + 20/80，SWaT）", "",
             "- 测试集冻结；阈值来自 Normal-99；未重新训练",
             "- 12 步窗口；IF/USAD/LSTMVAE/AT 各自 Normal-99 阈值",
             "- `PA_F1_win` 是 12 步窗口级 point-adjust，"
             "与之前逐点 PA-F1 不完全同口径", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, x in df.iterrows():
        vals = []
        for c in cols:
            v = x[c]
            if isinstance(v, (float, np.floating)):
                vals.append("nan" if np.isnan(v) else f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")


if __name__ == "__main__":
    main()
