# -*- coding: utf-8 -*-
"""在完全相同的评估标准下比较 SWaT 重构模型。

统一标准
--------
1. 数据：同一份 SWaT_all_1s.csv，Normal 训练、Attack 前 20% 验证、
   后 80% 测试（项目统一 framework 协议）。
2. 时间轴：所有模型都取验证集/测试集时间戳的**交集**，保证逐点标签
   和评估区间完全一致。
3. 统一分数：``point_recon_mse_last``，即窗口最后时刻的通道平均平方
   重构误差。该分数对每个模型用同一个公式从 ``recon_last/true_last``
   重新计算，不依赖各模型自己的分数命名。
4. 统一阈值：验证集上最大化 F1（``val_best_f1``），测试集标签不参与
   阈值选择。
5. 两种统一评估粒度：
   - point    ：逐点分数 + 逐点标签；
   - window12 ：对逐点分数做 12 步滑动平均，窗口标签取 12 步内最大值。
6. 指标：AUC、PR-AUC、Precision、Recall、F1、FPR、FNR、Specificity、
   Balanced Accuracy、MCC、TP/FP/TN/FN；point 级额外给 PA-F1。

另外输出一张“各模型原生最佳分数”补充表：在同一 framework 划分上，
从模型自己可用的原生分数（recon_mean/usad_score/kl/fused 等）中按验证
集 F1 选最佳。该表用于参考，主表以统一分数表为准。

用法::

    python -I compare_reconstruction_models.py
    python -I compare_reconstruction_models.py --out-dir runs/recon_comparison
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.common.artifacts import load_recon_split
from models.common.protocol import evaluate_fixed, select_threshold
from models.common.scores import recon_point_candidates, recon_window_candidates
from models.reconstruction.usad_extended_metrics import point_adjust, prf

ROOT = Path(__file__).resolve().parent

MODELS = [
    ("AE", "ae_swat"),
    ("VAE", "vae_swat"),
    ("AnomalyTransformer", "anomalytransformer_swat"),
    ("USAD", "usad_swat"),
    ("LSTM-VAE", "lstmvae_swat"),
]

# 非重构参考基线：孤立森林；仅用同一时间戳和同一阈值规则对照，
# 不参与“重构模型”排名。
REFERENCE = [
    ("IsolationForest", "iforest_swat", "point_iforest"),
]

# 各模型“原生”分数；channel_norm_* 是检测阶段构造的派生分数，
# 不属于模型自身输出，故不纳入原生最佳分数比较。
NATIVE_SCORES = {
    "AE": {
        "recon_max", "recon_mean", "recon_top3_mean",
        "point_recon_mae_last", "point_recon_mse_last",
    },
    "VAE": {
        "recon_max", "recon_mean", "recon_top3_mean", "kl",
        "point_recon_mae_last", "point_recon_mse_last", "point_kl",
    },
    "AnomalyTransformer": {
        "recon_max", "recon_mean", "recon_top3_mean",
        "assoc_max", "fused",
        "point_recon_mae_last", "point_recon_mse_last", "point_assoc_last",
    },
    "USAD": {
        "recon_max", "recon_mean", "recon_top3_mean",
        "usad_score", "usad_score_paper", "usad_max",
        "point_recon_mae_last", "point_recon_mse_last",
        "point_usad_last", "point_usad_paper_last", "point_usad_max",
    },
    "LSTM-VAE": {
        "recon_mean", "recon_max", "recon_top3_mean", "kl",
        "point_recon_mse_last", "point_recon_mae_last", "point_kl",
    },
}

LEVELS = {
    "point": ("label_point", "label_point"),
    "window": ("label_window", "label_window"),
}


# ---------------------------------------------------------------------------
# 数据与统一分数
# ---------------------------------------------------------------------------
def _point_frame(artifact: str, split: str) -> pd.DataFrame:
    """统一逐点分数：窗口末时刻的通道平均平方重构误差。"""
    art = load_recon_split(ROOT / "runs" / artifact, split)
    recon = np.asarray(art["channel"]["recon_last"], dtype=np.float64)
    true = np.asarray(art["channel"]["true_last"], dtype=np.float64)
    score = ((recon - true) ** 2).mean(axis=1)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(art["timestamp"]),
        "label": np.asarray(art["label_point"]).astype(np.int8),
        "score": score,
    }).set_index("timestamp")
    if not df.index.is_unique:
        raise ValueError(f"{artifact}/{split}: 时间戳重复")
    return df.sort_index()


def _score_frame(artifact: str, split: str,
                 score_name: str) -> pd.DataFrame:
    """纯打分模型（如 IsolationForest）的逐点分数。"""
    art = load_recon_split(ROOT / "runs" / artifact, split)
    score = art["point_scores"][score_name]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(art["timestamp"]),
        "label": np.asarray(art["label_point"]).astype(np.int8),
        "score": np.asarray(score, dtype=np.float64),
    }).set_index("timestamp")
    if not df.index.is_unique:
        raise ValueError(f"{artifact}/{split}: 时间戳重复")
    return df.sort_index()


def _align(frames: dict[str, pd.DataFrame]):
    """取所有模型时间戳交集，并校验逐点标签一致。"""
    common = None
    for df in frames.values():
        idx = set(df.index)
        common = idx if common is None else common & idx
    common = pd.DatetimeIndex(sorted(common))
    aligned, ref_name, ref_label = {}, None, None
    for name, df in frames.items():
        sub = df.loc[common]
        label = sub["label"].to_numpy(np.int8)
        if ref_label is None:
            ref_name, ref_label = name, label
        elif not np.array_equal(label, ref_label):
            n_diff = int((label != ref_label).sum())
            raise ValueError(
                f"{name} 与 {ref_name} 的逐点标签不一致: {n_diff} 个点")
        aligned[name] = sub
    return aligned, ref_label


def _inverse_transform(values: np.ndarray, meta: dict) -> np.ndarray:
    """把标准化空间的重构值还原到原始物理量。"""
    scaler = meta.get("scaler") or {}
    mean = scaler.get("mean")
    scale = scaler.get("scale")
    values = np.asarray(values, dtype=np.float64)
    if mean is None or scale is None:
        return values
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if str(scaler.get("type", "")).startswith("minmax"):
        return (values - mean) / scale
    return values * scale + mean


def _recon_accuracy(name: str, artifact: str, meta: dict,
                    common_idx: pd.DatetimeIndex):
    """原始物理量下的重构精度（测试集公共时间轴）。

    返回 (summary, per_channel_rows)。nMAE/nRMSE 按每个通道真值的
    动态范围归一化，避免不同量纲直接平均。
    """
    art = load_recon_split(ROOT / "runs" / artifact, "test")
    ts = pd.to_datetime(art["timestamp"])
    recon = _inverse_transform(
        np.asarray(art["channel"]["recon_last"], dtype=np.float64), meta)
    true = _inverse_transform(
        np.asarray(art["channel"]["true_last"], dtype=np.float64), meta)
    pos = pd.Index(ts).get_indexer(common_idx)
    if (pos < 0).any():
        raise ValueError(f"{artifact}: 测试时间戳不包含公共区间")
    recon = recon[pos]
    true = true[pos]

    channels = list(meta["channels"])
    diff = recon - true
    abs_err = np.abs(diff)
    sq_err = diff ** 2
    mae = abs_err.mean(axis=0)
    rmse = np.sqrt(sq_err.mean(axis=0))
    rng = np.maximum(true.max(axis=0) - true.min(axis=0), 1e-9)
    nmae = mae / rng
    nrmse = rmse / rng

    mape = []
    for j in range(len(channels)):
        t = true[:, j]
        mask = np.abs(t) > 1e-6
        mape.append(float(np.mean(abs_err[mask, j] / np.abs(t[mask])) * 100)
                    if mask.any() else float("nan"))
    mape = np.asarray(mape, dtype=np.float64)

    per_channel = []
    for j, ch in enumerate(channels):
        per_channel.append({
            "model": name, "channel": ch,
            "MAE_raw": float(mae[j]),
            "RMSE_raw": float(rmse[j]),
            "MAPE_percent": float(mape[j]),
            "true_range": float(rng[j]),
            "nMAE_range": float(nmae[j]),
            "nRMSE_range": float(nrmse[j]),
        })

    summary = {
        "model": name,
        "n_channels": len(channels),
        "n_points": int(len(common_idx)),
        "MAE_mean_raw": float(np.mean(mae)),
        "RMSE_mean_raw": float(np.mean(rmse)),
        "MAE_median_raw": float(np.median(mae)),
        "RMSE_median_raw": float(np.median(rmse)),
        "nMAE_mean": float(np.mean(nmae)),
        "nRMSE_mean": float(np.mean(nrmse)),
        "nMAE_median": float(np.median(nmae)),
        "nRMSE_median": float(np.median(nrmse)),
        "nMAE_p90": float(np.quantile(nmae, 0.90)),
        "nRMSE_p90": float(np.quantile(nrmse, 0.90)),
        "MAPE_mean_percent": float(np.nanmean(mape)),
        "MAPE_median_percent": float(np.nanmedian(mape)),
    }
    return summary, per_channel


def _window12(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """逐点分数 -> 12 步滑窗均值；窗口标签 = 滑窗内标签最大值。"""
    score = df["score"].to_numpy(np.float64)
    label = df["label"].to_numpy(np.int8)
    ws = np.lib.stride_tricks.sliding_window_view(
        score, window).mean(axis=1)
    wy = np.lib.stride_tricks.sliding_window_view(
        label, window).max(axis=1)
    return pd.DataFrame({
        "score": ws, "label": wy,
    }, index=df.index[window - 1:])


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def _metric_row(*, model: str, dataset: str, protocol: str, level: str,
                score_name: str, seq_len: int, threshold: float,
                val_f1: float, score, labels) -> dict:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    ev = evaluate_fixed(score_name, score, labels, threshold,
                        "val_best_f1")
    alarm = (score >= float(threshold)).astype(np.int8)
    tp = int(((alarm == 1) & (labels == 1)).sum())
    fp = int(((alarm == 1) & (labels == 0)).sum())
    fn = int(((alarm == 0) & (labels == 1)).sum())
    tn = int(((alarm == 0) & (labels == 0)).sum())
    pos = tp + fn
    neg = tn + fp
    fpr = fp / neg if neg else float("nan")
    fnr = fn / pos if pos else float("nan")
    specificity = tn / neg if neg else float("nan")
    recall = ev["Recall"]
    bal_acc = 0.5 * (recall + specificity)
    mcc = (float(matthews_corrcoef(labels, alarm))
           if len(np.unique(labels)) > 1 else float("nan"))

    pa_f1 = float("nan")
    pa_precision = float("nan")
    pa_recall = float("nan")
    if level == "point":
        try:
            adjusted = point_adjust(labels, alarm)
            pa_precision, pa_recall, pa_f1 = prf(labels, adjusted)
        except Exception:
            pass

    return {
        "model": model,
        "dataset": dataset,
        "protocol": protocol,
        "level": level,
        "score": score_name,
        "seq_len": int(seq_len),
        "threshold": float(threshold),
        "val_F1": float(val_f1),
        "AUC": ev["AUC"],
        "PR_AUC": ev["PR_AUC"],
        "Precision": ev["Precision"],
        "Recall": ev["Recall"],
        "F1": ev["F1"],
        "FPR": fpr,
        "FNR": fnr,
        "Specificity": specificity,
        "BalancedAccuracy": bal_acc,
        "MCC": mcc,
        "PA_F1": pa_f1,
        "PA_Precision": pa_precision,
        "PA_Recall": pa_recall,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "n": int(len(labels)),
        "positives": int(pos),
    }


def _best_rows(df: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    """按验证集 F1 选每组最佳分数。"""
    idx = df.groupby(group, dropna=False)["val_F1"].idxmax()
    return df.loc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 原生最佳分数（补充表）
# ---------------------------------------------------------------------------
def _native_rows(name: str, artifact: str, dataset: str,
                 seq_len: int) -> list[dict]:
    val = load_recon_split(ROOT / "runs" / artifact, "val")
    test = load_recon_split(ROOT / "runs" / artifact, "test")
    val_scores = {
        "point": recon_point_candidates(val, ref_artifact=val),
        "window": recon_window_candidates(val),
    }
    test_scores = {
        "point": recon_point_candidates(test, ref_artifact=val),
        "window": recon_window_candidates(test),
    }
    labels = {
        "point": (np.asarray(val["label_point"]),
                  np.asarray(test["label_point"])),
        "window": (np.asarray(val["label_window"]),
                   np.asarray(test["label_window"])),
    }
    allowed = NATIVE_SCORES[name]
    rows = []
    for level in ("point", "window"):
        val_y, test_y = labels[level]
        for score_name, val_s in val_scores[level].items():
            if score_name not in allowed or score_name not in test_scores[level]:
                continue
            test_s = test_scores[level][score_name]
            # 某些候选分数长度可能与标签差 1，按最短长度截断
            n_val = min(len(val_s), len(val_y))
            n_test = min(len(test_s), len(test_y))
            thr, val_f1 = select_threshold(val_s[:n_val], val_y[:n_val])
            rows.append(_metric_row(
                model=name, dataset=dataset,
                protocol="native framework (val_best_f1)",
                level=level, score_name=score_name, seq_len=seq_len,
                threshold=thr, val_f1=val_f1,
                score=test_s[:n_test], labels=test_y[:n_test],
            ))
    return rows


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
SHOW = ["model", "level", "score", "threshold", "val_F1", "AUC",
        "PR_AUC", "Precision", "Recall", "F1", "FPR", "FNR",
        "BalancedAccuracy", "MCC", "PA_F1"]
ACC_SHOW = ["model", "n_channels", "n_points", "MAE_median_raw",
            "RMSE_median_raw", "nMAE_median", "nRMSE_median",
            "nMAE_p90", "nRMSE_p90", "MAPE_median_percent"]
SUMMARY_SHOW = ["model", "point_AUC", "point_PR_AUC", "point_F1",
                "point_FPR", "window12_AUC", "window12_F1",
                "window12_FPR", "nRMSE_median", "MAPE_median_percent",
                "native_score", "native_F1"]


def _write_markdown(path: Path, sections: list[tuple[str, pd.DataFrame]],
                    protocol_lines: list[str]) -> None:
    lines = ["# SWaT 重构模型统一评估对比", ""]
    lines += protocol_lines
    lines += [""]
    for item in sections:
        title, df = item[0], item[1]
        lines += [f"## {title}", ""]
        cols = (item[2] if len(item) > 2
                else [c for c in SHOW if c in df.columns])
        lines += ["| " + " | ".join(cols) + " |",
                  "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, row in df.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if isinstance(v, (float, np.floating)):
                    vals.append("nan" if np.isnan(v) else f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="SWaT 重构模型统一协议对比")
    p.add_argument("--out-dir", default="runs/recon_comparison")
    p.add_argument("--window", type=int, default=12,
                   help="统一窗口评估的滑窗长度（默认 12）")
    args = p.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_lens, metas, artifacts = {}, {}, {}
    val_frames, test_frames = {}, {}
    for name, artifact in MODELS:
        meta = json.loads(
            (ROOT / "runs" / artifact / "meta.json").read_text(
                encoding="utf-8"))
        seq_lens[name] = int(meta["seq_len"])
        metas[name] = meta
        artifacts[name] = artifact
        val_frames[name] = _point_frame(artifact, "val")
        test_frames[name] = _point_frame(artifact, "test")

    val_aligned, _ = _align(val_frames)
    test_aligned, test_labels = _align(test_frames)
    val_labels = next(iter(val_aligned.values()))["label"].to_numpy(np.int8)
    common_val_idx = next(iter(val_aligned.values())).index
    common_test_idx = next(iter(test_aligned.values())).index

    print("=== 统一评估区间 ===")
    print(f"val : {val_aligned[next(iter(val_aligned))].index[0]} ~ "
          f"{val_aligned[next(iter(val_aligned))].index[-1]} | "
          f"n={len(val_labels):,} | 正例率={val_labels.mean():.4f}")
    print(f"test: {test_aligned[next(iter(test_aligned))].index[0]} ~ "
          f"{test_aligned[next(iter(test_aligned))].index[-1]} | "
          f"n={len(test_labels):,} | 正例率={test_labels.mean():.4f}")

    # ---- 统一分数：point ----
    point_rows = []
    for name, _ in MODELS:
        val_s = val_aligned[name]["score"].to_numpy(np.float64)
        test_s = test_aligned[name]["score"].to_numpy(np.float64)
        thr, val_f1 = select_threshold(val_s, val_labels)
        point_rows.append(_metric_row(
            model=name, dataset="SWaT",
            protocol="unified point (common timestamps)",
            level="point", score_name="point_recon_mse_last",
            seq_len=seq_lens[name], threshold=thr, val_f1=val_f1,
            score=test_s, labels=test_labels,
        ))
    point_df = pd.DataFrame(point_rows).sort_values("F1", ascending=False)

    # ---- 统一分数：12 步窗口 ----
    window_rows = []
    for name, _ in MODELS:
        val_w = _window12(val_aligned[name], args.window)
        test_w = _window12(test_aligned[name], args.window)
        thr, val_f1 = select_threshold(
            val_w["score"].to_numpy(np.float64),
            val_w["label"].to_numpy(np.int8))
        window_rows.append(_metric_row(
            model=name, dataset="SWaT",
            protocol=f"unified {args.window}-step window (common timestamps)",
            level=f"window{args.window}",
            score_name="point_recon_mse_last",
            seq_len=seq_lens[name], threshold=thr, val_f1=val_f1,
            score=test_w["score"].to_numpy(np.float64),
            labels=test_w["label"].to_numpy(np.int8),
        ))
    window_df = pd.DataFrame(window_rows).sort_values("F1", ascending=False)

    # ---- 原始物理量下的重构精度（测试集公共时间轴） ----
    acc_rows, acc_channel_rows = [], []
    for name, _ in MODELS:
        summary, per_channel = _recon_accuracy(
            name, artifacts[name], metas[name], common_test_idx)
        acc_rows.append(summary)
        acc_channel_rows.extend(per_channel)
    acc_df = pd.DataFrame(acc_rows).sort_values("nRMSE_median")
    acc_channel_df = pd.DataFrame(acc_channel_rows)

    # ---- 非重构参考基线：同一时间轴 + 同一阈值规则 ----
    ref_rows = []
    for name, artifact, score_name in REFERENCE:
        meta = json.loads(
            (ROOT / "runs" / artifact / "meta.json").read_text(
                encoding="utf-8"))
        val_ref = _score_frame(artifact, "val", score_name).loc[common_val_idx]
        test_ref = _score_frame(
            artifact, "test", score_name).loc[common_test_idx]
        thr, val_f1 = select_threshold(
            val_ref["score"].to_numpy(np.float64),
            val_ref["label"].to_numpy(np.int8))
        row = _metric_row(
            model=name, dataset="SWaT",
            protocol="reference (same timestamps & val_best_f1)",
            level="point", score_name=score_name,
            seq_len=int(meta.get("seq_len", 12)),
            threshold=thr, val_f1=val_f1,
            score=test_ref["score"].to_numpy(np.float64),
            labels=test_ref["label"].to_numpy(np.int8),
        )
        row["type"] = "reference (non-reconstruction)"
        ref_rows.append(row)
    ref_df = pd.DataFrame(ref_rows).sort_values("F1", ascending=False)

    # ---- 原生最佳分数（补充） ----
    native_rows = []
    for name, artifact in MODELS:
        native_rows.extend(_native_rows(
            name, artifact, "SWaT", seq_lens[name]))
    native_all = pd.DataFrame(native_rows)
    native_level = _best_rows(native_all, ["model", "level"])
    native_primary = _best_rows(native_level, ["model"]).sort_values(
        "F1", ascending=False)

    # ---- 一页总览 ----
    summary_df = point_df[[
        "model", "AUC", "PR_AUC", "F1", "FPR"
    ]].rename(columns={
        "AUC": "point_AUC", "PR_AUC": "point_PR_AUC",
        "F1": "point_F1", "FPR": "point_FPR",
    })
    summary_df = summary_df.merge(
        window_df[["model", "AUC", "F1", "FPR"]].rename(columns={
            "AUC": f"window{args.window}_AUC",
            "F1": f"window{args.window}_F1",
            "FPR": f"window{args.window}_FPR",
        }), on="model")
    summary_df = summary_df.merge(
        acc_df[["model", "nRMSE_median", "MAPE_median_percent"]], on="model")
    summary_df = summary_df.merge(
        native_primary[["model", "score", "F1"]].rename(columns={
            "score": "native_score", "F1": "native_F1",
        }), on="model")
    summary_df = summary_df.sort_values("point_F1", ascending=False)
    summary_show = [c for c in SUMMARY_SHOW if c in summary_df.columns]

    # ---- 保存 ----
    point_df.to_csv(out_dir / "comparison_unified_point.csv",
                    index=False, encoding="utf-8-sig")
    window_df.to_csv(out_dir / "comparison_unified_window12.csv",
                     index=False, encoding="utf-8-sig")
    if len(ref_df):
        ref_df.to_csv(out_dir / "comparison_reference_iforest.csv",
                      index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "comparison_summary.csv",
                      index=False, encoding="utf-8-sig")
    acc_df.to_csv(out_dir / "comparison_recon_accuracy.csv",
                  index=False, encoding="utf-8-sig")
    acc_channel_df.to_csv(
        out_dir / "comparison_recon_accuracy_by_channel.csv",
        index=False, encoding="utf-8-sig")
    native_all.to_csv(out_dir / "comparison_native_all.csv",
                      index=False, encoding="utf-8-sig")
    native_level.to_csv(out_dir / "comparison_native_by_level.csv",
                        index=False, encoding="utf-8-sig")
    native_primary.to_csv(out_dir / "comparison_native_primary.csv",
                          index=False, encoding="utf-8-sig")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "protocol": {
                "dataset": "SWaT",
                "split": "Normal train / Attack 20% val / Attack 80% test",
                "common_val_timestamps": [
                    str(val_aligned[next(iter(val_aligned))].index[0]),
                    str(val_aligned[next(iter(val_aligned))].index[-1])],
                "common_test_timestamps": [
                    str(test_aligned[next(iter(test_aligned))].index[0]),
                    str(test_aligned[next(iter(test_aligned))].index[-1])],
                "unified_point_score": "point_recon_mse_last",
                "unified_window_score": (
                    f"{args.window}-step sliding mean of point score"),
                "threshold": "val_best_f1",
                "window_size": int(args.window),
            },
            "summary": summary_df.to_dict(orient="records"),
            "unified_point": point_rows,
            "unified_window": window_rows,
            "reference": ref_rows,
            "recon_accuracy": acc_rows,
            "native_all": native_rows,
        }, f, ensure_ascii=False, indent=2)

    protocol_lines = [
        "- 数据集：SWaT；Normal 训练，Attack 前 20% 验证，后 80% 测试。",
        f"- 验证集公共区间：{val_aligned[next(iter(val_aligned))].index[0]}"
        f" ~ {val_aligned[next(iter(val_aligned))].index[-1]}"
        f"（{len(val_labels):,} 点）。",
        f"- 测试集公共区间：{test_aligned[next(iter(test_aligned))].index[0]}"
        f" ~ {test_aligned[next(iter(test_aligned))].index[-1]}"
        f"（{len(test_labels):,} 点）。",
        "- 统一分数：窗口最后时刻的通道平均平方重构误差"
        "（`point_recon_mse_last`）。",
        f"- 统一窗口评估：对逐点分数做 {args.window} 步滑动平均，"
        f"窗口标签取 {args.window} 步内最大值。",
        "- 阈值：只在验证集上按 F1 选择（`val_best_f1`），测试集标签"
        "不参与任何选择。",
        "- 重构精度：在测试集（含攻击段）的公共时间轴上，先把 recon/true "
        "反标准化到原始物理量；nMAE/nRMSE 再按各通道真值的动态范围归一化。"
        "主表取跨通道中位数（p90 用于看尾部），避免少数近似恒定通道把均值放大。",
        "- 注意：AE/VAE/AnomalyTransformer 的模型窗口长度为 60，"
        "USAD/LSTM-VAE 为 12；这是模型自身结构差异，评估口径相同。",
        "- 未纳入：DCdetector、SBT（缺少可复现的 checkpoint/artifact，"
        "仅有旧 CSV，无法保证同一标准）。",
        "- 参考基线：IsolationForest 不是重构模型，仅用同一时间轴和同一"
        "阈值规则列出，不参与重构模型排名。",
    ]
    sections = [("一页总览（按统一逐点 F1 排序）", summary_df, summary_show),
                ("统一逐点评估（主表）", point_df),
                (f"统一 {args.window} 步窗口评估（主表）", window_df),
                ("重构精度（原始物理量，测试集含攻击段）",
                 acc_df, ACC_SHOW)]
    if len(ref_df):
        sections.append(("参考基线（非重构，不参与排名）", ref_df))
    sections.append(("各模型原生最佳分数（补充）", native_primary))
    _write_markdown(
        out_dir / "COMPARISON.md", sections, protocol_lines,
    )

    # ---- 控制台 ----
    print("\n=== 一页总览 ===")
    print(summary_df[summary_show].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== 统一逐点评估 ===")
    print(point_df[SHOW].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n=== 统一 {args.window} 步窗口评估 ===")
    print(window_df[SHOW].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== 重构精度（原始物理量，测试集公共时间轴） ===")
    print(acc_df[ACC_SHOW].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    if len(ref_df):
        print("\n=== 参考基线（非重构，不参与排名） ===")
        print(ref_df[SHOW].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== 各模型原生最佳分数（补充） ===")
    print(native_primary[SHOW].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n结果已保存: {out_dir}")


if __name__ == "__main__":
    main()
