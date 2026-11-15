# -*- coding: utf-8 -*-
"""目录内所有预测类/重构类模型的统一 12 步窗口基准。

冻结协议：
  - 训练：Normal only
  - 验证：Attack 前 20%，用于选阈值（val_best_f1）
  - 测试：Attack 后 80%，冻结；测试集不参与阈值
Oracle：
  - 使用测试集标签搜索最优阈值/方向，得到上限（不可部署）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score,
                             precision_recall_curve,
                             precision_recall_fscore_support,
                             roc_auc_score)

from models.common.protocol import select_threshold
from models.reconstruction.usad_extended_metrics import point_adjust

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "runs" / "recon_comparison" / "ALL_MODELS_BENCHMARK.csv"
OUT_MD = ROOT / "runs" / "recon_comparison" / "ALL_MODELS_BENCHMARK.md"

MODELS = [
    # forecasting
    ("LSTM", "lstm_swat", "forecast"),
    ("Transformer", "transformer_swat", "forecast"),
    ("iTransformer", "itransformer_swat", "forecast"),
    ("TCN", "tcn_swat", "forecast"),
    ("TimesNet", "timesnet_swat", "forecast"),
    ("MICN", "micn_swat", "forecast"),
    ("ModernTCN", "moderntcn_swat", "forecast"),
    ("Seq2Seq-LSTM", "seq2seq_lstm_swat", "forecast"),
    # reconstruction
    ("AE", "ae_swat", "reconstruction"),
    ("VAE", "vae_swat", "reconstruction"),
    ("AnomalyTransformer(project)", "anomalytransformer_swat",
     "reconstruction"),
    ("USAD", "usad_swat", "reconstruction"),
    ("LSTM-VAE", "lstmvae_swat", "reconstruction"),
    ("USAD(hidden=100)", "usad_h100", "reconstruction"),
    # classical reference
    ("IsolationForest", "iforest_swat", "classical"),
]


def _point_frame(artifact: str, split: str) -> pd.DataFrame:
    path = ROOT / "runs" / artifact
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    task = meta.get("task", "forecast")
    with np.load(path / f"{split}.npz", allow_pickle=False) as z:
        keys = set(z.keys())
        if task == "forecast":
            pred = np.asarray(z["pred"], dtype=np.float64)
            true = np.asarray(z["true"], dtype=np.float64)
            if pred.ndim == 3:
                pred = pred[:, 0, :]
                true = true[:, 0, :]
            score = ((pred - true) ** 2).mean(axis=1)
            label = np.asarray(z["label"])
            if label.ndim > 1:
                label = label[:, 0]
            label = label.astype(int)
        elif task == "score":
            score = np.asarray(z["pt_point_iforest"], dtype=np.float64)
            label = np.asarray(z["label_point"]).astype(int)
        else:
            if "pt_point_recon_mse_last" in keys:
                score = np.asarray(
                    z["pt_point_recon_mse_last"], dtype=np.float64)
            elif "ch_recon_last" in keys and "ch_true_last" in keys:
                diff = (np.asarray(z["ch_recon_last"], dtype=np.float64)
                        - np.asarray(z["ch_true_last"], dtype=np.float64))
                score = (diff ** 2).mean(axis=1)
            elif "ts_sq_ts" in keys:
                score = np.asarray(z["ts_sq_ts"], dtype=np.float64)[:, -1]
            else:
                raise KeyError(f"{artifact}: 找不到 point score")
            label = np.asarray(z["label_point"]).astype(int)
        ts = pd.to_datetime(z["timestamp"])
    n = min(len(score), len(label), len(ts))
    df = pd.DataFrame({"score": np.nan_to_num(score[:n]),
                       "label": label[:n]}, index=ts[:n])
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()


def _window12(df: pd.DataFrame, w: int = 12) -> tuple[pd.DataFrame, np.ndarray]:
    s = np.lib.stride_tricks.sliding_window_view(
        df["score"].to_numpy(), w).mean(axis=1)
    y = np.lib.stride_tricks.sliding_window_view(
        df["label"].to_numpy(), w).max(axis=1)
    return pd.DataFrame({"score": s, "label": y},
                        index=df.index[w - 1:]), y


def _metrics(y, s, thr):
    pred = (s > thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    neg = y == 0
    fpr = float(((pred == 1) & neg).sum() / max(neg.sum(), 1))
    auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan
    ap = (float(average_precision_score(y, s))
          if len(np.unique(y)) > 1 else np.nan)
    pa = point_adjust(y, pred)
    pp, pr, pf1, _ = precision_recall_fscore_support(
        y, pa, average="binary", zero_division=0)
    return {
        "Precision": float(p), "Recall": float(r), "F1": float(f1),
        "FPR": fpr, "AUC": auc, "PR_AUC": ap,
        "PA_Precision": float(pp), "PA_Recall": float(pr),
        "PA_F1": float(pf1),
    }, pred


def _oracle(y, s):
    best = None
    for sign in (+1, -1):
        z = sign * s
        prec, rec, thr = precision_recall_curve(y, z)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        k = int(np.argmax(f1[:-1])) if len(thr) else 0
        t = float(thr[k]) if len(thr) else float(np.max(z))
        cand = (f1[k], sign, t)
        if best is None or cand[0] > best[0]:
            best = cand
    _, sign, t = best
    return sign, t


def main():
    # ---------- 12-step frames ----------
    test_frames, val_data = {}, {}
    for name, artifact, family in MODELS:
        try:
            v, _ = _window12(_point_frame(artifact, "val"))
            t, _ = _window12(_point_frame(artifact, "test"))
            val_data[name] = (v["score"].to_numpy(),
                              v["label"].to_numpy(int))
            test_frames[name] = t
        except Exception as exc:
            print(f"[skip] {name}: {exc}")

    # AT paper：12-step energy
    at_dir = ROOT / "repro_anomaly_transformer/runs/swat_20_80"
    try:
        te = np.load(at_dir / "test_energy.npy")
        va = np.load(at_dir / "val_energy.npy")
        ts = pd.read_csv(at_dir / "test_scores.csv")
        yp = ts["label"].to_numpy(np.int8)[:len(te)]
        data_y = np.load(
            ROOT / "repro_anomaly_transformer/data/swat/test_y.npy")
        n_val = int(len(data_y) * 0.2)
        yv = data_y[:len(va)].astype(int)
        w = 12
        va_w = np.lib.stride_tricks.sliding_window_view(va, w).mean(axis=1)
        yv_w = np.lib.stride_tricks.sliding_window_view(yv, w).max(axis=1)
        te_w = np.lib.stride_tricks.sliding_window_view(te, w).mean(axis=1)
        yt_w = np.lib.stride_tricks.sliding_window_view(yp, w).max(axis=1)
        tsw = pd.to_datetime(ts["timestamp"])[:len(te)][w - 1:]
        nv = min(len(va_w), len(yv_w))
        nt = min(len(te_w), len(yt_w), len(tsw))
        val_data["AnomalyTransformer(paper)"] = (
            va_w[:nv], yv_w[:nv])
        test_frames["AnomalyTransformer(paper)"] = pd.DataFrame(
            {"score": te_w[:nt], "label": yt_w[:nt]},
            index=tsw[:nt])
        MODELS.append(("AnomalyTransformer(paper)",
                       "swat_20_80", "reconstruction"))
    except Exception as exc:
        print(f"[skip] AT paper: {exc}")

    # ---------- common test 12-step index ----------
    names = list(test_frames.keys())
    if not names:
        raise RuntimeError("没有可用模型")
    common = test_frames[names[0]].index
    for name in names[1:]:
        common = common.intersection(test_frames[name].index)
    common = pd.DatetimeIndex(sorted(common))
    print(f"common 12-step windows={len(common):,}")

    y_common = None
    data = {}
    for name in names:
        sub = test_frames[name].loc[common]
        lab = sub["label"].to_numpy(int)
        if y_common is None:
            y_common = lab
        elif not np.array_equal(y_common, lab):
            print(f"[warn] {name} labels differ on common index")
        data[name] = sub["score"].to_numpy(float)

    # ---------- evaluate ----------
    rows = []
    for name, artifact, family in MODELS:
        if name not in data:
            continue
        vs, vy = val_data[name]
        thr, vf = select_threshold(vs, vy)
        m, _ = _metrics(y_common, data[name], thr)
        sign, t_or = _oracle(y_common, data[name])
        mo, _ = _metrics(y_common, sign * data[name], t_or)
        rows.append({
            "model": name, "family": family, "level": "window12",
            "frozen_threshold": float(thr), "frozen_val_F1": float(vf),
            "frozen_Precision": m["Precision"],
            "frozen_Recall": m["Recall"], "frozen_F1": m["F1"],
            "frozen_FPR": m["FPR"], "frozen_AUC": m["AUC"],
            "frozen_PR_AUC": m["PR_AUC"], "frozen_PA_F1": m["PA_F1"],
            "oracle_threshold": float(t_or), "oracle_sign": int(sign),
            "oracle_Precision": mo["Precision"],
            "oracle_Recall": mo["Recall"], "oracle_F1": mo["F1"],
            "oracle_FPR": mo["FPR"], "oracle_PA_F1": mo["PA_F1"],
            "n": int(len(y_common)),
            "positives": int(y_common.sum()),
        })

    # ---------- SBT (window60) ----------
    sbt_dir = ROOT / "repro_anomaly_transformer/runs/sbt_20_80"
    if (sbt_dir / "metrics.json").exists():
        m = json.loads((sbt_dir / "metrics.json").read_text(
            encoding="utf-8"))
        d = pd.read_csv(sbt_dir / "test_scores.csv")
        s = d["score"].to_numpy(float)
        y = d["label_window"].to_numpy(int)
        # 找到 val 上选出的阈值/方向（metrics 中只保存了最终阈值）
        thr = float(m["threshold"])
        # 统一用窗口级 PA，和 12 步表保持一致
        pred_f = (s > thr).astype(int)
        pa_f = point_adjust(y, pred_f)
        pp_f, pr_f, pf_f, _ = precision_recall_fscore_support(
            y, pa_f, average="binary", zero_division=0)
        # 用 metrics 的 frozen 结果；oracle 用测试分数
        sign, t_or = _oracle(y, s)
        mo, _ = _metrics(y, sign * s, t_or)
        rows.append({
            "model": "SBT", "family": "reconstruction",
            "level": "window60",
            "frozen_threshold": thr, "frozen_val_F1": m.get("val_F1", np.nan),
            "frozen_Precision": m.get("Precision"),
            "frozen_Recall": m.get("Recall"),
            "frozen_F1": m.get("F1"), "frozen_FPR": m.get("FPR"),
            "frozen_AUC": m.get("AUC"), "frozen_PR_AUC": m.get("PR_AUC"),
            "frozen_PA_F1": float(pf_f),
            "oracle_threshold": float(t_or), "oracle_sign": int(sign),
            "oracle_Precision": mo["Precision"],
            "oracle_Recall": mo["Recall"], "oracle_F1": mo["F1"],
            "oracle_FPR": mo["FPR"], "oracle_PA_F1": mo["PA_F1"],
            "n": int(len(y)), "positives": int(y.sum()),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["family", "frozen_F1"],
                        ascending=[True, False])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    cols = ["model", "family", "level",
            "frozen_Precision", "frozen_Recall", "frozen_F1",
            "frozen_FPR", "frozen_AUC", "frozen_PR_AUC",
            "frozen_PA_F1",
            "oracle_Precision", "oracle_Recall", "oracle_F1",
            "oracle_FPR", "oracle_PA_F1"]
    lines = ["# 全部预测/重构模型基准（SWaT）", "",
             "- 训练：Normal only",
             "- 冻结阈值：Attack 前 20% 验证集 `val_best_f1`；"
             "测试集不参与阈值",
             "- 测试：Attack 后 80% 冻结",
             "- Oracle：使用测试标签搜索阈值/方向，仅作上限，不可部署",
             "- 12 步窗口（SBT 为 60 步窗口）",
             "- `PA_F1` 是窗口级 point-adjust", ""]
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
    print(df[cols].to_string(index=False,
                             float_format=lambda x: f"{x:.4f}"))
    print(f"已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")


if __name__ == "__main__":
    main()
