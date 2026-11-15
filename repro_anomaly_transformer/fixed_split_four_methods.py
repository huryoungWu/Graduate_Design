# -*- coding: utf-8 -*-
"""固定划分（非 k-fold）下四种方法的对比。

划分：前 50% 攻击段训练、接下来 20% 验证、最后 30% 测试。

- U_N99：Normal-99 阈值，不用异常数据
- U_CAL：训练攻击窗口选阈值，异常数据只用于校准
- SUP_HGB：原始窗口统计 + 异常数据训练 HGB（全局分类器）
- SUP_HGB_oracle：测试集选阈值，仅作上限
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)
from torch.utils.data import DataLoader

from models.common.config import resolve_device
from models.common.protocol import select_threshold
from models.reconstruction.sbt import build_model as build_sbt
from models.reconstruction.usad_extended_metrics import point_adjust

from .benchmark_all_models import _point_frame, _window12
from .dataset import WindowDataset, load_prepared
from .supervised_fair import _raw_stats

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "08_SWaT" / "SWaT_all_1s.csv"
SEG = ROOT / "08_SWaT" / "SWaT_attack_segments.csv"
OUT_CSV = ROOT / "runs" / "recon_comparison" / "FIXED_SPLIT_FOUR_METHODS.csv"
OUT_MD = ROOT / "runs" / "recon_comparison" / "FIXED_SPLIT_FOUR_METHODS.md"

PROJECT_MODELS = [
    ("LSTM", "lstm_swat"), ("Transformer", "transformer_swat"),
    ("iTransformer", "itransformer_swat"), ("TCN", "tcn_swat"),
    ("TimesNet", "timesnet_swat"), ("MICN", "micn_swat"),
    ("ModernTCN", "moderntcn_swat"), ("Seq2Seq-LSTM", "seq2seq_lstm_swat"),
    ("AE", "ae_swat"), ("VAE", "vae_swat"),
    ("AnomalyTransformer(project)", "anomalytransformer_swat"),
]
NATIVE_MODELS = [
    ("USAD", "usad_swat", "win_usad_score"),
    ("LSTM-VAE", "lstmvae_swat", "win_recon_mean"),
    ("IsolationForest", "iforest_swat", "win_iforest_mean"),
]


def _prf(y, p):
    p0, r0, f0, _ = precision_recall_fscore_support(
        y, p, average="binary", zero_division=0)
    return float(p0), float(r0), float(f0)


def _metrics(y, s, thr):
    pred = (s > thr).astype(int)
    p, r, f = _prf(y, pred)
    neg = y == 0
    fpr = float(((pred == 1) & neg).sum() / max(neg.sum(), 1))
    auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan
    ap = (float(average_precision_score(y, s))
          if len(np.unique(y)) > 1 else np.nan)
    pa = point_adjust(y, pred)
    _, _, pf = _prf(y, pa)
    return dict(Precision=p, Recall=r, F1=f, FPR=fpr,
                AUC=auc, PR_AUC=ap, PA_F1=pf)


def _full_from_val_test(val, test):
    df = pd.concat([val, test])
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def _native_frame(artifact, key):
    frames = {}
    for split in ("val", "test"):
        with np.load(ROOT / "runs" / artifact / f"{split}.npz",
                    allow_pickle=False) as z:
            frames[split] = pd.DataFrame(
                {"score": np.asarray(z[key], dtype=np.float64),
                 "label": np.asarray(z["label_window"]).astype(int)},
                index=pd.to_datetime(z["timestamp"]))
    return _full_from_val_test(frames["val"], frames["test"])


def _split_by_segments(df, win, val_start, test_start):
    start = df.index - pd.Timedelta(seconds=win - 1)
    return np.where(start < val_start, "train",
                    np.where(start < test_start, "val", "test"))


def main():
    device = resolve_device("cuda")
    full = {}
    thresholds = {}

    # 项目内预测/重构：point error 12-step
    proj_th = pd.read_csv(ROOT / "runs/normal99_project_models/comparison.csv")
    proj_th = proj_th[proj_th.method == "q99_all"].set_index("model")[
        "threshold"].to_dict()
    for name, artifact in PROJECT_MODELS:
        try:
            v, _ = _window12(_point_frame(artifact, "val"))
            t, _ = _window12(_point_frame(artifact, "test"))
            full[name] = _full_from_val_test(v, t)
            key_name = ("AnomalyTransformer"
                        if name == "AnomalyTransformer(project)" else name)
            thresholds[name] = float(proj_th[key_name])
        except Exception as exc:
            print(f"[skip] {name}: {exc}")

    # 原生窗口分数
    th = pd.read_csv(ROOT / "runs/normal99_full_attack/comparison.csv")
    th = th[(th.dataset == "SWaT") & (th.method == "q99_all")]
    th = th.set_index("model")["threshold"].to_dict()
    for name, artifact, key in NATIVE_MODELS:
        try:
            full[name] = _native_frame(artifact, key)
            thresholds[name] = float(th[name])
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
        # val 的时间戳来自完整 Attack 段前 20%
        data_ts = np.load(
            ROOT / "repro_anomaly_transformer/data/swat/test_timestamp.npy")
        v_ts = pd.to_datetime(data_ts)[:len(va)][w - 1:][:nv]
        val = pd.DataFrame({"score": va_w[:nv], "label": yv_w[:nv]},
                           index=v_ts)
        test = pd.DataFrame({"score": te_w[:nt], "label": yt_w[:nt]},
                            index=tsw[:nt])
        full["AnomalyTransformer(paper)"] = _full_from_val_test(val, test)
        tr = np.load(
            ROOT / "repro_anomaly_transformer/runs/swat/train_energy.npy")
        tr_w = np.lib.stride_tricks.sliding_window_view(tr, w).mean(axis=1)
        thresholds["AnomalyTransformer(paper)"] = float(
            np.percentile(tr_w, 99))
    except Exception as exc:
        print(f"[skip] AT paper: {exc}")

    # SBT：window60；test 分数已有，val 分数用 checkpoint 现算
    try:
        sbt_dir = (ROOT / "repro_anomaly_transformer/runs/sbt_20_80")
        test_s = pd.read_csv(sbt_dir / "test_scores.csv")
        test_s = pd.DataFrame({
            "score": test_s["score"].to_numpy(float),
            "label": test_s["label_window"].to_numpy(int),
        }, index=pd.to_datetime(test_s["timestamp"]))
        # val 20% 分数
        data = load_prepared(ROOT / "repro_anomaly_transformer/data/swat")
        n_val = int(len(data["test_y"]) * 0.2)
        val_X = data["test_X"][:n_val]
        val_y = data["test_y"][:n_val].astype(int)
        model = build_sbt(val_X.shape[1], 60).to(device)
        model.load_state_dict(torch.load(
            sbt_dir / "sbt_checkpoint.pth", map_location=device,
            weights_only=True))
        model.eval()
        ds = WindowDataset(val_X, val_y, win_size=60, step=1, mode="val")
        loader = DataLoader(ds, batch_size=128, shuffle=False,
                            num_workers=0, pin_memory=True)
        vals = []
        with torch.no_grad():
            for xb, _ in loader:
                out = model.score_batch(xb.to(device, non_blocking=True))
                vals.append(out["window"]["score_max"].cpu().numpy())
        vs = np.concatenate(vals).reshape(-1)
        vy = np.lib.stride_tricks.sliding_window_view(
            val_y, 60).max(axis=1)
        n = min(len(vs), len(vy))
        v_ts = pd.to_datetime(data["test_timestamp"])[:n_val][59:][:n]
        val_s = pd.DataFrame({"score": vs[:n], "label": vy[:n]},
                             index=v_ts)
        full["SBT"] = _full_from_val_test(val_s, test_s)
        thresholds["SBT"] = 0.00120511
    except Exception as exc:
        print(f"[skip] SBT: {exc}")

    # 固定 20/80：前 20% 攻击段训练/校准，后 80% 冻结测试
    _meta = pd.read_csv(CSV, usecols=["timestamp", "period"])
    _meta["timestamp"] = pd.to_datetime(_meta["timestamp"])
    _atk_ts = _meta.loc[
        _meta["period"].astype(str).str.lower() == "attack",
        "timestamp"].to_numpy()
    n_val = int(len(_atk_ts) * 0.2)
    test_start = pd.Timestamp(_atk_ts[n_val])
    train_start = pd.Timestamp(_atk_ts[0])
    val_start_hgb = train_start + 0.7 * (test_start - train_start)
    print(f"20/80 fixed split | train<{test_start} | test>={test_start}")

    rows = []
    for name, df in full.items():
        tr = df[df.index < test_start]
        te = df[df.index >= test_start]
        if len(te) == 0 or len(tr) == 0:
            continue
        # U_N99
        thr_n = thresholds.get(name, np.nan)
        if np.isfinite(thr_n):
            m = _metrics(te["label"].to_numpy(int),
                         te["score"].to_numpy(float), thr_n)
            rows.append({"model": name, "method": "U_N99",
                         "threshold": thr_n, **m})
        # U_CAL
        thr_c, _ = select_threshold(tr["score"].to_numpy(float),
                                    tr["label"].to_numpy(int))
        m = _metrics(te["label"].to_numpy(int),
                     te["score"].to_numpy(float), thr_c)
        rows.append({"model": name, "method": "U_CAL",
                     "threshold": thr_c, **m})

    # SUP_HGB：原始窗口统计（固定划分）
    df = pd.read_csv(CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feats = [c for c in df.columns
             if c not in ("timestamp", "label", "period")]
    period = df["period"].astype(str).str.lower()
    normal = df[period == "normal"].reset_index(drop=True)
    attack = df[period == "attack"].reset_index(drop=True)
    Xn = normal[feats].to_numpy(np.float32)
    Xa = attack[feats].to_numpy(np.float32)
    sn = _raw_stats(Xn)
    sa = _raw_stats(Xa)
    cols = [f"{st}_{c}" for st in ("mean", "std", "last", "diff")
            for c in feats]
    nidx = pd.to_datetime(normal["timestamp"]).iloc[11:]
    aidx = pd.to_datetime(attack["timestamp"]).iloc[11:]
    nlab = np.zeros(len(sn), np.int8)
    alab = np.lib.stride_tricks.sliding_window_view(
        attack["label"].to_numpy(np.int8), 12).max(axis=1)
    f_n = pd.DataFrame(sn, columns=cols, index=nidx)
    f_a = pd.DataFrame(sa, columns=cols, index=aidx)
    f_a["label"] = alab
    split = np.where(
        f_a.index < val_start_hgb, "train",
        np.where(f_a.index < test_start, "val", "test"))
    f_a["split"] = split
    X_train = np.concatenate([
        f_n.to_numpy(np.float32),
        f_a[f_a.split == "train"][cols].to_numpy(np.float32)])
    y_train = np.concatenate([
        nlab, f_a[f_a.split == "train"]["label"].to_numpy(int)])
    X_val = f_a[f_a.split == "val"][cols].to_numpy(np.float32)
    y_val = f_a[f_a.split == "val"]["label"].to_numpy(int)
    X_test = f_a[f_a.split == "test"][cols].to_numpy(np.float32)
    y_test = f_a[f_a.split == "test"]["label"].to_numpy(int)
    wts = np.where(y_train == 1,
                   len(y_train) / max(2 * (y_train == 1).sum(), 1),
                   len(y_train) / max(2 * (y_train == 0).sum(), 1))
    clf = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.08, max_leaf_nodes=31,
        min_samples_leaf=50, random_state=42)
    clf.fit(X_train, y_train, sample_weight=wts)
    pv = clf.predict_proba(X_val)[:, 1]
    thr, _ = select_threshold(pv, y_val)
    pt = clf.predict_proba(X_test)[:, 1]
    m = _metrics(y_test, pt, thr)
    rows.append({"model": "SUP_HGB(raw features)",
                 "method": "SUP_HGB", "threshold": thr, **m})
    # oracle
    qs = np.linspace(0, 100, 200)
    best = (-1, None, None)
    for t in np.unique(np.quantile(pt, qs / 100)):
        mm = _metrics(y_test, pt, t)
        if mm["F1"] > best[0]:
            best = (mm["F1"], t, mm)
    rows.append({"model": "SUP_HGB(raw features)",
                 "method": "SUP_HGB_oracle",
                 "threshold": best[1], **best[2]})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    cols_out = ["model", "method", "threshold", "Precision", "Recall",
                "F1", "FPR", "AUC", "PR_AUC", "PA_F1"]
    lines = ["# 固定 20/80 四种方法对比（SWaT）", "",
             "- 前 20% 攻击段：U_CAL/监督训练（其中前 70% train、后 30% val）",
             "- 后 80% 攻击段：冻结测试",
             "- U_N99：Normal-99 阈值（不用异常数据）",
             "- U_CAL：训练攻击窗口选阈值（只用于校准）",
             "- SUP_HGB：原始窗口统计 + 异常数据训练 HGB（全局）",
             "- SUP_HGB_oracle：测试集选阈值，仅作上限", ""]
    lines.append("| " + " | ".join(cols_out) + " |")
    lines.append("|" + "|".join(["---"] * len(cols_out)) + "|")
    for _, x in out.iterrows():
        vals = []
        for c in cols_out:
            v = x[c]
            if isinstance(v, (float, np.floating)):
                vals.append("nan" if np.isnan(v) else f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(out[cols_out].to_string(index=False,
                                  float_format=lambda x: f"{x:.4f}"))
    print(f"已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")


if __name__ == "__main__":
    main()
