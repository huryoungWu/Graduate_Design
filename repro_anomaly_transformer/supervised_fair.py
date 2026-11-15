# -*- coding: utf-8 -*-
"""更公平的监督 pilot：按攻击段分组的 5 折时间块交叉验证。

- 特征：原始窗口统计（mean/std/last/diff，45 通道）+ IF/USAD/LSTM-VAE 分数
- 5 折：每折 7 段攻击；测试折完全没见过的攻击段
- 对比：
  1. Normal-99 阈值（不用异常数据）
  2. 训练折攻击窗口选阈值（半监督校准）
  3. HGB 分类器（异常数据参与训练）
  4. 分类器 Oracle（测试集选阈值，上限）
"""
from __future__ import annotations

import gc
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

from models.classical.normal99_full_attack import _scale
from models.common.config import resolve_device
from models.common.data import ReconstructionDataset
from models.common.protocol import select_threshold
from models.reconstruction.lstm_vae import LSTMVAE
from models.reconstruction.usad import UsadModel
from models.reconstruction.usad_extended_metrics import point_adjust

from .dataset import load_prepared

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "08_SWaT" / "SWaT_all_1s.csv"
SEG = ROOT / "08_SWaT" / "SWaT_attack_segments.csv"
OUT_CSV = ROOT / "runs" / "recon_comparison" / "SUPERVISED_FAIR.csv"
OUT_MD = ROOT / "runs" / "recon_comparison" / "SUPERVISED_FAIR.md"
W = 12
N_FOLDS = 5


def _window_scores(model, X, L=12, batch=8192, point_key="sq_ts"):
    ds = ReconstructionDataset(X, np.zeros(len(X), np.int8), L)
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=0, pin_memory=True)
    out = []
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        for xb, _ in loader:
            o = model.score_batch(xb.to(dev))
            out.append(o["ts"][point_key][:, -1].cpu().numpy())
    return np.concatenate(out).reshape(-1)


def _to12(score, label, ts):
    s = np.lib.stride_tricks.sliding_window_view(
        np.asarray(score, np.float64), W).mean(axis=1)
    y = np.lib.stride_tricks.sliding_window_view(
        np.asarray(label, np.int8), W).max(axis=1)
    return pd.DataFrame({"score": s, "label": y},
                        index=pd.to_datetime(ts)[W - 1:])


def _raw_stats(X):
    # sliding_window_view(axis=0) 的形状是 (N-W+1, C, W)，
    # 窗口在最后一维，不是 axis=1。
    sw = np.lib.stride_tricks.sliding_window_view(X, W, axis=0)
    mean = sw.mean(axis=-1)
    std = sw.std(axis=-1)
    last = sw[:, :, -1]
    diff = sw[:, :, -1] - sw[:, :, 0]
    return np.concatenate([mean, std, last, diff], axis=1).astype(np.float32)


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


def main():
    device = resolve_device("cuda")
    df = pd.read_csv(CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feats = [c for c in df.columns
             if c not in ("timestamp", "label", "period")]
    period = df["period"].astype(str).str.lower()
    normal = df[period == "normal"].reset_index(drop=True)
    attack = df[period == "attack"].reset_index(drop=True)
    print(f"normal={len(normal):,} attack={len(attack):,} feats={len(feats)}",
          flush=True)

    # ---- 模型分数 ----
    score_frames = {}
    # IF
    from sklearn.ensemble import IsolationForest
    d = load_prepared(ROOT / "repro_anomaly_transformer/data/swat")
    ifm = IsolationForest(n_estimators=100, max_samples="auto",
                          contamination="auto", max_features=1.0,
                          bootstrap=False, random_state=42, n_jobs=-1)
    ifm.fit(d["train_X"])
    tr_ps = -ifm.score_samples(d["train_X"])
    te_ps = -ifm.score_samples(d["test_X"])
    score_frames["IF"] = {
        "normal": _to12(tr_ps, np.zeros(len(tr_ps), np.int8),
                        d["train_timestamp"]),
        "attack": _to12(te_ps, d["test_y"], d["test_timestamp"]),
    }
    print("[score] IF done", flush=True)

    for name, scaler, ckpt, builder in [
        ("USAD", "minmax", ROOT / "runs/usad_swat/model.pt",
         lambda c, L: UsadModel(L, c, hidden_size=10, clip_input=False)),
        ("LSTMVAE", "minmax11", ROOT / "runs/lstmvae_swat/model.pt",
         lambda c, L: LSTMVAE(c, L, intermediate_dim=15, z_dim=3,
                              clip_input=False)),
    ]:
        nx, ax, _ = _scale(normal, attack, feats, scaler, False)
        model = builder(len(feats), W).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))
        nps = _window_scores(model, nx)
        aps = _window_scores(model, ax)
        score_frames[name] = {
            "normal": _to12(nps, np.zeros(len(nps), np.int8),
                            normal["timestamp"].iloc[W - 1:]),
            "attack": _to12(aps, attack["label"].to_numpy(np.int8)[W - 1:],
                            attack["timestamp"].iloc[W - 1:]),
        }
        del model, nx, ax
        torch.cuda.empty_cache()
        print(f"[score] {name} done", flush=True)

    # ---- 原始窗口统计 ----
    def _stats_frame(seg, label_mode):
        X = seg[feats].to_numpy(np.float32)
        stats = _raw_stats(X)
        cols = [f"{st}_{c}" for st in ("mean", "std", "last", "diff")
                for c in feats]
        idx = pd.to_datetime(seg["timestamp"]).iloc[W - 1:]
        out = pd.DataFrame(stats, columns=cols, index=idx)
        if label_mode == "zero":
            out["label"] = 0
        else:
            out["label"] = np.lib.stride_tricks.sliding_window_view(
                seg["label"].to_numpy(np.int8), W).max(axis=1)
        return out

    print("[stats] normal/attack ...", flush=True)
    normal_feat = _stats_frame(normal, "zero")
    attack_feat = _stats_frame(attack, "ones")
    for name in ("IF", "USAD", "LSTMVAE"):
        normal_feat = normal_feat.join(
            score_frames[name]["normal"][["score"]].rename(
                columns={"score": name}), how="inner")
        attack_feat = attack_feat.join(
            score_frames[name]["attack"][["score"]].rename(
                columns={"score": name}), how="inner")
    normal_feat = normal_feat.dropna()
    attack_feat = attack_feat.dropna()
    print(f"[stats] normal={normal_feat.shape} attack={attack_feat.shape}",
          flush=True)

    # ---- 5 折：按攻击段分组 ----
    seg = pd.read_csv(SEG)
    seg["start"] = pd.to_datetime(seg["start"])
    seg = seg.sort_values("start").reset_index(drop=True)
    n = len(seg)
    group_size = int(np.ceil(n / N_FOLDS))
    starts = [seg.iloc[min(k * group_size, n - 1)]["start"]
              for k in range(N_FOLDS)]
    starts = np.array(starts, dtype="datetime64[ns]")
    fold = np.searchsorted(starts, attack_feat.index.values,
                           side="right") - 1
    fold = np.clip(fold, 0, N_FOLDS - 1)
    attack_feat["fold"] = fold

    # Normal 段全部作为训练负样本
    Xn = normal_feat.drop(columns="label").to_numpy(np.float32)
    yn = np.zeros(len(normal_feat), np.int8)
    base_cols = [c for c in attack_feat.columns
                 if c not in ("label", "fold")]
    rows = []
    for k in range(N_FOLDS):
        tr = attack_feat[attack_feat.fold != k]
        te = attack_feat[attack_feat.fold == k]
        Xa = tr[base_cols].to_numpy(np.float32)
        ya = tr["label"].to_numpy(int)
        Xtr = np.concatenate([Xn, Xa], axis=0)
        ytr = np.concatenate([yn, ya])
        Xte = te[base_cols].to_numpy(np.float32)
        yte = te["label"].to_numpy(int)
        print(f"[fold {k}] train={len(ytr):,} pos={ytr.mean():.4f} | "
              f"test={len(yte):,} pos={yte.mean():.4f}", flush=True)

        # 1) Normal-99 基线 + 2) 训练折攻击阈值（半监督校准）
        for name in ("IF", "USAD", "LSTMVAE"):
            nscore = normal_feat[name].to_numpy(float)
            thr_n = float(np.percentile(nscore, 99))
            m_n = _metrics(yte, te[name].to_numpy(float), thr_n)
            thr_c, _ = select_threshold(tr[name].to_numpy(float), ya)
            m_c = _metrics(yte, te[name].to_numpy(float), thr_c)
            rows.append({"fold": k, "method": f"U_N99_{name}",
                         "type": "unsup", "threshold": thr_n, **m_n})
            rows.append({"fold": k, "method": f"U_CAL_{name}",
                         "type": "semi", "threshold": thr_c, **m_c})

        # 3) 监督分类器
        wts = np.where(ytr == 1,
                       len(ytr) / max(2 * (ytr == 1).sum(), 1),
                       len(ytr) / max(2 * (ytr == 0).sum(), 1))
        clf = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, max_leaf_nodes=31,
            min_samples_leaf=50, random_state=42)
        clf.fit(Xtr, ytr, sample_weight=wts)
        pt = clf.predict_proba(Xte)[:, 1]
        m_s = _metrics(yte, pt, 0.5)
        rows.append({"fold": k, "method": "SUP_HGB",
                     "type": "supervised", "threshold": 0.5, **m_s})
        # 4) supervised Oracle
        qs = np.linspace(0, 100, 200)
        best = (-1, None, None)
        for t in np.unique(np.quantile(pt, qs / 100)):
            mm = _metrics(yte, pt, t)
            if mm["F1"] > best[0]:
                best = (mm["F1"], t, mm)
        rows.append({"fold": k, "method": "SUP_HGB_oracle",
                     "type": "oracle", "threshold": best[1], **best[2]})
        del Xtr, Xte, clf, pt
        gc.collect()

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    summary = (df_out.groupby(["method", "type"])
               [["Precision", "Recall", "F1", "FPR", "AUC",
                 "PR_AUC", "PA_F1"]]
               .agg(["mean", "std"]).reset_index())
    cols = ["method", "type", "Precision", "Recall", "F1", "FPR",
            "AUC", "PR_AUC", "PA_F1"]
    lines = ["# 公平监督 pilot（SWaT，按攻击段 5 折）", "",
             "- 5 折：每折 7 段攻击；测试折完全没见过的攻击段",
             "- 特征：原始窗口 mean/std/last/diff（45 通道）"
             " + IF/USAD/LSTM-VAE 分数",
             "- U_N99：Normal-99 阈值（不用异常数据）",
             "- U_CAL：训练折攻击窗口选阈值（半监督校准）",
             "- SUP_HGB：异常数据参与训练分类器",
             "- SUP_HGB_oracle：测试集选阈值，仅作上限", ""]
    # 简洁展示每个方法均值
    mean = df_out.groupby(["method", "type"])[
        ["Precision", "Recall", "F1", "FPR", "AUC", "PR_AUC", "PA_F1"]
    ].mean().reset_index()
    lines.append("## 5 折平均")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, x in mean.iterrows():
        vals = []
        for c in cols:
            v = x[c]
            if isinstance(v, (float, np.floating)):
                vals.append("nan" if np.isnan(v) else f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(mean[cols].to_string(index=False,
                               float_format=lambda x: f"{x:.4f}"))
    print(f"已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")


if __name__ == "__main__":
    main()
