# -*- coding: utf-8 -*-
"""监督/半监督 pilot：训练集加入异常攻击段，按攻击段划分 train/val/test。

- 特征：IF / USAD / LSTM-VAE 的 12 步窗口分数
- 训练：Normal 段 + 前 50% 攻击段
- 验证：接下来 20% 攻击段
- 测试：最后 30% 攻击段（完全没见过的攻击段）
- 对比：无监督阈值（只用 Normal/验证） vs 监督分类器 vs Oracle
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from models.classical.normal99_full_attack import _scale, load_periods
from models.common.config import resolve_device
from models.common.protocol import select_threshold
from models.reconstruction.lstm_vae import LSTMVAE
from models.reconstruction.usad import UsadModel
from models.reconstruction.usad_extended_metrics import point_adjust

from .dataset import load_prepared

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "08_SWaT" / "SWaT_all_1s.csv"
SEG = ROOT / "08_SWaT" / "SWaT_attack_segments.csv"
OUT_CSV = ROOT / "runs" / "recon_comparison" / "SUPERVISED_PILOT.csv"
OUT_MD = ROOT / "runs" / "recon_comparison" / "SUPERVISED_PILOT.md"


def _window12(df, label_col="label"):
    s = df["score"].to_numpy(float)
    y = df[label_col].to_numpy(int)
    w = 12
    s12 = np.lib.stride_tricks.sliding_window_view(s, w).mean(axis=1)
    y12 = np.lib.stride_tricks.sliding_window_view(y, w).max(axis=1)
    return pd.DataFrame({"score": s12, "label": y12},
                        index=df.index[w - 1:])


def _point_scores(model, X, L=12, batch=8192):
    from models.common.data import ReconstructionDataset
    ds = ReconstructionDataset(X, np.zeros(len(X), np.int8), L)
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=0, pin_memory=True)
    out = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            o = model.score_batch(xb.to(next(model.parameters()).device))
            out.append(o["ts"]["sq_ts"][:, -1].cpu().numpy())
    return np.concatenate(out).reshape(-1)


def _model_frames():
    frames = {}
    device = resolve_device("cuda")

    # IF：用 prepared 51 维数据
    from sklearn.ensemble import IsolationForest
    d = load_prepared(ROOT / "repro_anomaly_transformer/data/swat")
    tr_x, te_x = d["train_X"], d["test_X"]
    tr_y = np.zeros(len(tr_x), np.int8)
    te_y = d["test_y"].astype(int)
    tr_ts = pd.to_datetime(d["train_timestamp"])
    te_ts = pd.to_datetime(d["test_timestamp"])
    ifm = IsolationForest(n_estimators=100, max_samples="auto",
                          contamination="auto", max_features=1.0,
                          bootstrap=False, random_state=42, n_jobs=-1)
    ifm.fit(tr_x)
    tr_ps = -ifm.score_samples(tr_x)
    te_ps = -ifm.score_samples(te_x)
    frames["IF"] = {
        "normal": _window12(pd.DataFrame(
            {"score": tr_ps, "label": tr_y}, index=tr_ts)),
        "attack": _window12(pd.DataFrame(
            {"score": te_ps, "label": te_y}, index=te_ts)),
    }

    # USAD / LSTM-VAE
    normal, attack, cols = load_periods(str(CSV))
    for name, scaler, ckpt, builder in [
        ("USAD", "minmax", ROOT / "runs/usad_swat/model.pt",
         lambda c, L: UsadModel(L, c, hidden_size=10, clip_input=False)),
        ("LSTMVAE", "minmax11", ROOT / "runs/lstmvae_swat/model.pt",
         lambda c, L: LSTMVAE(c, L, intermediate_dim=15, z_dim=3,
                              clip_input=False)),
    ]:
        nx, ax, _ = _scale(normal, attack, cols, scaler, False)
        model = builder(len(cols), 12).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))
        nps = _point_scores(model, nx)
        aps = _point_scores(model, ax)
        n_ts = pd.to_datetime(normal["timestamp"]).iloc[11:]
        a_ts = pd.to_datetime(attack["timestamp"]).iloc[11:]
        n_lab = np.zeros(len(nps), np.int8)
        a_lab = attack["label"].to_numpy(np.int8)[11:]
        frames[name] = {
            "normal": _window12(pd.DataFrame(
                {"score": nps, "label": n_lab}, index=n_ts)),
            "attack": _window12(pd.DataFrame(
                {"score": aps, "label": a_lab}, index=a_ts)),
        }
        del model, nx, ax
        torch.cuda.empty_cache()

    return frames


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
    frames = _model_frames()
    names = list(frames.keys())

    # 特征对齐
    normal = None
    for name in names:
        f = frames[name]["normal"][["score"]].rename(
            columns={"score": name})
        normal = f if normal is None else normal.join(f, how="inner")
    attack = None
    for name in names:
        f = frames[name]["attack"][["score", "label"]].rename(
            columns={"score": name})
        if attack is None:
            attack = f
        else:
            attack = attack.join(f.drop(columns="label"), how="inner")
    attack["label"] = attack["label"].astype(int)
    normal = normal.dropna()
    attack = attack.dropna()
    print(f"normal windows={len(normal):,} | attack windows={len(attack):,}")

    # 按攻击段划分：前 50% train，接下来 20% val，最后 30% test
    seg = pd.read_csv(SEG)
    seg["start"] = pd.to_datetime(seg["start"])
    seg = seg.sort_values("start").reset_index(drop=True)
    n = len(seg)
    n_tr = int(n * 0.5)
    n_va = int(n * 0.2)
    val_start = seg.iloc[n_tr]["start"]
    test_start = seg.iloc[n_tr + n_va]["start"]
    print(f"segments={n} | train<{val_start} | val<{test_start} | test>="
          f"{test_start}")

    # 12 步窗口起点决定归属，避免窗口跨入下一个 split
    start_ts = attack.index - pd.Timedelta(seconds=11)
    split = np.where(start_ts < val_start, "train",
                     np.where(start_ts < test_start, "val", "test"))
    attack["split"] = split
    # Normal 段全部作为训练负样本
    train_x = pd.concat([
        normal.assign(label=0),
        attack[attack.split == "train"],
    ])
    val = attack[attack.split == "val"]
    test = attack[attack.split == "test"]
    if len(val) == 0 or len(test) == 0:
        raise RuntimeError("val/test split 为空")

    Xtr = train_x[names].to_numpy(float)
    ytr = train_x["label"].to_numpy(int)
    Xva = val[names].to_numpy(float)
    yva = val["label"].to_numpy(int)
    Xte = test[names].to_numpy(float)
    yte = test["label"].to_numpy(int)
    print(f"train={len(ytr):,} (pos {ytr.mean():.4f}) | "
          f"val={len(yva):,} (pos {yva.mean():.4f}) | "
          f"test={len(yte):,} (pos {yte.mean():.4f})")

    rows = []
    # 无监督基线：每个模型分数，阈值在 val 上选
    for name in names:
        thr, vf = select_threshold(val[name].to_numpy(float), yva)
        m = _metrics(yte, test[name].to_numpy(float), thr)
        rows.append({"model": f"unsup_{name}", "type": "unsup",
                     "val_F1": vf, "threshold": thr, **m})

    # 监督分类器
    clfs = {
        "LR": make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000,
                                               class_weight="balanced")),
        "HGB": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
            min_samples_leaf=50, random_state=42),
    }
    best = None
    for cname, clf in clfs.items():
        clf.fit(Xtr, ytr)
        pv = clf.predict_proba(Xva)[:, 1]
        thr, vf = select_threshold(pv, yva)
        m = _metrics(yte, clf.predict_proba(Xte)[:, 1], thr)
        rows.append({"model": f"sup_{cname}", "type": "supervised",
                     "val_F1": vf, "threshold": thr, **m})
        if best is None or vf > best[0]:
            best = (vf, cname, clf)
    vf, cname, clf = best
    pv = clf.predict_proba(Xva)[:, 1]
    thr, _ = select_threshold(pv, yva)
    pt = clf.predict_proba(Xte)[:, 1]
    m = _metrics(yte, pt, thr)
    # Oracle（测试集上选阈值）
    qs = np.linspace(0, 100, 300)
    best_o = (-1, None, None)
    for t in np.unique(np.quantile(pt, qs / 100)):
        mm = _metrics(yte, pt, t)
        if mm["F1"] > best_o[0]:
            best_o = (mm["F1"], t, mm)
    rows.append({"model": f"sup_{cname}_oracle", "type": "oracle",
                 "val_F1": vf, "threshold": best_o[1], **best_o[2]})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    cols = ["model", "type", "val_F1", "Precision", "Recall",
            "F1", "FPR", "AUC", "PR_AUC", "PA_F1", "threshold"]
    lines = ["# 监督 pilot（SWaT，按攻击段划分）", "",
             "- 训练：Normal 段 + 前 50% 攻击段",
             "- 验证：接下来 20% 攻击段",
             "- 测试：最后 30% 攻击段（未见过）",
             "- 特征：IF / USAD / LSTM-VAE 的 12 步窗口分数",
             "- 无监督基线：阈值只在验证集上选",
             "- Oracle：测试集上选阈值，仅作上限", ""]
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
