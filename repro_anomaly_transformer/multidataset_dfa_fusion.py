# -*- coding: utf-8 -*-
"""把 SWaT 上最优的两个 DFA-COPOD E6 融合配置应用到多数据集。

配置：
- ``tcn_raw_mean_z``：原始 COPOD + TCN 隐特征 COPOD，z-score 后 mean；
- ``tcn_whiten_max_z``：原始 COPOD + PCA 白化 TCN 隐特征 COPOD，z-score 后 max。

每个数据集：
- 原始 COPOD 在 Normal 上拟合；
- TCN 在该数据集的 Normal 上训练（SWaT 复用已训练权重），
  取最后一时刻 64 维隐层作为特征；
- 隐特征 COPOD（raw / PCA-whiten）在 Normal 上拟合；
- 60 窗分数 -> 12 步均值，与原始 COPOD 的 12 步分数对齐（丢掉前 60 个窗口）；
- 阈值：Normal99（主）、ValBestF1、ValFPR1。

输出：runs/recon_comparison/MULTIDATASET_DFA_FUSION.csv / .md
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from models.common.config import resolve_device
from models.common.protocol import select_threshold
from models.forecasting.tcn import TCNForecaster
from pyod.models.copod import COPOD

from .benchmark_all_models import _metrics
from .dfa_copod import _copod_scores, _window_max, _window_mean

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "multidataset"
RC = ROOT / "runs" / "recon_comparison"
OUT_CSV = RC / "MULTIDATASET_DFA_FUSION.csv"
OUT_MD = RC / "MULTIDATASET_DFA_FUSION.md"
SCORE_DIR = ROOT / "runs" / "dfa_fusion_scores"

DATASETS = [
    "psm", "sml_machine-1-1", "sml_machine-1-2", "sml_machine-2-1",
    "batadal", "battledim_2018", "battledim_2019",
    "smap_A-1", "smap_A-2", "smap_P-1", "smap_S-1",
    "leakdb_hanoi500", "leakdb_hanoi1000", "leakdb_net1",
    "swat", "wadi",
]
L = 60
W = 12
FIT_N = 200000
TCN_EPOCHS = 10
TCN_MAX_BATCHES = 800
SOURCES = ["Normal99", "ValBestF1", "ValFPR1"]


class _WindowDS(Dataset):
    def __init__(self, X, seq_len=L):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, i):
        return self.X[i:i + self.seq_len], self.X[i + self.seq_len]


def _train_tcn(Xtr, device, seed=42):
    n_ch = Xtr.shape[1]
    model = TCNForecaster(n_ch, L, 1).to(device)
    ds = _WindowDS(Xtr)
    if len(ds) < 10:
        return None, 0.0
    loader = DataLoader(ds, batch_size=256, shuffle=True,
                        num_workers=0, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    torch.manual_seed(seed)
    t0 = time.time()
    for epoch in range(TCN_EPOCHS):
        model.train()
        for i, (x, y) in enumerate(loader):
            if i >= TCN_MAX_BATCHES:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred[:, 0, :], y)
            loss.backward()
            opt.step()
    return model.eval(), time.time() - t0


@torch.no_grad()
def _extract_tcn(model, X, device, batch=2048):
    view = np.lib.stride_tricks.sliding_window_view(X, L, axis=0)
    parts = []
    for start in range(0, len(view), batch):
        w = view[start:start + batch].transpose(0, 2, 1)
        t = torch.from_numpy(np.ascontiguousarray(w)).float().to(device)
        h = model.blocks(model.input_proj(t.transpose(1, 2)))
        f = h[:, :, -1]
        parts.append(f.cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def _fit_copod(X, seed=42, fit_n=FIT_N):
    if fit_n is None or fit_n >= len(X):
        Xf = X
    else:
        idx = np.random.RandomState(seed).choice(
            len(X), int(fit_n), replace=False)
        Xf = X[idx]
    t0 = time.time()
    m = COPOD(contamination=0.01, n_jobs=1)
    m.fit(Xf)
    return m, time.time() - t0


def _prep_raw(point, y):
    s = _window_mean(point, W)[60:]
    yy = _window_max(y, W)[60:]
    n = min(len(s), len(yy))
    return s[:n], yy[:n]


def _prep_tcn(window, y):
    s = _window_mean(window, W)[1:]
    yy = _window_max(y, W)[60:]
    n = min(len(s), len(yy))
    return s[:n], yy[:n]


def _prep_raw_normal(point):
    return _window_mean(point, W)


def _prep_tcn_normal(window):
    return _window_mean(window, W)


def _stats(x):
    x = np.asarray(x, dtype=np.float64)
    return float(x.mean()), float(x.std() if x.std() > 1e-12 else 1.0)


def _z(x, st):
    return (np.asarray(x, dtype=np.float64) - st[0]) / st[1]


def _eval(dataset, config, s_n, sv, yv, st_, yt, fit_s, score_s,
          train_n, extra=None):
    rows = []
    for source in SOURCES:
        if source == "Normal99":
            thr = float(np.percentile(s_n, 99))
            sign = 1
        elif source == "ValBestF1":
            thr, _ = select_threshold(sv, yv)
            thr, sign = float(thr), 1
        else:
            nv = sv[yv == 0]
            thr = float(np.percentile(nv if len(nv) else sv, 99))
            sign = 1
        met, _ = _metrics(yt, sign * st_, thr)
        row = {
            "dataset": dataset, "config": config,
            "threshold_source": source, "threshold": float(thr),
            **met, "n": int(len(yt)),
            "positives": int(np.asarray(yt).sum()),
            "fit_s": float(fit_s), "score_s": float(score_s),
            "train_n": int(train_n),
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    device = resolve_device("auto")
    print(f"[dfa-fusion] device={device}", flush=True)
    rows = []

    datasets = args.datasets or DATASETS
    for ds in datasets:
        d = CACHE / ds
        if not d.exists():
            print(f"[skip] {ds}: cache missing")
            continue
        Xtr = np.load(d / "train_X.npy")
        Xv = np.load(d / "val_X.npy")
        yv = np.load(d / "val_y.npy")
        Xt = np.load(d / "test_X.npy")
        yt = np.load(d / "test_y.npy")
        print(f"\n[dfa-fusion] {ds}: train={len(Xtr):,} "
              f"val={len(Xv):,} test={len(Xt):,} "
              f"C={Xtr.shape[1]}", flush=True)

        # 1) raw COPOD
        copod_r, fit_r = _fit_copod(Xtr, fit_n=None)
        t0 = time.time()
        sn_r = _copod_scores(copod_r, Xtr)
        sv_r = _copod_scores(copod_r, Xv)
        st_r = _copod_scores(copod_r, Xt)
        score_r = time.time() - t0
        rv, yv2 = _prep_raw(sv_r, yv)
        rt, yt2 = _prep_raw(st_r, yt)
        rn = _prep_raw_normal(sn_r)
        rows += _eval(ds, "raw_copod", rn, rv, yv2, rt, yt2,
                      fit_r, score_r, len(Xtr))
        SCORE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            SCORE_DIR / f"{ds}__raw_copod.npz",
            normal=rn, val=rv, test=rt,
            y_val=yv2, y_test=yt2)

        # 2) TCN
        if ds == "swat" and (ROOT / "runs/tcn_swat/model.pt").exists():
            model = TCNForecaster(Xtr.shape[1], L, 1).to(device)
            model.load_state_dict(torch.load(
                ROOT / "runs/tcn_swat/model.pt", map_location=device))
            model.eval()
            train_t = 0.0
            print("  TCN: reuse runs/tcn_swat/model.pt", flush=True)
        else:
            model, train_t = _train_tcn(Xtr, device)
            if model is None:
                print("  [skip] TCN: too few training windows")
                continue
            print(f"  TCN trained: {train_t:.1f}s", flush=True)

        t0 = time.time()
        Ftr = _extract_tcn(model, Xtr, device)
        Fv = _extract_tcn(model, Xv, device)
        Ft = _extract_tcn(model, Xt, device)
        feat_t = time.time() - t0
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 3) TCN raw COPOD
        copod_tr, fit_tr = _fit_copod(Ftr)
        t0 = time.time()
        sn_tr = _copod_scores(copod_tr, Ftr)
        sv_tr = _copod_scores(copod_tr, Fv)
        st_tr = _copod_scores(copod_tr, Ft)
        score_tr = time.time() - t0

        # 4) TCN PCA-whiten COPOD
        pca_idx = np.random.RandomState(42).choice(
            len(Ftr), min(FIT_N, len(Ftr)), replace=False)
        pca = PCA(n_components=min(32, Ftr.shape[1]),
                  whiten=True, random_state=42).fit(Ftr[pca_idx])
        Ftr_w = pca.transform(Ftr)
        Fv_w = pca.transform(Fv)
        Ft_w = pca.transform(Ft)
        copod_tw, fit_tw = _fit_copod(Ftr_w)
        t0 = time.time()
        sn_tw = _copod_scores(copod_tw, Ftr_w)
        sv_tw = _copod_scores(copod_tw, Fv_w)
        st_tw = _copod_scores(copod_tw, Ft_w)
        score_tw = time.time() - t0

        # 5) 对齐
        tv_r, ty_r = _prep_raw(sv_r, yv)
        tt_r, ty_t = _prep_raw(st_r, yt)
        tv_tr, _ = _prep_tcn(sv_tr, yv)
        tt_tr, _ = _prep_tcn(st_tr, yt)
        tv_tw, _ = _prep_tcn(sv_tw, yv)
        tt_tw, _ = _prep_tcn(st_tw, yt)
        # 标签以 raw 的为准
        yv2, yt2 = ty_r, ty_t
        nv = min(len(tv_r), len(tv_tr), len(tv_tw), len(yv2))
        nt = min(len(tt_r), len(tt_tr), len(tt_tw), len(yt2))
        if int(np.asarray(yt2[:nt]).sum()) == 0:
            print(f"  [skip] {ds}: alignment 后测试集无正例")
            continue
        tv_r, tv_tr, tv_tw = tv_r[:nv], tv_tr[:nv], tv_tw[:nv]
        yv2 = yv2[:nv]
        tt_r, tt_tr, tt_tw = tt_r[:nt], tt_tr[:nt], tt_tw[:nt]
        yt2 = yt2[:nt]

        rn = _prep_raw_normal(sn_r)
        trn = _prep_tcn_normal(sn_tr)
        twn = _prep_tcn_normal(sn_tw)
        T = min(len(rn), len(trn), len(twn))
        rn, trn, twn = rn[-T:], trn[-T:], twn[-T:]

        stat_r = _stats(rn)
        stat_tr = _stats(trn)
        stat_tw = _stats(twn)
        zr_n = _z(rn, stat_r)
        zr_v = _z(tv_r, stat_r)
        zr_t = _z(tt_r, stat_r)
        zt_n = _z(trn, stat_tr)
        zv_t = _z(tv_tr, stat_tr)
        zt_t = _z(tt_tr, stat_tr)
        zw_n = _z(twn, stat_tw)
        zv_w = _z(tv_tw, stat_tw)
        zw_t = _z(tt_tw, stat_tw)

        # config 1: TCN-raw mean_z
        f_n = 0.5 * (zr_n + zt_n)
        f_v = 0.5 * (zr_v + zv_t)
        f_t = 0.5 * (zr_t + zt_t)
        rows += _eval(ds, "tcn_raw_mean_z", f_n, f_v, yv2, f_t, yt2,
                      fit_r + train_t + feat_t + fit_tr,
                      score_r + score_tr, len(Xtr))
        np.savez_compressed(
            SCORE_DIR / f"{ds}__tcn_raw_mean_z.npz",
            normal=f_n, val=f_v, test=f_t,
            y_val=yv2, y_test=yt2)

        # config 2: TCN-whiten max_z
        f_n = np.maximum(zr_n, zw_n)
        f_v = np.maximum(zr_v, zv_w)
        f_t = np.maximum(zr_t, zw_t)
        rows += _eval(ds, "tcn_whiten_max_z", f_n, f_v, yv2, f_t, yt2,
                      fit_r + train_t + feat_t + fit_tw,
                      score_r + score_tw, len(Xtr))
        np.savez_compressed(
            SCORE_DIR / f"{ds}__tcn_whiten_max_z.npz",
            normal=f_n, val=f_v, test=f_t,
            y_val=yv2, y_test=yt2)
        print(f"  done: raw={score_r:.1f}s tcnraw={score_tr:.1f}s "
              f"tcnwhiten={score_tw:.1f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # ---------------- Markdown ----------------
    lines = [
        "# 跨数据集 DFA-COPOD 融合（两个 SWaT 最优配置）", "",
        "- `tcn_raw_mean_z`：原始 COPOD + TCN-raw COPOD，mean_z",
        "- `tcn_whiten_max_z`：原始 COPOD + TCN-PCA-whiten COPOD，max_z",
        "- 每个数据集 Normal only 训练；12 步窗口；"
        "60 窗 TCN 分数与原始 COPOD 对齐到 row 71",
        "- 阈值：Normal99 / ValBestF1 / ValFPR1；**不使用测试标签选阈值**",
        "- TCN 每个数据集重新训练（SWaT 复用已训练权重），"
        f"{TCN_EPOCHS} epochs、每 epoch 最多 {TCN_MAX_BATCHES} batch", "",
    ]
    for source in SOURCES:
        sub = out[out.threshold_source == source]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="dataset", columns="config",
                              values="F1", aggfunc="mean")
        lines += [f"## {source} F1", "",
                  "| dataset | " + " | ".join(piv.columns) + " |",
                  "|" + "|".join(["---"] * (len(piv.columns) + 1)) + "|"]
        for ds, row in piv.iterrows():
            vals = ["nan" if pd.isna(x) else f"{x:.4f}" for x in row.values]
            lines.append(f"| {ds} | " + " | ".join(vals) + " |")
        lines.append("")

    # 平均
    v = out[out.threshold_source == "Normal99"]
    agg = v.groupby("config").agg(
        mean_F1=("F1", "mean"), std_F1=("F1", "std"),
        mean_PA=("PA_F1", "mean"), mean_AUC=("AUC", "mean"),
        datasets=("dataset", "nunique")).sort_values("mean_F1",
                                                     ascending=False)
    lines += ["## Normal99 平均（全部数据集）", "",
              "| config | mean F1 | std F1 | mean PA-F1 | mean AUC | "
              "datasets |", "|---|---|---|---|---|---|"]
    for c, r in agg.iterrows():
        lines.append(
            f"| {c} | {r.mean_F1:.4f} | {r.std_F1:.4f} | "
            f"{r.mean_PA:.4f} | {r.mean_AUC:.4f} | "
            f"{int(r.datasets)} |")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
