# -*- coding: utf-8 -*-
"""WADI 鲁棒性探针：输入按训练分位截断 + 白化特征截断。

只做对照实验，不写入 runs/。用法：
    D:/Python/python.exe _scripts/wadi_robust_probe.py --epochs 3 --max-batches 500
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.forecasting.tcn import TCNForecaster  # noqa: E402
from pyod.models.copod import COPOD  # noqa: E402

L = 60
W = 12
FIT_N = 200_000


class WindowDS(Dataset):
    def __init__(self, X, seq_len=L):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, i):
        return self.X[i:i + self.seq_len], self.X[i + self.seq_len]


def clip_to_train(Xtr, *others, q=(0.001, 0.999)):
    lo = np.quantile(Xtr, q[0], axis=0)
    hi = np.quantile(Xtr, q[1], axis=0)
    return [np.clip(a, lo, hi) for a in (Xtr, *others)]


def train_tcn(Xtr, device, epochs, max_batches, seed=42):
    torch.manual_seed(seed)
    model = TCNForecaster(Xtr.shape[1], L, 1).to(device)
    ds = WindowDS(Xtr)
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0,
                        drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        losses = []
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x)[:, 0, :], y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"  epoch {ep+1}: loss={np.mean(losses):.5f} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return model.eval()


@torch.no_grad()
def extract_tcn(model, X, device, batch=2048):
    view = np.lib.stride_tricks.sliding_window_view(X, L, axis=0)
    parts = []
    for start in range(0, len(view), batch):
        w = view[start:start + batch].transpose(0, 2, 1)
        t = torch.from_numpy(np.ascontiguousarray(w)).float().to(device)
        h = model.blocks(model.input_proj(t.transpose(1, 2)))
        parts.append(h[:, :, -1].cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


@torch.no_grad()
def extract_forecast_err(model, X, device, batch=2048):
    """每个窗口的下一时刻预测误差（MSE，跨通道平均）。"""
    view = np.lib.stride_tricks.sliding_window_view(X, L + 1, axis=0)
    parts = []
    for start in range(0, len(view), batch):
        w = view[start:start + batch].transpose(0, 2, 1)  # [B,L+1,C]
        x = torch.from_numpy(np.ascontiguousarray(w[:, :-1, :])).float().to(device)
        y = torch.from_numpy(np.ascontiguousarray(w[:, -1, :])).float().to(device)
        pred = model(x)[:, 0, :]
        parts.append(((pred - y) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float64)


def fit_copod(X, seed=42, fit_n=FIT_N):
    if fit_n is None or fit_n >= len(X):
        Xf = X
    else:
        idx = np.random.RandomState(seed).choice(
            len(X), int(fit_n), replace=False)
        Xf = X[idx]
    m = COPOD(contamination=0.01, n_jobs=1)
    m.fit(Xf)
    return m


def copod_scores(m, X, batch=100_000):
    return np.concatenate(
        [m.decision_function(X[i:i + batch])
         for i in range(0, len(X), batch)])


def win_mean(x, w=W):
    x = np.asarray(x, dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(x, w).mean(axis=1)


def win_max(y, w=W):
    y = np.asarray(y, dtype=np.int8)
    return np.lib.stride_tricks.sliding_window_view(y, w).max(axis=1)


def prep_raw(s, y):
    s = win_mean(s, W)[60:]
    yy = win_max(y, W)[60:]
    n = min(len(s), len(yy))
    return s[:n], yy[:n]


def prep_tcn(s, y):
    s = win_mean(s, W)[1:]
    yy = win_max(y, W)[60:]
    n = min(len(s), len(yy))
    return s[:n], yy[:n]


def prep_raw_normal(s):
    return win_mean(s, W)


def prep_tcn_normal(s):
    return win_mean(s, W)


def ecdf_map(x, ref):
    ref = np.sort(np.asarray(ref, dtype=np.float64))
    return np.searchsorted(ref, x, side="right") / max(len(ref), 1)


def metrics(y, s, thr):
    from sklearn.metrics import (average_precision_score,
                                 precision_recall_fscore_support,
                                 roc_auc_score)
    pred = (s > thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    neg = y == 0
    fpr = float(((pred == 1) & neg).sum() / max(neg.sum(), 1))
    auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan
    return dict(P=float(p), R=float(r), F1=float(f1), FPR=fpr, AUC=auc)


def report(name, y, s, n):
    thr = float(np.percentile(n, 99))
    m = metrics(y, s, thr)
    print(f"  {name:28s} thr={thr:8.3f} P={m['P']:.4f} R={m['R']:.4f} "
          f"F1={m['F1']:.4f} FPR={m['FPR']:.4f} AUC={m['AUC']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="wadi")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-batches", type=int, default=500)
    ap.add_argument("--clip", action="store_true", default=True)
    ap.add_argument("--no-clip", dest="clip", action="store_false")
    args = ap.parse_args()

    d = ROOT / "runs" / "multidataset" / args.dataset
    Xtr = np.load(d / "train_X.npy")
    Xv = np.load(d / "val_X.npy")
    yv = np.load(d / "val_y.npy")
    Xt = np.load(d / "test_X.npy")
    yt = np.load(d / "test_y.npy")
    print(f"[{args.dataset}] train={Xtr.shape} val={Xv.shape} "
          f"test={Xt.shape} clip={args.clip}", flush=True)
    # raw COPOD 必须在未截断数据上做：COPOD 依赖尾部极值
    Xtr_raw, Xv_raw, Xt_raw = Xtr, Xv, Xt
    if args.clip:
        Xtr, Xv, Xt = clip_to_train(Xtr, Xv, Xt)
        print("  TCN input clipped to train [0.1%, 99.9%]", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}", flush=True)
    model = train_tcn(Xtr, device, args.epochs, args.max_batches)
    Ftr = extract_tcn(model, Xtr, device)
    Fv = extract_tcn(model, Xv, device)
    Ft = extract_tcn(model, Xt, device)
    En = extract_forecast_err(model, Xtr, device)
    Ev = extract_forecast_err(model, Xv, device)
    Et = extract_forecast_err(model, Xt, device)
    print(f"  features {Ftr.shape}, forecast err {En.shape}", flush=True)

    # raw COPOD (unclipped, full train — 与现有 raw 基线一致)
    m_r = fit_copod(Xtr_raw, fit_n=None)
    sn_r, sv_r, st_r = (copod_scores(m_r, Xtr_raw),
                        copod_scores(m_r, Xv_raw),
                        copod_scores(m_r, Xt_raw))
    # TCN raw
    m_tr = fit_copod(Ftr)
    sn_tr, sv_tr, st_tr = (copod_scores(m_tr, Ftr), copod_scores(m_tr, Fv),
                           copod_scores(m_tr, Ft))
    # TCN whiten with post-whiten clipping
    idx = np.random.RandomState(42).choice(
        len(Ftr), min(FIT_N, len(Ftr)), replace=False)
    pca = PCA(n_components=min(32, Ftr.shape[1]), whiten=True,
              random_state=42).fit(Ftr[idx])
    Ftr_w, Fv_w, Ft_w = (pca.transform(Ftr), pca.transform(Fv),
                         pca.transform(Ft))
    lo_w = np.quantile(Ftr_w, 0.001, axis=0)
    hi_w = np.quantile(Ftr_w, 0.999, axis=0)
    Ftr_w, Fv_w, Ft_w = [np.clip(a, lo_w, hi_w)
                         for a in (Ftr_w, Fv_w, Ft_w)]
    m_tw = fit_copod(Ftr_w)
    sn_tw, sv_tw, st_tw = (copod_scores(m_tw, Ftr_w),
                           copod_scores(m_tw, Fv_w),
                           copod_scores(m_tw, Ft_w))

    # align
    tv_r, yv2 = prep_raw(sv_r, yv)
    tt_r, yt2 = prep_raw(st_r, yt)
    tv_tr, _ = prep_tcn(sv_tr, yv)
    tt_tr, _ = prep_tcn(st_tr, yt)
    tv_tw, _ = prep_tcn(sv_tw, yv)
    tt_tw, _ = prep_tcn(st_tw, yt)
    tv_e, _ = prep_tcn(Ev, yv)
    tt_e, _ = prep_tcn(Et, yt)
    nv = min(len(tv_r), len(tv_tr), len(tv_tw), len(tv_e), len(yv2))
    nt = min(len(tt_r), len(tt_tr), len(tt_tw), len(tt_e), len(yt2))
    tv_r, tv_tr, tv_tw, tv_e = (tv_r[:nv], tv_tr[:nv], tv_tw[:nv],
                                tv_e[:nv])
    yv2 = yv2[:nv]
    tt_r, tt_tr, tt_tw, tt_e = (tt_r[:nt], tt_tr[:nt], tt_tw[:nt],
                                tt_e[:nt])
    yt2 = yt2[:nt]
    rn = prep_raw_normal(sn_r)
    trn = prep_tcn_normal(sn_tr)
    twn = prep_tcn_normal(sn_tw)
    en = prep_tcn_normal(En)
    T = min(len(rn), len(trn), len(twn), len(en))
    rn, trn, twn, en = rn[-T:], trn[-T:], twn[-T:], en[-T:]

    # z-space (existing)
    stat = lambda x: (float(np.mean(x)), float(np.std(x) if np.std(x) > 1e-12 else 1.0))  # noqa: E731
    sr, std = stat(rn), stat(trn)
    sw = stat(twn)
    z = lambda x, s: (x - s[0]) / s[1]  # noqa: E731

    def report_all(name, y, s, n, sv, yv):
        from sklearn.metrics import precision_recall_curve
        from models.common.protocol import select_threshold
        for src in ("Normal99", "ValBestF1", "Oracle"):
            if src == "Normal99":
                thr = float(np.percentile(n, 99))
            elif src == "ValBestF1":
                thr, _ = select_threshold(sv, yv)
                thr = float(thr)
            else:
                p, r, t = precision_recall_curve(y, s)
                f = 2 * p * r / (p + r + 1e-12)
                k = int(np.argmax(f[:-1])) if len(t) else 0
                thr = float(t[k]) if len(t) else float(np.max(s))
            m = metrics(y, s, thr)
            print(f"  {name:22s} {src:9s} thr={thr:8.3f} "
                  f"F1={m['F1']:.4f} P={m['P']:.4f} R={m['R']:.4f} "
                  f"FPR={m['FPR']:.4f} AUC={m['AUC']:.4f}")

    print("\n[z-space fusion]")
    report_all("raw", yt2, z(tt_r, sr), z(rn, sr), z(tv_r, sr), yv2)
    report_all("tcn_raw mean_z", yt2,
               0.5 * (z(tt_r, sr) + z(tt_tr, std)),
               0.5 * (z(rn, sr) + z(trn, std)),
               0.5 * (z(tv_r, sr) + z(tv_tr, std)), yv2)
    report_all("tcn_whiten mean_z", yt2,
               0.5 * (z(tt_r, sr) + z(tt_tw, sw)),
               0.5 * (z(rn, sr) + z(twn, sw)),
               0.5 * (z(tv_r, sr) + z(tv_tw, sw)), yv2)
    report_all("tcn_whiten max_z", yt2,
               np.maximum(z(tt_r, sr), z(tt_tw, sw)),
               np.maximum(z(rn, sr), z(twn, sw)),
               np.maximum(z(tv_r, sr), z(tv_tw, sw)), yv2)

    # percentile-space fusion (robust)
    pr_n = ecdf_map(rn, rn)
    pt_n = ecdf_map(trn, trn)
    pw_n = ecdf_map(twn, twn)
    pr_t = ecdf_map(tt_r, rn)
    pt_t = ecdf_map(tt_tr, trn)
    pw_t = ecdf_map(tt_tw, twn)
    pr_v = ecdf_map(tv_r, rn)
    pt_v = ecdf_map(tv_tr, trn)
    pw_v = ecdf_map(tv_tw, twn)
    pe_n = ecdf_map(en, en)
    pe_t = ecdf_map(tt_e, en)
    pe_v = ecdf_map(tv_e, en)
    print("\n[percentile fusion]")
    report_all("raw", yt2, pr_t, pr_n, pr_v, yv2)
    report_all("forecast_err", yt2, pe_t, pe_n, pe_v, yv2)
    report_all("tcn_raw mean_p", yt2, 0.5 * (pr_t + pt_t),
               0.5 * (pr_n + pt_n), 0.5 * (pr_v + pt_v), yv2)
    report_all("tcn_whiten mean_p", yt2, 0.5 * (pr_t + pw_t),
               0.5 * (pr_n + pw_n), 0.5 * (pr_v + pw_v), yv2)
    report_all("tcn_whiten max_p", yt2, np.maximum(pr_t, pw_t),
               np.maximum(pr_n, pw_n), np.maximum(pr_v, pw_v), yv2)
    report_all("raw+err mean_p", yt2, 0.5 * (pr_t + pe_t),
               0.5 * (pr_n + pe_n), 0.5 * (pr_v + pe_v), yv2)
    report_all("raw+err max_p", yt2, np.maximum(pr_t, pe_t),
               np.maximum(pr_n, pe_n), np.maximum(pr_v, pe_v), yv2)
    report_all("raw+err+tcn mean_p", yt2,
               (pr_t + pe_t + pt_t) / 3.0,
               (pr_n + pe_n + pt_n) / 3.0,
               (pr_v + pe_v + pt_v) / 3.0, yv2)

    # dynamic rolling threshold on fused percentile (1% FPR, 6h at 1s)
    def rolling_thr(s, win=21600, q=0.99):
        out = np.empty_like(s, dtype=np.float64)
        for i in range(len(s)):
            a = s[max(0, i - win):i]
            out[i] = np.quantile(a, q) if len(a) >= 100 else np.nan
        return out

    fused = 0.5 * (pr_t + pw_t)
    thr_roll = rolling_thr(fused)
    valid = np.isfinite(thr_roll)
    if valid.sum() > 0:
        pred = (fused[valid] > thr_roll[valid]).astype(int)
        from sklearn.metrics import f1_score
        print(f"\n[dynamic rolling 1%FPR 6h] n={valid.sum()} "
              f"F1={f1_score(yt2[valid], pred):.4f} "
              f"FPR={pred[yt2[valid]==0].mean():.4f} "
              f"R={pred[yt2[valid]==1].mean():.4f}")


if __name__ == "__main__":
    main()
