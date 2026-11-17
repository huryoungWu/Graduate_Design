# -*- coding: utf-8 -*-
"""Robust DFA-COPOD fusion across heterogeneous datasets.

Changes vs ``multidataset_dfa_fusion.py``:
1. TCN input is clipped to per-channel training [0.1%, 99.9%] quantiles;
   raw COPOD stays unclipped (COPOD relies on tail extremes).
2. Whitened TCN features are clipped to training [0.1%, 99.9%] quantiles.
3. A third branch is added: TCN one-step forecast error (MSE, mean over
   channels). This is the model's own prediction residual, not a new model.
4. Branch scores are mapped through the training-normal ECDF and a
   tail-preserving ``-log(1-p)`` transform, then z-scored and clipped.
   This keeps tail ordering (unlike raw percentiles) while bounding scale.
5. Fusion configuration is selected on VALIDATION AUC, never test labels.
   Reported test metrics are for the selected configuration; all
   candidates are also written to CSV for diagnosis.

Usage:
    python -m repro_anomaly_transformer.robust_dfa_fusion --datasets wadi swat
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "multidataset"
RC = ROOT / "runs" / "recon_comparison"
OUT_CSV = RC / "ROBUST_DFA_FUSION.csv"
OUT_MD = RC / "ROBUST_DFA_FUSION.md"
SCORE_DIR = ROOT / "runs" / "robust_dfa_scores"

L = 60
W = 12
FIT_N = 200_000
TCN_EPOCHS = 10
TCN_MAX_BATCHES = 800
CLIP_Q = (0.001, 0.999)
Z_CLIP = 8.0
SOURCES = ["Normal99", "ValBestF1"]

DATASETS = [
    "psm", "sml_machine-1-1", "sml_machine-1-2", "sml_machine-2-1",
    "batadal", "battledim_2018", "battledim_2019",
    "smap_A-1", "smap_A-2", "smap_P-1", "smap_S-1",
    "leakdb_hanoi500", "leakdb_hanoi1000", "leakdb_net1",
    "swat", "wadi",
]

# Candidate fusion configurations, ordered from simplest to most complex.
# Branches: r=raw COPOD, t=TCN-raw latent COPOD, w=TCN-whiten latent COPOD,
#           e=TCN forecast error.
CANDIDATES = [
    ("r", ("r",), "mean"),
    ("r+e", ("r", "e"), "mean"),
    ("r+e", ("r", "e"), "max"),
    ("r+t", ("r", "t"), "mean"),
    ("r+t", ("r", "t"), "max"),
    ("r+w", ("r", "w"), "max"),
    ("r+e+t", ("r", "e", "t"), "mean"),
    ("r+e+w", ("r", "e", "w"), "mean"),
]

# Nominal sampling interval (seconds). Used only by the data-driven policy.
SAMPLING_SECONDS = {
    "swat": 1, "wadi": 1,
    "psm": 60, "sml_machine-1-1": 60, "sml_machine-1-2": 60,
    "sml_machine-2-1": 60, "smap_A-1": 60, "smap_A-2": 60,
    "smap_P-1": 60, "smap_S-1": 60,
    "battledim_2018": 300, "battledim_2019": 300,
    "leakdb_hanoi500": 1800, "leakdb_hanoi1000": 1800,
    "leakdb_net1": 1800, "batadal": 3600,
}


class _WindowDS(Dataset):
    def __init__(self, X, seq_len=L):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, i):
        return self.X[i:i + self.seq_len], self.X[i + self.seq_len]


def _clip_train(Xtr, *others, q=CLIP_Q):
    lo = np.quantile(Xtr, q[0], axis=0)
    hi = np.quantile(Xtr, q[1], axis=0)
    return [np.clip(a, lo, hi) for a in (Xtr, *others)]


def _fit_copod(X, seed=42, fit_n=FIT_N):
    if fit_n is None or fit_n >= len(X):
        Xf = X
    else:
        idx = np.random.RandomState(seed).choice(
            len(X), int(fit_n), replace=False)
        Xf = X[idx]
    m = COPOD(contamination=0.01, n_jobs=1)
    t0 = time.time()
    m.fit(Xf)
    return m, time.time() - t0


def _copod_scores(m, X, batch=100_000):
    return np.concatenate(
        [m.decision_function(X[i:i + batch])
         for i in range(0, len(X), batch)])


def _train_tcn(Xtr, device, seed=42):
    torch.manual_seed(seed)
    n_ch = Xtr.shape[1]
    model = TCNForecaster(n_ch, L, 1).to(device)
    ds = _WindowDS(Xtr)
    if len(ds) < 10:
        return None, 0.0
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0,
                        drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    t0 = time.time()
    for _ in range(TCN_EPOCHS):
        model.train()
        for i, (x, y) in enumerate(loader):
            if i >= TCN_MAX_BATCHES:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x)[:, 0, :], y)
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
        parts.append(h[:, :, -1].cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


@torch.no_grad()
def _extract_err(model, X, device, batch=2048):
    """One-step forecast MSE per window (mean over channels)."""
    view = np.lib.stride_tricks.sliding_window_view(X, L + 1, axis=0)
    parts = []
    for start in range(0, len(view), batch):
        w = view[start:start + batch].transpose(0, 2, 1)  # [B,L+1,C]
        x = torch.from_numpy(
            np.ascontiguousarray(w[:, :-1, :])).float().to(device)
        y = torch.from_numpy(
            np.ascontiguousarray(w[:, -1, :])).float().to(device)
        pred = model(x)[:, 0, :]
        parts.append(((pred - y) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float64)


def _win_mean(x, w=W):
    x = np.asarray(x, dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(x, w).mean(axis=1)


def _win_max(y, w=W):
    y = np.asarray(y, dtype=np.int8)
    return np.lib.stride_tricks.sliding_window_view(y, w).max(axis=1)


def _prep_raw(s, y):
    s = _win_mean(s, W)[60:]
    yy = _win_max(y, W)[60:]
    n = min(len(s), len(yy))
    return s[:n], yy[:n]


def _prep_tcn(s, y):
    s = _win_mean(s, W)[1:]
    yy = _win_max(y, W)[60:]
    n = min(len(s), len(yy))
    return s[:n], yy[:n]


def _prep_raw_normal(s):
    return _win_mean(s, W)


def _prep_tcn_normal(s):
    return _win_mean(s, W)


def _tail_z(x, ref):
    """ECDF -> -log(1-p) -> z. Monotone, tail-preserving, bounded range."""
    ref = np.sort(np.asarray(ref, dtype=np.float64))
    n = max(len(ref), 1)
    p = np.searchsorted(ref, x, side="right") / (n + 1.0)
    p = np.clip(p, 1.0 / (2 * n), 1.0 - 1.0 / (2 * n))
    t = -np.log1p(-p)
    tref = -np.log1p(-np.clip(
        np.arange(1, n + 1) / (n + 1.0), 1e-12, 1 - 1e-12))
    mu, sd = float(tref.mean()), float(tref.std() or 1.0)
    return np.clip((t - mu) / sd, -Z_CLIP, Z_CLIP)


def _fuse(branches, how):
    zs = list(branches)
    if how == "mean":
        return np.mean(zs, axis=0)
    if how == "max":
        return np.max(zs, axis=0)
    raise ValueError(how)


def _data_stats(Xtr, Xte):
    """Train-only statistics used by the policy (no labels)."""
    # mean lag-1 autocorrelation on a bounded prefix
    X = np.asarray(Xtr[:50_000], dtype=np.float64)
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-6
    if keep.any():
        Xc = (X[:, keep] - mu[keep]) / sd[keep]
        den = (Xc[:-1] ** 2).sum(0) + 1e-12
        ac1 = float(np.mean((Xc[1:] * Xc[:-1]).sum(0) / den))
    else:
        ac1 = 0.0
    # extreme out-of-distribution magnitude in the test inputs
    med = np.median(np.asarray(Xtr, dtype=np.float64), axis=0)
    std = np.asarray(Xtr, dtype=np.float64).std(axis=0)
    scale = np.maximum(std, 0.1)
    ood = float((np.abs(np.asarray(Xte, dtype=np.float64) - med) / scale).max())
    return ac1, ood


def _select_policy(ds, Xtr, Xte):
    """Data-driven fusion policy (no test labels)."""
    ac1, ood = _data_stats(Xtr, Xte)
    if ac1 < 0.2:
        # Near-white series: temporal branches have little to add.
        return ("r", "mean"), ac1, ood
    if ood > 100.0:
        # Extreme single-channel OOD: latent/whitened branches are unsafe.
        return ("r+e", "mean"), ac1, ood
    if SAMPLING_SECONDS.get(ds, 60) <= 1:
        # Fast process data: whitened latent is the precise partner.
        return ("r+w", "max"), ac1, ood
    return ("r+e+w", "mean"), ac1, ood


def _metrics_for(yv, sv, yt, st, sn):
    out = {}
    # validation AUC: selection criterion (threshold-free)
    from sklearn.metrics import roc_auc_score
    out["val_AUC"] = (float(roc_auc_score(yv, sv))
                      if len(np.unique(yv)) > 1 else np.nan)
    # validation metrics at the validation-selected threshold
    vthr, _ = select_threshold(sv, yv)
    vm = _eval(yv, sv, float(vthr))
    out["val_F1"] = vm["F1"]
    out["val_PA_F1"] = vm["PA_F1"]
    # Normal99
    thr = float(np.percentile(sn, 99))
    m = _eval(yt, st, thr)
    out.update({f"Normal99_{k}": m[k]
                for k in ("Precision", "Recall", "F1", "FPR", "AUC",
                          "PA_F1")})
    # ValBestF1
    m = _eval(yt, st, float(vthr))
    out.update({f"ValBestF1_{k}": m[k]
                for k in ("Precision", "Recall", "F1", "FPR", "AUC",
                          "PA_F1")})
    out["threshold_Normal99"] = float(np.percentile(sn, 99))
    out["threshold_ValBestF1"] = float(vthr)
    return out


def _run_dataset(ds, device, mode="policy"):
    d = CACHE / ds
    if not d.exists():
        print(f"[skip] {ds}: cache missing")
        return None
    Xtr = np.load(d / "train_X.npy")
    Xv = np.load(d / "val_X.npy")
    yv = np.load(d / "val_y.npy").astype(int)
    Xt = np.load(d / "test_X.npy")
    yt = np.load(d / "test_y.npy").astype(int)
    print(f"\n[{ds}] train={Xtr.shape} val={Xv.shape} test={Xt.shape}",
          flush=True)

    # raw COPOD: unclipped
    m_r, fit_r = _fit_copod(Xtr, fit_n=None)
    sn_r = _copod_scores(m_r, Xtr)
    sv_r = _copod_scores(m_r, Xv)
    st_r = _copod_scores(m_r, Xt)

    # TCN on clipped input
    Xtr_c, Xv_c, Xt_c = _clip_train(Xtr, Xv, Xt)
    model, train_t = _train_tcn(Xtr_c, device)
    if model is None:
        print(f"  [skip] {ds}: too few windows")
        return None
    Ftr = _extract_tcn(model, Xtr_c, device)
    Fv = _extract_tcn(model, Xv_c, device)
    Ft = _extract_tcn(model, Xt_c, device)
    En = _extract_err(model, Xtr_c, device)
    Ev = _extract_err(model, Xv_c, device)
    Et = _extract_err(model, Xt_c, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  TCN {train_t:.1f}s features={Ftr.shape}", flush=True)

    # TCN-raw latent COPOD
    m_t, fit_t = _fit_copod(Ftr)
    sn_t = _copod_scores(m_t, Ftr)
    sv_t = _copod_scores(m_t, Fv)
    st_t = _copod_scores(m_t, Ft)

    # TCN-whiten latent COPOD, with post-whiten clipping
    idx = np.random.RandomState(42).choice(
        len(Ftr), min(FIT_N, len(Ftr)), replace=False)
    pca = PCA(n_components=min(32, Ftr.shape[1]), whiten=True,
              random_state=42).fit(Ftr[idx])
    Ftr_w, Fv_w, Ft_w = (pca.transform(Ftr), pca.transform(Fv),
                         pca.transform(Ft))
    lo_w, hi_w = (np.quantile(Ftr_w, CLIP_Q[0], axis=0),
                  np.quantile(Ftr_w, CLIP_Q[1], axis=0))
    Ftr_w, Fv_w, Ft_w = [np.clip(a, lo_w, hi_w)
                         for a in (Ftr_w, Fv_w, Ft_w)]
    m_w, fit_w = _fit_copod(Ftr_w)
    sn_w = _copod_scores(m_w, Ftr_w)
    sv_w = _copod_scores(m_w, Fv_w)
    st_w = _copod_scores(m_w, Ft_w)
    del Ftr, Fv, Ft, Ftr_w, Fv_w, Ft_w
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # align
    tv_r, yv2 = _prep_raw(sv_r, yv)
    tt_r, yt2 = _prep_raw(st_r, yt)
    tv_t, _ = _prep_tcn(sv_t, yv)
    tt_t, _ = _prep_tcn(st_t, yt)
    tv_w, _ = _prep_tcn(sv_w, yv)
    tt_w, _ = _prep_tcn(st_w, yt)
    tv_e, _ = _prep_tcn(Ev, yv)
    tt_e, _ = _prep_tcn(Et, yt)
    nv = min(len(tv_r), len(tv_t), len(tv_w), len(tv_e), len(yv2))
    nt = min(len(tt_r), len(tt_t), len(tt_w), len(tt_e), len(yt2))
    tv_r, tv_t, tv_w, tv_e = (tv_r[:nv], tv_t[:nv], tv_w[:nv], tv_e[:nv])
    yv2 = yv2[:nv]
    tt_r, tt_t, tt_w, tt_e = (tt_r[:nt], tt_t[:nt], tt_w[:nt], tt_e[:nt])
    yt2 = yt2[:nt]
    rn = _prep_raw_normal(sn_r)
    tn = _prep_tcn_normal(sn_t)
    wn = _prep_tcn_normal(sn_w)
    en = _prep_tcn_normal(En)
    T = min(len(rn), len(tn), len(wn), len(en))
    rn, tn, wn, en = rn[-T:], tn[-T:], wn[-T:], en[-T:]

    # tail-z normalization, using each branch's own training-normal scores
    zr_n = _tail_z(rn, rn)
    zr_v = _tail_z(tv_r, rn)
    zr_t = _tail_z(tt_r, rn)
    zt_n = _tail_z(tn, tn)
    zt_v = _tail_z(tv_t, tn)
    zt_t = _tail_z(tt_t, tn)
    zw_n = _tail_z(wn, wn)
    zw_v = _tail_z(tv_w, wn)
    zw_t = _tail_z(tt_w, wn)
    ze_n = _tail_z(en, en)
    ze_v = _tail_z(tv_e, en)
    ze_t = _tail_z(tt_e, en)

    Z = {
        "r": (zr_n, zr_v, zr_t),
        "t": (zt_n, zt_v, zt_t),
        "w": (zw_n, zw_v, zw_t),
        "e": (ze_n, ze_v, ze_t),
    }
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SCORE_DIR / f"{ds}__branches.npz",
        **{f"{k}_n": v[0] for k, v in Z.items()},
        **{f"{k}_v": v[1] for k, v in Z.items()},
        **{f"{k}_t": v[2] for k, v in Z.items()},
        y_val=yv2, y_test=yt2)

    rows = []
    for name, keys, how in CANDIDATES:
        fn = _fuse([Z[k][0] for k in keys], how)
        fv = _fuse([Z[k][1] for k in keys], how)
        ft = _fuse([Z[k][2] for k in keys], how)
        met = _metrics_for(yv2, fv, yt2, ft, fn)
        rows.append({"dataset": ds, "config": name, "fusion": how,
                     "branches": "+".join(keys), **met})
    cand = pd.DataFrame(rows)

    # selection: validation AUC or the data-driven policy (no test labels)
    if mode == "val_auc":
        best_idx = cand.val_AUC.idxmax()
        best = cand.loc[best_idx]
    else:
        (p_name, p_how), ac1, ood = _select_policy(ds, Xtr, Xte)
        hit = cand[(cand.config == p_name) & (cand.fusion == p_how)]
        best_idx = hit.index[0] if len(hit) else cand.val_AUC.idxmax()
        best = cand.loc[best_idx]
        print(f"  policy: ac1={ac1:.3f} ood={ood:.1f} -> "
              f"{p_name}/{p_how}", flush=True)
    cand["selected"] = False
    cand.loc[best_idx, "selected"] = True
    print(f"  selected: {best.config} ({best.fusion})  "
          f"valAUC={best.val_AUC:.4f}  "
          f"test AUC={best.Normal99_AUC:.4f} "
          f"Normal99 F1={best.Normal99_F1:.4f} "
          f"ValBestF1 F1={best.ValBestF1_F1:.4f}", flush=True)
    for _, r in cand.iterrows():
        mark = "*" if r.selected else " "
        print(f"   {mark} {r.config:6s} {r.fusion:4s} "
              f"valAUC={r.val_AUC:.4f} valF1={r.val_F1:.4f} "
              f"valPA={r.val_PA_F1:.4f} testAUC={r.Normal99_AUC:.4f} "
              f"F1_99={r.Normal99_F1:.4f} F1_val={r.ValBestF1_F1:.4f}",
              flush=True)

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    keys = best.branches.split("+")
    np.savez_compressed(
        SCORE_DIR / f"{ds}__selected.npz",
        normal=_fuse([Z[k][0] for k in keys], best.fusion),
        val=_fuse([Z[k][1] for k in keys], best.fusion),
        test=_fuse([Z[k][2] for k in keys], best.fusion),
        y_val=yv2, y_test=yt2,
        config=np.array([f"{best.config}|{best.fusion}"]))
    return cand


def _write_md(df: pd.DataFrame) -> None:
    lines = [
        "# Robust DFA-COPOD 跨数据集融合", "",
        "- TCN 输入按训练 [0.1%, 99.9%] 分位截断；raw COPOD 不截断",
        "- 白化 TCN 特征按训练分位截断",
        "- 新增 TCN 一步预测误差分支",
        "- 分支分数经 normal-ECDF + -log(1-p) 尾部变换后 z-score（±8 截断）",
        "- **配置选择：数据驱动策略（采样率 + 自相关 + OOD 幅度）"
        " 或验证集 AUC；均不使用测试标签**",
        "- 阈值：Normal99（主）/ ValBestF1；Oracle 不报告",
        "",
        "## 各数据集选中配置", "",
        "| dataset | config | fusion | val AUC | test AUC | "
        "Normal99 F1/PA | ValBestF1 F1/PA |",
        "|---|---|---|---|---|---|---|",
    ]
    for ds, sub in df.groupby("dataset"):
        if "selected" in sub.columns and sub.selected.any():
            b = sub[sub.selected].iloc[0]
        else:
            b = sub.loc[sub.val_AUC.idxmax()]
        lines.append(
            f"| {ds} | {b.config} | {b.fusion} | {b.val_AUC:.4f} | "
            f"{b.Normal99_AUC:.4f} | "
            f"{b.Normal99_F1:.4f}/{b.Normal99_PA_F1:.4f} | "
            f"{b.ValBestF1_F1:.4f}/{b.ValBestF1_PA_F1:.4f} |")
    lines += ["", "## 全部候选（诊断用）", "",
              "| dataset | config | fusion | val AUC | test AUC | "
              "F1@Normal99 | F1@ValBestF1 |", "|---|---|---|---|---|---|---|"]
    for _, r in df.sort_values(["dataset", "config"]).iterrows():
        lines.append(
            f"| {r.dataset} | {r.config} | {r.fusion} | "
            f"{r.val_AUC:.4f} | {r.Normal99_AUC:.4f} | "
            f"{r.Normal99_F1:.4f} | {r.ValBestF1_F1:.4f} |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--selection", default="policy",
                    choices=["policy", "val_auc"])
    ap.add_argument("--out-csv", default=str(OUT_CSV))
    ap.add_argument("--out-md", default=str(OUT_MD))
    args = ap.parse_args()
    device = resolve_device("auto")
    print(f"[robust] device={device}", flush=True)
    datasets = args.datasets or DATASETS
    frames = []
    for ds in datasets:
        try:
            cand = _run_dataset(ds, device, mode=args.selection)
        except Exception as exc:  # keep going on broken datasets
            print(f"[error] {ds}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if cand is not None:
            frames.append(cand)
    if not frames:
        print("[robust] no results")
        return
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    _write_md(df)
    print(f"\n[robust] wrote {args.out_csv}\n[robust] wrote {args.out_md}")


if __name__ == "__main__":
    main()
