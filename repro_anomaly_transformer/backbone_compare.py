# -*- coding: utf-8 -*-
"""Compare TCN-like backbones under the robust DFA-COPOD fusion.

Backbones:
- TCN (baseline, ``models/forecasting/tcn.py``)
- ModernTCN (repo ``models/forecasting/moderntcn.py``, doc item 7)
- MICN (repo ``models/forecasting/micn.py``, multi-scale conv)
- SE-TCN (channel attention TCN, doc item 2; implemented here)

For every backbone, the same three branches are built:
raw COPOD, backbone latent COPOD (raw + PCA-whiten), backbone forecast error.
Fusion candidates are identical; the data-driven policy selects one row per
backbone (no test labels). All candidates are written for diagnosis.

Run one dataset per process to keep memory bounded:
    python -m repro_anomaly_transformer.backbone_compare --dataset wadi
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from models.common.config import resolve_device
from models.forecasting.micn import MICNForecaster
from models.forecasting.moderntcn import ModernTCNForecaster
from models.forecasting.tcn import TCNForecaster

from .robust_dfa_fusion import (
    CANDIDATES, FIT_N, L, W, _clip_train, _copod_scores, _fit_copod,
    _fuse, _metrics_for, _prep_raw, _prep_raw_normal, _prep_tcn,
    _prep_tcn_normal, _select_policy, _tail_z,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "multidataset"
OUT_DIR = ROOT / "runs" / "backbone_compare"
TCN_EPOCHS = 10
TCN_MAX_BATCHES = 800


# ----------------------------------------------------------------------
# SE-TCN
# ----------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, ch, kernel=3, dilation=1, dropout=0.1, reduction=4):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, kernel, padding=pad,
                              dilation=dilation)
        self.norm = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ch, max(ch // reduction, 4)), nn.ReLU(),
            nn.Linear(max(ch // reduction, 4), ch), nn.Sigmoid())

    def forward(self, x):
        out = self.conv(x)[:, :, :x.size(-1)]
        out = self.norm(out + x)
        scale = self.se(out).unsqueeze(-1)
        return self.drop(torch.relu(out * scale))


class SETCNForecaster(nn.Module):
    def __init__(self, n_channels, seq_len, pred_len=1, hidden=64,
                 dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.input_proj = nn.Conv1d(n_channels, hidden, 1)
        self.blocks = nn.Sequential(
            SEBlock(hidden, dilation=1, dropout=dropout),
            SEBlock(hidden, dilation=2, dropout=dropout),
            SEBlock(hidden, dilation=4, dropout=dropout),
            SEBlock(hidden, dilation=8, dropout=dropout),
        )
        self.head = nn.Linear(hidden, pred_len * n_channels)

    def encode(self, x):
        h = self.blocks(self.input_proj(x.transpose(1, 2)))
        return h[:, :, -1]

    def forward(self, x):
        out = self.head(self.encode(x))
        return out.view(-1, self.pred_len, self.n_channels)


BACKBONES = ["TCN", "ModernTCN", "MICN", "SE-TCN"]


def build_model(kind, n_channels):
    if kind == "TCN":
        return TCNForecaster(n_channels, L, 1)
    if kind == "ModernTCN":
        return ModernTCNForecaster(n_channels, L, 1)
    if kind == "MICN":
        return MICNForecaster(n_channels, L, 1)
    if kind == "SE-TCN":
        return SETCNForecaster(n_channels, L, 1)
    raise ValueError(kind)


def encode(kind, model, x):
    """Use the representation at the last time step (same semantics as TCN)."""
    if kind == "TCN":
        return model.blocks(model.input_proj(x.transpose(1, 2)))[:, :, -1]
    if kind == "ModernTCN":
        h = torch.relu(model.dwconv(x.transpose(1, 2)))
        h = torch.relu(model.pwconv(h))
        return model.drop(model.norm(h))[:, :, -1]
    if kind == "MICN":
        x = x.transpose(1, 2)
        h = model.fuse(torch.cat([
            torch.relu(model.local_conv(x)),
            torch.relu(model.global_conv(x))], dim=1))
        return h[:, :, -1]
    return model.encode(x)


class _WindowDS(Dataset):
    def __init__(self, X, seq_len=L):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, i):
        return self.X[i:i + self.seq_len], self.X[i + self.seq_len]


def train_model(kind, Xtr, device, seed=42, epochs=None,
                max_batches=None):
    epochs = TCN_EPOCHS if epochs is None else epochs
    max_batches = TCN_MAX_BATCHES if max_batches is None else max_batches
    torch.manual_seed(seed)
    model = build_model(kind, Xtr.shape[1]).to(device)
    ds = _WindowDS(Xtr)
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x)[:, 0, :], y)
            loss.backward()
            opt.step()
    return model.eval(), time.time() - t0


@torch.no_grad()
def extract_latent(kind, model, X, device, batch=2048):
    view = np.lib.stride_tricks.sliding_window_view(X, L, axis=0)
    parts = []
    for s in range(0, len(view), batch):
        w = view[s:s + batch].transpose(0, 2, 1)
        t = torch.from_numpy(np.ascontiguousarray(w)).float().to(device)
        parts.append(encode(kind, model, t).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


@torch.no_grad()
def extract_err(kind, model, X, device, batch=2048):
    view = np.lib.stride_tricks.sliding_window_view(X, L + 1, axis=0)
    parts = []
    for s in range(0, len(view), batch):
        w = view[s:s + batch].transpose(0, 2, 1)
        x = torch.from_numpy(
            np.ascontiguousarray(w[:, :-1, :])).float().to(device)
        y = torch.from_numpy(
            np.ascontiguousarray(w[:, -1, :])).float().to(device)
        pred = model(x)[:, 0, :]
        parts.append(((pred - y) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float64)


def evaluate_backbone(ds, kind, Xtr, Xv, Xt, yv, yt, device,
                      raw_scores, policy, epochs, max_batches):
    Xtr_c, Xv_c, Xt_c = _clip_train(Xtr, Xv, Xt)
    model, train_t = train_model(kind, Xtr_c, device,
                                 epochs=epochs, max_batches=max_batches)
    Ftr = extract_latent(kind, model, Xtr_c, device)
    Fv = extract_latent(kind, model, Xv_c, device)
    Ft = extract_latent(kind, model, Xt_c, device)
    En = extract_err(kind, model, Xtr_c, device)
    Ev = extract_err(kind, model, Xv_c, device)
    Et = extract_err(kind, model, Xt_c, device)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    m_t, _ = _fit_copod(Ftr)
    sn_t = _copod_scores(m_t, Ftr)
    sv_t = _copod_scores(m_t, Fv)
    st_t = _copod_scores(m_t, Ft)

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
    m_w, _ = _fit_copod(Ftr_w)
    sn_w = _copod_scores(m_w, Ftr_w)
    sv_w = _copod_scores(m_w, Fv_w)
    st_w = _copod_scores(m_w, Ft_w)
    del Ftr, Fv, Ft, Ftr_w, Fv_w, Ft_w
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sn_r, sv_r, st_r = raw_scores
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

    Z = {
        "r": (_tail_z(rn, rn), _tail_z(tv_r, rn), _tail_z(tt_r, rn)),
        "t": (_tail_z(tn, tn), _tail_z(tv_t, tn), _tail_z(tt_t, tn)),
        "w": (_tail_z(wn, wn), _tail_z(tv_w, wn), _tail_z(tt_w, wn)),
        "e": (_tail_z(en, en), _tail_z(tv_e, en), _tail_z(tt_e, en)),
    }
    rows = []
    for name, keys, how in CANDIDATES:
        fn = _fuse([Z[k][0] for k in keys], how)
        fv = _fuse([Z[k][1] for k in keys], how)
        ft = _fuse([Z[k][2] for k in keys], how)
        met = _metrics_for(yv2, fv, yt2, ft, fn)
        rows.append({"dataset": ds, "backbone": kind, "config": name,
                     "fusion": how, "branches": "+".join(keys),
                     **met})
    cand = pd.DataFrame(rows)
    p_name, p_how = policy
    hit = cand[(cand.config == p_name) & (cand.fusion == p_how)]
    sel_idx = hit.index[0] if len(hit) else cand.val_AUC.idxmax()
    cand["selected"] = False
    cand.loc[sel_idx, "selected"] = True
    cand["best_valauc"] = False
    cand.loc[cand.val_AUC.idxmax(), "best_valauc"] = True
    cand["best_testF1"] = False
    cand.loc[cand.Normal99_F1.idxmax(), "best_testF1"] = True
    print(f"  [{kind}] train={train_t:.1f}s selected "
          f"{cand.loc[sel_idx, 'config']}/{cand.loc[sel_idx, 'fusion']} "
          f"N99 F1={cand.loc[sel_idx, 'Normal99_F1']:.4f} "
          f"P={cand.loc[sel_idx, 'Normal99_Precision']:.4f} "
          f"R={cand.loc[sel_idx, 'Normal99_Recall']:.4f} "
          f"AUC={cand.loc[sel_idx, 'Normal99_AUC']:.4f}", flush=True)
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbones", nargs="*", default=BACKBONES)
    ap.add_argument("--epochs", type=int, default=TCN_EPOCHS)
    ap.add_argument("--max-batches", type=int, default=TCN_MAX_BATCHES)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    ds = args.dataset
    d = CACHE / ds
    if not d.exists():
        raise SystemExit(f"missing cache: {d}")
    Xtr = np.load(d / "train_X.npy")
    Xv = np.load(d / "val_X.npy")
    yv = np.load(d / "val_y.npy").astype(int)
    Xt = np.load(d / "test_X.npy")
    yt = np.load(d / "test_y.npy").astype(int)
    print(f"[backbone] {ds} train={Xtr.shape} val={Xv.shape} "
          f"test={Xt.shape}", flush=True)

    device = resolve_device("auto")
    raw_scores = None
    m_r, fit_r = _fit_copod(Xtr, fit_n=None)
    sn_r = _copod_scores(m_r, Xtr)
    sv_r = _copod_scores(m_r, Xv)
    st_r = _copod_scores(m_r, Xt)
    raw_scores = (sn_r, sv_r, st_r)
    del m_r
    gc.collect()
    policy, ac1, ood = _select_policy(ds, Xtr, Xt)
    print(f"  raw COPOD {fit_r:.1f}s policy={policy} "
          f"ac1={ac1:.3f} ood={ood:.1f}", flush=True)

    frames = []
    for kind in args.backbones:
        try:
            cand = evaluate_backbone(ds, kind, Xtr, Xv, Xt, yv, yt,
                                     device, raw_scores, policy,
                                     args.epochs, args.max_batches)
            frames.append(cand)
        except Exception as exc:
            print(f"  [error] {kind}: {type(exc).__name__}: {exc}",
                  flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not frames:
        print("[backbone] no results")
        return
    out = pd.concat(frames, ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"{ds}.csv", index=False, encoding="utf-8-sig")
    print(f"[backbone] wrote {out_dir / (ds + '.csv')}", flush=True)


if __name__ == "__main__":
    main()
