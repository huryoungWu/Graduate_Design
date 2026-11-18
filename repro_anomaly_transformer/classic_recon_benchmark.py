# -*- coding: utf-8 -*-
"""Benchmark classic reconstruction/representation models.

Models: AnomalyTransformer, DCdetector, SBT (USAD/LSTMVAE already covered).

Two modes:
- cache dataset (multivariate): pointwise Normal99 metrics
- telemanom channel: event-level Normal99 metrics

Usage:
    python -m repro_anomaly_transformer.classic_recon_benchmark --dataset swat
    python -m repro_anomaly_transformer.classic_recon_benchmark --channel A-5
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
from torch.utils.data import DataLoader, Dataset

from models.reconstruction.anomaly_transformer import AnomalyTransformer
from models.reconstruction.dcdetector import DCdetector
from models.reconstruction.sbt import SBT

from .robust_dfa_fusion import (
    L, _clip_train, _metrics_for, _prep_tcn, _prep_tcn_normal,
)
from .smap_msl_event import event_metrics, load_channel, point_metrics

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "multidataset"
OUT_CACHE = ROOT / "runs" / "classic_recon_cache"
OUT_CH = ROOT / "runs" / "classic_recon_channels"
EPOCHS = 5
MAX_BATCHES = 200
MODELS = ["anomaly_transformer", "dcdetector", "sbt"]
LABEL = {
    "anomaly_transformer": "AnomalyTransformer recon-error only",
    "dcdetector": "DCdetector contrastive-score only",
    "sbt": "SBT view-diff only",
}


class WindowDS(Dataset):
    def __init__(self, X, seq_len=L):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, i):
        return self.X[i:i + self.seq_len], self.X[i + self.seq_len]


def build(kind, n_ch):
    if kind == "anomaly_transformer":
        return AnomalyTransformer(n_ch, L)
    if kind == "dcdetector":
        return DCdetector(n_ch, L)
    if kind == "sbt":
        return SBT(n_ch, L)
    raise ValueError(kind)


def train(kind, Xtr, device, epochs, max_batches):
    torch.manual_seed(42)
    model = build(kind, Xtr.shape[1]).to(device)
    # DCdetector's contrastive loss is O((B*L)^2); use a smaller batch.
    bs = 32 if kind == "dcdetector" else 128
    loader = DataLoader(WindowDS(Xtr), batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            opt.zero_grad()
            loss = model.training_loss(x)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            opt.step()
    return model.eval(), time.time() - t0


@torch.no_grad()
def scores(kind, model, X, device, batch=512):
    view = np.lib.stride_tricks.sliding_window_view(X, L, axis=0)
    rec, assoc = [], []
    for s in range(0, len(view), batch):
        w = view[s:s + batch].transpose(0, 2, 1)
        t = torch.from_numpy(np.ascontiguousarray(w)).float().to(device)
        d = model.score_batch(t)
        if kind == "anomaly_transformer":
            r = d["window"]["recon_max"]
            a = d["window"]["assoc_max"]
            if torch.is_tensor(r):
                r = r.detach().cpu().numpy()
                a = a.detach().cpu().numpy()
            rec.append(np.asarray(r).reshape(-1))
            assoc.append(np.asarray(a).reshape(-1))
        else:
            v = d["window"]["score_mean"]
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy()
            rec.append(np.asarray(v).reshape(-1))
    if kind == "anomaly_transformer":
        return (np.concatenate(rec).astype(np.float64),
                np.concatenate(assoc).astype(np.float64))
    return np.concatenate(rec).astype(np.float64)


def _at_fuse(rn, an, rv, av, rt, at):
    """Normalize recon/assoc with train-normal min-max, then sum."""
    def norm(x, lo, hi):
        return (x - lo) / (hi - lo + 1e-8)
    lo_r, hi_r = float(rn.min()), float(rn.max())
    lo_a, hi_a = float(an.min()), float(an.max())
    return (norm(rn, lo_r, hi_r) + norm(an, lo_a, hi_a),
            norm(rv, lo_r, hi_r) + norm(av, lo_a, hi_a),
            norm(rt, lo_r, hi_r) + norm(at, lo_a, hi_a))


def run_cache(ds, device, epochs, max_batches):
    d = CACHE / ds
    Xtr = np.load(d / "train_X.npy")
    Xv = np.load(d / "val_X.npy")
    yv = np.load(d / "val_y.npy").astype(int)
    Xt = np.load(d / "test_X.npy")
    yt = np.load(d / "test_y.npy").astype(int)
    Xtr, Xv, Xt = _clip_train(Xtr, Xv, Xt)
    rows = []
    for kind in MODELS:
        try:
            model, fit_s = train(kind, Xtr, device, epochs, max_batches)
            s_tr = scores(kind, model, Xtr, device)
            s_v = scores(kind, model, Xv, device)
            s_t = scores(kind, model, Xt, device)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if kind == "anomaly_transformer":
                sn, sv, st = _at_fuse(*s_tr, *s_v, *s_t)
            else:
                sn, sv, st = s_tr, s_v, s_t
            n = _prep_tcn_normal(sn)
            v, yv2 = _prep_tcn(sv, yv)
            t, yt2 = _prep_tcn(st, yt)
            nv = min(len(v), len(yv2))
            nt = min(len(t), len(yt2))
            v, yv2 = v[:nv], yv2[:nv]
            t, yt2 = t[:nt], yt2[:nt]
            met = _metrics_for(yv2, v, yt2, t, n)
            rows.append({"dataset": ds, "protocol": "pointwise",
                         "model": LABEL[kind], "fit_s": fit_s, **met})
            print(f"  {ds} {kind} {fit_s:.1f}s F1={met['Normal99_F1']:.3f}",
                  flush=True)
        except Exception as exc:
            print(f"  [error] {ds} {kind}: {type(exc).__name__}: {exc}",
                  flush=True)
    return rows


def run_channel(ch, device, epochs, max_batches):
    loaded = load_channel(ch)
    if loaded is None:
        return []
    Xtr_raw, Xte_raw, y, spacecraft, cls = loaded
    Xtr, Xte = _clip_train(Xtr_raw, Xte_raw)
    rows = []
    for kind in MODELS:
        try:
            model, fit_s = train(kind, Xtr, device, epochs, max_batches)
            s_tr = scores(kind, model, Xtr, device)
            s_t = scores(kind, model, Xte, device)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if kind == "anomaly_transformer":
                rn, an = s_tr
                rt, at_ = s_t
                lo_r, hi_r = float(rn.min()), float(rn.max())
                lo_a, hi_a = float(an.min()), float(an.max())
                norm = lambda x, lo, hi: (x - lo) / (hi - lo + 1e-8)
                sn = norm(rn, lo_r, hi_r) + norm(an, lo_a, hi_a)
                st = norm(rt, lo_r, hi_r) + norm(at_, lo_a, hi_a)
            else:
                sn, st = s_tr, s_t
            n = _prep_tcn_normal(sn)
            t, yy = _prep_tcn(st, y)
            nt = min(len(t), len(yy))
            t, yy = t[:nt], yy[:nt]
            from .robust_dfa_fusion import _tail_z
            zn = _tail_z(n, n)
            zt = _tail_z(t, n)
            thr = float(np.percentile(zn, 99))
            em = event_metrics(yy, (zt > thr).astype(int))
            if em:
                rows.append({"channel": ch, "spacecraft": spacecraft,
                             "class": cls, "model": LABEL[kind],
                             "fit_s": fit_s, **em,
                             **{f"pt_{k}": v for k, v in
                                point_metrics(yy, zt, thr).items()
                                if k != "pred"}})
                print(f"  {ch} {kind} {fit_s:.1f}s eF1={em['event_f1']:.3f}",
                      flush=True)
        except Exception as exc:
            print(f"  [error] {ch} {kind}: {type(exc).__name__}: {exc}",
                  flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--channel", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-batches", type=int, default=MAX_BATCHES)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset or args.datasets:
        dss = args.datasets or [args.dataset]
        for ds in dss:
            t0 = time.time()
            rows = run_cache(ds, device, args.epochs, args.max_batches)
            if rows:
                OUT_CACHE.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(
                    OUT_CACHE / f"{ds}.csv", index=False,
                    encoding="utf-8-sig")
            print(f"[cache] {ds} done {time.time()-t0:.1f}s", flush=True)
    if args.channel or args.channels:
        chs = args.channels or [args.channel]
        for ch in chs:
            t0 = time.time()
            rows = run_channel(ch, device, args.epochs, args.max_batches)
            if rows:
                OUT_CH.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(
                    OUT_CH / f"{ch}.csv", index=False,
                    encoding="utf-8-sig")
            print(f"[channel] {ch} done {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
