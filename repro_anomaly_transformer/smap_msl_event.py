# -*- coding: utf-8 -*-
"""Per-channel SMAP/MSL anomaly detection with event-level metrics.

Each telemanom channel is an independent univariate telemetry stream.
Models: raw COPOD, TCN forecast error, AE reconstruction, USAD reconstruction.

Protocol:
- train/{ch}.parquet = normal training stream (standardized, constant cols dropped)
- test/{ch}.parquet = labeled test stream; labels from labeled_anomalies.parquet
- score -> tail-z against train-normal ECDF
- Normal99 threshold from train-normal scores
- event metrics: event recall/precision/F1, median delay, false alarms/1000 normal
  plus pointwise P/R/F1/AUC for reference

Run one channel or a range:
    python -m repro_anomaly_transformer.smap_msl_event --channels A-1 P-1
"""
from __future__ import annotations

import argparse
import ast
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from models.common.config import resolve_device
from models.forecasting.moderntcn import ModernTCNForecaster
from models.forecasting.tcn import TCNForecaster
from models.reconstruction.ae import AutoEncoder
from models.reconstruction.usad import UsadModel

from .robust_dfa_fusion import (
    L, _clip_train, _prep_raw, _prep_raw_normal, _prep_tcn,
    _prep_tcn_normal, _tail_z,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "05_SMAP_MSL" / "telemanom" / "data"
OUT = ROOT / "runs" / "smap_msl_event"
EPOCHS = 3
MAX_BATCHES = 100
MODELS = ["raw", "tcn", "ae", "usad"]


class WindowDS(Dataset):
    def __init__(self, X, seq_len=L):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, i):
        return self.X[i:i + self.seq_len], self.X[i + self.seq_len]


def load_channel(ch):
    tr = pd.read_parquet(BASE / "train" / f"{ch}.parquet")
    te = pd.read_parquet(BASE / "test" / f"{ch}.parquet")
    lab = pd.read_parquet(BASE / "labeled_anomalies.parquet")
    row = lab[lab.chan_id == ch].iloc[0]
    seqs = row.anomaly_sequences
    if isinstance(seqs, str):
        seqs = ast.literal_eval(seqs)
    seqs = list(seqs) if not isinstance(seqs, np.ndarray) else seqs
    if len(seqs) == 2 and np.isscalar(seqs[0]):
        seqs = [seqs]
    feats = [c for c in tr.columns if c != "timestep"]
    Xtr = tr[feats].to_numpy(np.float64)
    Xte = te[feats].to_numpy(np.float64)
    y = np.zeros(len(te), dtype=np.int8)
    for a, b in seqs:
        a, b = int(a), int(b)
        y[max(a, 0):min(b + 1, len(te))] = 1
    # remove constant columns on train
    keep = Xtr.std(0) > 1e-12
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    if Xtr.shape[1] == 0:
        return None
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    return Xtr, Xte, y, row.spacecraft, str(row["class"])


def train_forecast(kind, Xtr, device):
    torch.manual_seed(42)
    cls = TCNForecaster if kind == "tcn" else ModernTCNForecaster
    model = cls(Xtr.shape[1], L, 1).to(device)
    loader = DataLoader(WindowDS(Xtr), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(EPOCHS):
        model.train()
        for i, (x, y) in enumerate(loader):
            if i >= MAX_BATCHES:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x)[:, 0, :], y)
            loss.backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def forecast_err(model, X, device, batch=1024):
    view = np.lib.stride_tricks.sliding_window_view(X, L + 1, axis=0)
    parts = []
    for s in range(0, len(view), batch):
        w = view[s:s + batch].transpose(0, 2, 1)
        x = torch.from_numpy(np.ascontiguousarray(w[:, :-1, :])).float().to(device)
        y = torch.from_numpy(np.ascontiguousarray(w[:, -1, :])).float().to(device)
        pred = model(x)[:, 0, :]
        parts.append(((pred - y) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def train_recon(kind, Xtr, device):
    torch.manual_seed(42)
    if kind == "ae":
        model = AutoEncoder(L, Xtr.shape[1]).to(device)
        loader = DataLoader(WindowDS(Xtr), batch_size=256, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(EPOCHS):
            model.train()
            for i, (x, _) in enumerate(loader):
                if i >= MAX_BATCHES:
                    break
                x = x.to(device)
                opt.zero_grad()
                loss = model.training_loss(x)
                loss.backward()
                opt.step()
    else:
        model = UsadModel(L, Xtr.shape[1]).to(device)
        n = min(len(Xtr) - L, MAX_BATCHES * 256)
        ds = torch.utils.data.Subset(WindowDS(Xtr), list(range(n)))
        loader = DataLoader(ds, batch_size=256, shuffle=True)
        model.fit(loader, epochs=EPOCHS, lr=1e-4, device=device, name="USAD")
    return model.eval()


@torch.no_grad()
def recon_err(model, X, device, batch=1024):
    view = np.lib.stride_tricks.sliding_window_view(X, L, axis=0)
    parts = []
    for s in range(0, len(view), batch):
        w = view[s:s + batch].transpose(0, 2, 1)
        x = torch.from_numpy(np.ascontiguousarray(w)).float().to(device)
        recon = model(x)
        if isinstance(recon, tuple):
            recon = recon[0]
        err = ((recon[:, -1, :] - x[:, -1, :]) ** 2).mean(dim=1)
        parts.append(err.cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def segments(y):
    y = np.asarray(y).astype(int)
    d = np.diff(np.r_[0, y, 0])
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1) - 1
    return list(zip(s, e))


def event_metrics(y, pred, grace=0):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    gt = segments(y)
    al = segments(pred)
    if not gt:
        return None
    hits, delays = 0, []
    for s, e in gt:
        window = pred[s:min(e + grace + 1, len(pred))]
        idx = np.flatnonzero(window)
        if len(idx):
            hits += 1
            delays.append(int(idx[0]))
    correct_alarms = 0
    for s, e in al:
        if any(not (e < gs or s > ge) for gs, ge in gt):
            correct_alarms += 1
    ev_p = correct_alarms / max(len(al), 1)
    ev_r = hits / max(len(gt), 1)
    ev_f1 = (2 * ev_p * ev_r / (ev_p + ev_r)
             if (ev_p + ev_r) > 0 else 0.0)
    normal = y == 0
    fa_events = len(al) - correct_alarms
    fa_per_1000 = (fa_events / max(normal.sum(), 1)) * 1000
    return {
        "n_events": len(gt), "event_hits": hits,
        "event_recall": float(ev_r), "event_precision": float(ev_p),
        "event_f1": float(ev_f1),
        "median_delay": float(np.median(delays)) if delays else float("nan"),
        "fa_per_1000": float(fa_per_1000),
    }


def point_metrics(y, s, thr):
    from sklearn.metrics import (average_precision_score,
                                 precision_recall_fscore_support,
                                 roc_auc_score)
    pred = (s > thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")
    return {"P": float(p), "R": float(r), "F1": float(f1), "AUC": auc,
            "pred": pred}


def run_channel(ch, device):
    loaded = load_channel(ch)
    if loaded is None:
        return []
    Xtr_raw, Xte_raw, y, spacecraft, cls = loaded
    Xtr, Xte = _clip_train(Xtr_raw, Xte_raw)
    rows = []

    # raw COPOD
    from pyod.models.copod import COPOD
    m = COPOD(contamination=0.01, n_jobs=1)
    m.fit(Xtr_raw)
    sn = m.decision_function(Xtr_raw)
    st = m.decision_function(Xte_raw)
    n = _prep_raw_normal(sn)
    t, yy = _prep_raw(st, y)
    zn = _tail_z(n, n)
    zt = _tail_z(t, n)
    thr = float(np.percentile(zn, 99))
    em = event_metrics(yy, (zt > thr).astype(int))
    if em:
        rows.append({"channel": ch, "spacecraft": spacecraft,
                     "class": cls, "model": "raw-COPOD",
                     "threshold": thr, **em,
                     **{f"pt_{k}": v for k, v in
                        point_metrics(yy, zt, thr).items() if k != "pred"}})

    models = [("tcn", "TCN-forecast"), ("ae", "AE-recon"),
              ("usad", "USAD-recon")]
    for kind, label in models:
        try:
            if kind == "tcn":
                model = train_forecast("tcn", Xtr, device)
                en = forecast_err(model, Xtr, device)
                et = forecast_err(model, Xte, device)
            else:
                if kind == "usad":
                    Xtr_u, Xte_u = _minmax(Xtr, Xte)
                else:
                    Xtr_u, Xte_u = Xtr, Xte
                model = train_recon(kind, Xtr_u, device)
                en = recon_err(model, Xtr_u, device)
                et = recon_err(model, Xte_u, device)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            n = _prep_tcn_normal(en)
            t, yy = _prep_tcn(et, y)
            zn = _tail_z(n, n)
            zt = _tail_z(t, n)
            thr = float(np.percentile(zn, 99))
            em = event_metrics(yy, (zt > thr).astype(int))
            if em:
                rows.append({"channel": ch, "spacecraft": spacecraft,
                             "class": cls, "model": label,
                             "threshold": thr, **em,
                             **{f"pt_{k}": v for k, v in
                                point_metrics(yy, zt, thr).items()
                                if k != "pred"}})
        except Exception as exc:
            print(f"  [error] {ch} {kind}: {type(exc).__name__}: {exc}",
                  flush=True)
    return rows


def _minmax(Xtr, Xte):
    lo = Xtr.min(0)
    hi = Xtr.max(0)
    scale = np.maximum(hi - lo, 1e-6)
    return (np.clip((Xtr - lo) / scale, 0, 1),
            np.clip((Xte - lo) / scale, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    lab = pd.read_parquet(BASE / "labeled_anomalies.parquet")
    channels = args.channels or sorted(lab.chan_id.unique())
    channels = channels[args.start:args.end]
    device = resolve_device("auto")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[smap-msl] channels={len(channels)} device={device}", flush=True)
    all_rows = []
    for ch in channels:
        t0 = time.time()
        try:
            rows = run_channel(ch, device)
        except Exception as exc:
            print(f"[error] {ch}: {type(exc).__name__}: {exc}", flush=True)
            rows = []
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(out_dir / f"{ch}.csv", index=False,
                      encoding="utf-8-sig")
            all_rows += rows
            print(f"  {ch}: {len(rows)} models {time.time()-t0:.1f}s",
                  flush=True)
        else:
            print(f"  [skip] {ch}: no rows", flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            out_dir / "ALL.csv", index=False, encoding="utf-8-sig")
    print("[smap-msl] done", flush=True)


if __name__ == "__main__":
    main()
