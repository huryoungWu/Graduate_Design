# -*- coding: utf-8 -*-
"""SMAP/MSL per-channel event evaluation: TCN variants + COPOD fusion.

For each channel and each variant V in {TCN, ModernTCN, MICN, SE-TCN}:
- BASE-raw(V)    = raw COPOD + V-raw latent COPOD, mean_z
- BASE-whiten(V) = raw COPOD + V-whiten latent COPOD, max_z
plus raw COPOD as a single-branch baseline.

Metrics are event-level (recall/precision/F1, delay, FA/1000) at Normal99.
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from .backbone_compare import (BACKBONES, build_model, encode, extract_latent,
                               train_model)
from .robust_dfa_fusion import (
    FIT_N, _clip_train, _copod_scores, _fit_copod, _prep_raw,
    _prep_raw_normal, _prep_tcn, _prep_tcn_normal, _tail_z,
)
from .smap_msl_event import event_metrics, load_channel, point_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "smap_msl_tcn_variants"
EPOCHS = 3
MAX_BATCHES = 100


def _latent_scores(kind, model, Xtr, Xte, device):
    Ftr = extract_latent(kind, model, Xtr, device)
    Fte = extract_latent(kind, model, Xte, device)
    m_raw, _ = _fit_copod(Ftr, fit_n=min(FIT_N, len(Ftr)))
    sn = _copod_scores(m_raw, Ftr)
    st = _copod_scores(m_raw, Fte)
    idx = np.random.RandomState(42).choice(
        len(Ftr), min(FIT_N, len(Ftr)), replace=False)
    pca = PCA(n_components=min(32, Ftr.shape[1]), whiten=True,
              random_state=42).fit(Ftr[idx])
    Fw = pca.transform(Ftr)
    Ftw = pca.transform(Fte)
    lo = np.quantile(Fw, 0.001, axis=0)
    hi = np.quantile(Fw, 0.999, axis=0)
    Fw = np.clip(Fw, lo, hi)
    Ftw = np.clip(Ftw, lo, hi)
    m_w, _ = _fit_copod(Fw, fit_n=min(FIT_N, len(Fw)))
    snw = _copod_scores(m_w, Fw)
    stw = _copod_scores(m_w, Ftw)
    return sn, st, snw, stw


def run_channel(ch, device):
    loaded = load_channel(ch)
    if loaded is None:
        return []
    Xtr_raw, Xte_raw, y, spacecraft, cls = loaded
    Xtr, Xte = _clip_train(Xtr_raw, Xte_raw)

    # raw COPOD (point-level, unclipped)
    m_r, _ = _fit_copod(Xtr_raw, fit_n=None)
    sn_r = _copod_scores(m_r, Xtr_raw)
    st_r = _copod_scores(m_r, Xte_raw)
    rn = _prep_raw_normal(sn_r)
    rt, yy = _prep_raw(st_r, y)

    rows = []
    # raw-only baseline
    zn = _tail_z(rn, rn)
    zt = _tail_z(rt, rn)
    thr = float(np.percentile(zn, 99))
    em = event_metrics(yy, (zt > thr).astype(int))
    if em:
        rows.append({"channel": ch, "spacecraft": spacecraft, "class": cls,
                     "model": "raw-COPOD", **em,
                     **{f"pt_{k}": v for k, v in
                        point_metrics(yy, zt, thr).items() if k != "pred"}})

    for kind in BACKBONES:
        try:
            model, train_t = train_model(kind, Xtr, device,
                                         epochs=EPOCHS,
                                         max_batches=MAX_BATCHES)
            sn, st, snw, stw = _latent_scores(kind, model, Xtr, Xte, device)
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            ln = _prep_tcn_normal(sn)
            lt, _ = _prep_tcn(st, y)
            lwn = _prep_tcn_normal(snw)
            lwt, _ = _prep_tcn(stw, y)

            nt = min(len(rt), len(lt), len(lwt), len(yy))
            T = min(len(rn), len(ln), len(lwn))
            rn_t, ln_t, lwn_t = rn[-T:], ln[-T:], lwn[-T:]
            rt_t, lt_t, lwt_t = rt[:nt], lt[:nt], lwt[:nt]
            yy_t = yy[:nt]

            zr_n = _tail_z(rn_t, rn_t)
            zl_n = _tail_z(ln_t, ln_t)
            zw_n = _tail_z(lwn_t, lwn_t)
            zr_t = _tail_z(rt_t, rn_t)
            zl_t = _tail_z(lt_t, ln_t)
            zw_t = _tail_z(lwt_t, lwn_t)

            for label, fn, fv, ft in [
                ("BASE-raw", (zr_n + zl_n) / 2,
                 (zr_t + zl_t) / 2, None),
                ("BASE-whiten", np.maximum(zr_n, zw_n),
                 np.maximum(zr_t, zw_t), None),
            ]:
                thr = float(np.percentile(fn, 99))
                em = event_metrics(yy_t, (fv > thr).astype(int))
                if em:
                    rows.append({
                        "channel": ch, "spacecraft": spacecraft,
                        "class": cls, "model": f"{kind}-{label}",
                        "fit_s": train_t, **em,
                        **{f"pt_{k}": v for k, v in
                           point_metrics(yy_t, fv, thr).items()
                           if k != "pred"}})
        except Exception as exc:
            print(f"  [error] {ch} {kind}: {type(exc).__name__}: {exc}",
                  flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", nargs="*", default=None)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    lab = pd.read_parquet(
        ROOT / "05_SMAP_MSL/telemanom/data/labeled_anomalies.parquet")
    channels = args.channels or sorted(lab.chan_id.unique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[variants] channels={len(channels)} device={device}", flush=True)
    all_rows = []
    for ch in channels:
        t0 = time.time()
        try:
            rows = run_channel(ch, device)
        except Exception as exc:
            print(f"[error] {ch}: {type(exc).__name__}: {exc}", flush=True)
            rows = []
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"{ch}.csv", index=False,
                                      encoding="utf-8-sig")
            all_rows += rows
            print(f"  {ch}: {len(rows)} rows {time.time()-t0:.1f}s",
                  flush=True)
        else:
            print(f"  [skip] {ch}", flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if all_rows:
        pd.DataFrame(all_rows).to_csv(out_dir / "ALL.csv", index=False,
                                      encoding="utf-8-sig")
    print("[variants] done", flush=True)


if __name__ == "__main__":
    main()
