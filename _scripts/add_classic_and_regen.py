# -*- coding: utf-8 -*-
"""Append classic reconstruction model results to the all-datasets master."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "runs" / "recon_comparison"
CACHE = ROOT / "runs" / "classic_recon_cache"
CH = ROOT / "runs" / "classic_recon_channels"
MASTER = RC / "ALL_DATASETS_ALL_MODELS.csv"
VALID = ["batadal", "leakdb_hanoi1000", "leakdb_hanoi500", "leakdb_net1",
         "psm", "sml_machine-1-1", "sml_machine-1-2", "sml_machine-2-1",
         "swat", "wadi"]
DEG = ["battledim_2018", "battledim_2019"]
SMAPC = ["smap_A-1", "smap_A-2", "smap_P-1", "smap_S-1"]


def fam(ds):
    if ds in VALID:
        return "industrial_multivariate_clean"
    if ds in DEG:
        return "industrial_multivariate_degenerate"
    if ds in SMAPC:
        return "smap_cache_degenerate"
    return "telemanom_event"


def main():
    master = pd.read_csv(MASTER)
    # drop prior classic rows
    master = master[~master.source.isin(
        ["classic_recon_cache", "classic_recon_channels"])].copy()
    rows = []
    for p in sorted(CACHE.glob("*.csv")):
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            rows.append({
                "dataset": r.dataset, "family": fam(r.dataset),
                "protocol": "pointwise", "model": r.model,
                "P": r.Normal99_Precision, "R": r.Normal99_Recall,
                "F1": r.Normal99_F1, "PA_F1": r.Normal99_PA_F1,
                "AUC": r.Normal99_AUC, "delay": np.nan,
                "FA1000": np.nan, "point_F1": np.nan,
                "point_AUC": np.nan, "source": "classic_recon_cache",
                "spacecraft": np.nan, "class": np.nan})
    for p in sorted(CH.glob("*.csv")):
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            rows.append({
                "dataset": r.channel, "family": "telemanom_event",
                "protocol": "event", "model": r.model,
                "P": r.event_precision, "R": r.event_recall,
                "F1": r.event_f1, "PA_F1": np.nan, "AUC": np.nan,
                "delay": r.median_delay, "FA1000": r.fa_per_1000,
                "point_F1": r.pt_F1, "point_AUC": r.pt_AUC,
                "source": "classic_recon_channels",
                "spacecraft": r.spacecraft, "class": r["class"]})
    add = pd.DataFrame(rows)
    df = pd.concat([master, add], ignore_index=True)
    # normalize columns
    for c in ["spacecraft", "class"]:
        if c not in df:
            df[c] = np.nan
    df["family"] = df.dataset.map(fam)
    df.to_csv(MASTER, index=False, encoding="utf-8-sig")
    # best
    best_rows = []
    for ds, s in df.groupby("dataset", sort=True):
        ss = s[s.F1.notna()]
        if len(ss) == 0:
            continue
        mx = ss.F1.max()
        if mx <= 0:
            r = ss.iloc[0].copy()
            r["best_model"] = "—"
            r["ties"] = 0
            best_rows.append(r)
            continue
        w = ss[ss.F1 == mx]
        r = w.iloc[0].copy()
        r["best_model"] = w.iloc[0].model
        r["ties"] = len(w)
        best_rows.append(r)
    b = pd.DataFrame(best_rows)
    b["family"] = b.dataset.map(fam)
    b.to_csv(RC / "ALL_DATASETS_BEST.csv", index=False,
             encoding="utf-8-sig")
    print(f"master rows={len(df)} datasets={df.dataset.nunique()}")
    print(df[df.source.str.startswith("classic")].groupby("source")
          .agg(rows=("model", "size"), datasets=("dataset", "nunique"))
          .to_string())


if __name__ == "__main__":
    main()
