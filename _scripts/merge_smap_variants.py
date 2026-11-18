# -*- coding: utf-8 -*-
"""Merge SMAP/MSL TCN-variant+COPOD results with simple reference models."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VAR = ROOT / "runs" / "smap_msl_tcn_variants"
SIMPLE = ROOT / "runs" / "smap_msl_event"
OUT_CSV = ROOT / "runs" / "recon_comparison" / "SMAP_MSL_TCN_VARIANTS.csv"
OUT_MD = ROOT / "runs" / "recon_comparison" / "SMAP_MSL_TCN_VARIANTS.md"
VARIANTS = ["TCN", "ModernTCN", "MICN", "SE-TCN"]
CONFIGS = ["BASE-raw", "BASE-whiten"]
REF = ["TCN-forecast", "AE-recon", "USAD-recon"]


def read_dir(path, skip=("ALL.csv",)):
    frames = []
    for p in sorted(Path(path).glob("*.csv")):
        if p.name in skip:
            continue
        d = pd.read_csv(p)
        if len(d):
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    var = read_dir(VAR)
    simp = read_dir(SIMPLE)
    simp = simp[simp.model.isin(REF)].copy()
    simp["model"] = "REF-" + simp["model"]
    df = pd.concat([var, simp], ignore_index=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    fusion = [f"{v}-{c}" for v in VARIANTS for c in CONFIGS]
    # best per channel among fusion models and among all
    best_f, best_all = [], []
    for ch, sub in df.groupby("channel", sort=True):
        f = sub[sub.model.isin(fusion)]
        if len(f):
            b = f.loc[f.event_f1.idxmax()]
            if b.event_f1 > 0:
                best_f.append(b)
        b = sub.loc[sub.event_f1.idxmax()]
        best_all.append(b)
    best_f = pd.DataFrame(best_f)
    best_all = pd.DataFrame(best_all)

    lines = [
        "# SMAP/MSL: TCN variants + COPOD, per-channel event evaluation", "",
        "Each channel is an independent univariate telemetry stream. "
        "All metrics are event-level on the test stream at Normal99. ", "",
        "- 4 variants x 2 fusion configs: TCN / ModernTCN / MICN / SE-TCN "
        "with BASE-raw (mean_z) and BASE-whiten (max_z).",
        "- raw-COPOD is the single-branch baseline.",
        "- REF-* are the earlier simple TCN-forecast / AE / USAD results, "
        "included for reference only.",
        "- Event recall = fraction of GT sequences hit; event precision = "
        "fraction of alarm events overlapping GT; delay = first alarm minus "
        "sequence start; FA/1000 = false-alarm events per 1000 normal steps.",
        "",
        "## Mean event metrics across channels", "",
        "| model | channels | mean event R | mean event P | mean event F1 | "
        "mean delay | mean FA/1000 |", "|---|---|---|---|---|---|---|"]
    for m in ["raw-COPOD"] + fusion + ["REF-" + x for x in REF]:
        sub = df[df.model == m]
        if len(sub):
            lines.append(
                f"| {m} | {sub.channel.nunique()} | "
                f"{sub.event_recall.mean():.3f} | "
                f"{sub.event_precision.mean():.3f} | "
                f"{sub.event_f1.mean():.3f} | "
                f"{sub.median_delay.mean():.1f} | "
                f"{sub.fa_per_1000.mean():.2f} |")

    lines += ["", "## Mean event F1 by spacecraft", "",
              "| spacecraft | model | channels | mean event F1 |",
              "|---|---|---|---|"]
    for sc, sub in df.groupby("spacecraft"):
        for m in ["raw-COPOD"] + fusion + ["REF-" + x for x in REF]:
            x = sub[sub.model == m]
            if len(x):
                lines.append(f"| {sc} | {m} | {x.channel.nunique()} | "
                             f"{x.event_f1.mean():.3f} |")

    lines += ["", "## Best fusion model per channel (event F1)", "",
              "| channel | spacecraft | class | best fusion | event R | "
              "event P | event F1 | delay | FA/1000 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in best_f.iterrows():
        lines.append(
            f"| {r.channel} | {r.spacecraft} | {r['class']} | {r.model} | "
            f"{r.event_recall:.3f} | {r.event_precision:.3f} | "
            f"{r.event_f1:.3f} | {r.median_delay:.1f} | "
            f"{r.fa_per_1000:.2f} |")

    lines += ["", "## Tie-aware wins among fusion models (event F1 > 0)", "",
              "| model | wins |", "|---|---|"]
    wins = []
    for ch, sub in df[df.model.isin(fusion)].groupby("channel"):
        mx = sub.event_f1.max()
        if mx <= 0:
            continue
        for m in sub[sub.event_f1 == mx].model:
            wins.append(m)
    for m, c in pd.Series(wins).value_counts().items():
        lines.append(f"| {m} | {c} |")

    # full per-channel tables
    lines += ["", "## Per-channel tables", ""]
    for ch, sub in df.groupby("channel", sort=True):
        sc = sub.spacecraft.iloc[0]
        cls = sub["class"].iloc[0]
        lines += [f"### {ch} ({sc}, {cls})", "",
                  "| model | event R | event P | event F1 | delay | "
                  "FA/1000 | point F1 | point AUC |",
                  "|---|---|---|---|---|---|---|---|"]
        for m in ["raw-COPOD"] + fusion + ["REF-" + x for x in REF]:
            r = sub[sub.model == m]
            if len(r):
                r = r.iloc[0]
                lines.append(
                    f"| {m} | {r.event_recall:.3f} | "
                    f"{r.event_precision:.3f} | {r.event_f1:.3f} | "
                    f"{r.median_delay:.1f} | {r.fa_per_1000:.2f} | "
                    f"{r.pt_F1:.3f} | {r.pt_AUC:.3f} |")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    print("channels", df.channel.nunique())
    print("\nmeans:")
    print(df.groupby("model")[["event_recall", "event_precision", "event_f1",
                               "median_delay", "fa_per_1000"]].mean().round(3)
          .to_string())
    print("\nbest fusion counts:")
    print(best_f.model.value_counts().to_string())


if __name__ == "__main__":
    main()
