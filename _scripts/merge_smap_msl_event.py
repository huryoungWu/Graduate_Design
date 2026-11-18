# -*- coding: utf-8 -*-
"""Merge per-channel SMAP/MSL event results into tables + best-model summary."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "runs" / "smap_msl_event"
OUT_CSV = ROOT / "runs" / "recon_comparison" / "SMAP_MSL_EVENT.csv"
OUT_MD = ROOT / "runs" / "recon_comparison" / "SMAP_MSL_EVENT.md"
MODELS = ["raw-COPOD", "TCN-forecast", "AE-recon", "USAD-recon"]


def main():
    frames = []
    for p in sorted(SRC.glob("*.csv")):
        if p.name == "ALL.csv":
            continue
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if len(d):
            frames.append(d)
    if not frames:
        print("no per-channel csv")
        return
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # best per channel (mark channels where no model detects anything)
    best = []
    wins = []
    for ch, sub in df.groupby("channel", sort=True):
        mx = sub.event_f1.max()
        if mx <= 0:
            b = sub.iloc[0].copy()
            b["model"] = "— (no detection)"
            best.append(b)
            continue
        winners = sub[sub.event_f1 == mx]
        for _, w in winners.iterrows():
            wins.append(w.model)
        b = winners.iloc[0].copy()
        if len(winners) > 1:
            b["model"] = " / ".join(winners.model.tolist())
        best.append(b)
    best = pd.DataFrame(best)

    lines = [
        "# SMAP/MSL per-channel event-level evaluation", "",
        "Each channel is an independent univariate telemetry stream. "
        "Metrics are computed on the test stream; threshold = Normal99 "
        "of the training-normal score distribution.", "",
        "- Event recall: fraction of ground-truth anomaly sequences with "
        "at least one alarm inside the sequence.",
        "- Event precision: fraction of predicted alarm events that overlap "
        "a ground-truth sequence.",
        "- Event F1: harmonic mean of the two.",
        "- Median delay: first alarm index minus sequence start (steps), "
        "only for hit sequences.",
        "- FA/1000: false-alarm events per 1000 normal steps.",
        "- Point F1/AUC are shown for reference only; they are unstable for "
        "short events.", "",
        "## Best model per channel (by event F1)", "",
        "| channel | spacecraft | class | best model | event R | event P | "
        "event F1 | median delay | FA/1000 | point F1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in best.iterrows():
        lines.append(
            f"| {r.channel} | {r.spacecraft} | {r['class']} | {r.model} | "
            f"{r.event_recall:.3f} | {r.event_precision:.3f} | "
            f"{r.event_f1:.3f} | {r.median_delay:.1f} | "
            f"{r.fa_per_1000:.2f} | {r.pt_F1:.3f} |")

    n_pos = (best.model != "— (no detection)").sum()
    lines += ["", f"Channels with at least one detection: {n_pos} / "
              f"{best.channel.nunique()}; the remaining channels have "
              "event F1 = 0 for all four models.", "",
              "## Tie-aware wins among channels with a detection", "",
              "| model | wins |", "|---|---|"]
    for m, c in pd.Series(wins).value_counts().items():
        lines.append(f"| {m} | {c} |")

    # means
    lines += ["", "## Mean event metrics per model", "",
              "| model | channels | mean event R | mean event P | "
              "mean event F1 | median delay (mean) | mean FA/1000 |",
              "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        sub = df[df.model == m]
        lines.append(
            f"| {m} | {sub.channel.nunique()} | {sub.event_recall.mean():.3f} | "
            f"{sub.event_precision.mean():.3f} | {sub.event_f1.mean():.3f} | "
            f"{sub.median_delay.mean():.1f} | {sub.fa_per_1000.mean():.2f} |")

    # spacecraft split
    lines += ["", "## Mean event F1 by spacecraft", "",
              "| spacecraft | model | channels | mean event F1 |",
              "|---|---|---|---|"]
    for sc, sub in df.groupby("spacecraft"):
        for m in MODELS:
            x = sub[sub.model == m]
            if len(x):
                lines.append(f"| {sc} | {m} | {x.channel.nunique()} | "
                             f"{x.event_f1.mean():.3f} |")

    # full per-channel tables
    lines += ["", "## Per-channel tables", ""]
    for ch, sub in df.groupby("channel", sort=True):
        sc = sub.spacecraft.iloc[0]
        cls = sub["class"].iloc[0]
        lines += [f"### {ch} ({sc}, {cls})", "",
                  "| model | event R | event P | event F1 | median delay | "
                  "FA/1000 | point F1 | point AUC |",
                  "|---|---|---|---|---|---|---|---|"]
        for m in MODELS:
            r = sub[sub.model == m]
            if len(r):
                r = r.iloc[0]
                lines.append(
                    f"| {m} | {r.event_recall:.3f} | {r.event_precision:.3f} | "
                    f"{r.event_f1:.3f} | {r.median_delay:.1f} | "
                    f"{r.fa_per_1000:.2f} | {r.pt_F1:.3f} | {r.pt_AUC:.3f} |")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    print("channels:", best.channel.nunique())
    print("best model counts:")
    print(best.model.value_counts().to_string())
    print("\nmeans:")
    print(df.groupby("model")[["event_recall", "event_precision", "event_f1",
                               "median_delay", "fa_per_1000"]].mean().round(3)
          .to_string())


if __name__ == "__main__":
    main()
