# -*- coding: utf-8 -*-
"""Regenerate ALL_DATASETS_ALL_MODELS.md from the master CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "runs" / "recon_comparison"
MASTER = RC / "ALL_DATASETS_ALL_MODELS.csv"
BEST = RC / "ALL_DATASETS_BEST.csv"
OUT = RC / "ALL_DATASETS_ALL_MODELS.md"


def main():
    df = pd.read_csv(MASTER)
    b = pd.read_csv(BEST)
    lines = ["# All datasets: every model and best result", "",
             "- Pointwise datasets: P/R/F1/PA-F1/AUC at Normal99.",
             "- Telemanom channels: event P/R/F1, delay, FA/1000; point F1/AUC are reference only.",
             "- `*` marks the best model per dataset by primary F1.", "",
             "## Dataset inventory", "",
             "| family | datasets | models per dataset |", "|---|---|---|"]
    for fam, s in df.groupby("family"):
        n = s.groupby("dataset").size()
        lines.append(f"| {fam} | {s.dataset.nunique()} | {n.min()}–{n.max()} |")
    lines += ["", "## Best model on 10 clean multivariate datasets", "",
              "| dataset | best model | P | R | F1 | PA-F1 | AUC | ties |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in b[b.family == "industrial_multivariate_clean"].sort_values("dataset").iterrows():
        lines.append(f"| {r.dataset} | {r.best_model} | {r.P:.3f} | {r.R:.3f} | "
                     f"{r.F1:.3f} | {r.PA_F1:.3f} | {r.AUC:.3f} | {int(r.ties)} |")
    lines += ["", "## Degenerate pointwise datasets (do not rank by F1)", "",
              "| dataset | best model | P | R | F1 | PA-F1 | AUC | note |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in b[b.family.isin(["industrial_multivariate_degenerate",
                                 "smap_cache_degenerate"])].sort_values("dataset").iterrows():
        lines.append(f"| {r.dataset} | {r.best_model} | {r.P:.3f} | {r.R:.3f} | "
                     f"{r.F1:.3f} | {r.PA_F1:.3f} | {r.AUC:.3f} | degenerate labels |")
    eb = b[b.family == "telemanom_event"]
    lines += ["", "## Telemanom event channels: best model counts (79)", "",
              "| best model | channels |", "|---|---|"]
    for m, c in eb.best_model.value_counts().items():
        lines.append(f"| {m} | {c} |")
    lines += ["", "Top 15 event channels:", "",
              "| channel | spacecraft | class | best model | event P | event R | event F1 | delay | FA/1000 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in eb.sort_values("F1", ascending=False).head(15).iterrows():
        lines.append(f"| {r.dataset} | {r.spacecraft} | {r['class']} | {r.best_model} | "
                     f"{r.P:.3f} | {r.R:.3f} | {r.F1:.3f} | {r.delay:.1f} | {r.FA1000:.2f} |")
    lines += ["", "## Mean pointwise metrics on 10 clean datasets", "",
              "| model | n | mean P | mean R | mean F1 | mean PA-F1 | mean AUC |",
              "|---|---|---|---|---|---|---|"]
    clean = df[df.family == "industrial_multivariate_clean"]
    for m, s in clean.groupby("model"):
        if s.dataset.nunique() < 5:
            continue
        lines.append(f"| {m} | {s.dataset.nunique()} | {s.P.mean():.3f} | "
                     f"{s.R.mean():.3f} | {s.F1.mean():.3f} | "
                     f"{s.PA_F1.mean():.3f} | {s.AUC.mean():.3f} |")
    lines += ["", "## Mean event metrics on 79 telemanom channels", "",
              "| model | n | mean event P | mean event R | mean event F1 | mean delay | mean FA/1000 |",
              "|---|---|---|---|---|---|---|"]
    ev = df[df.family == "telemanom_event"]
    for m, s in ev.groupby("model"):
        lines.append(f"| {m} | {s.dataset.nunique()} | {s.P.mean():.3f} | "
                     f"{s.R.mean():.3f} | {s.F1.mean():.3f} | "
                     f"{s.delay.mean():.1f} | {s.FA1000.mean():.2f} |")
    lines += ["", "## Per-dataset pointwise tables (all models)", ""]
    dss = sorted(clean.dataset.unique()) + sorted(
        b[b.family.isin(["industrial_multivariate_degenerate",
                         "smap_cache_degenerate"])].dataset.unique())
    for ds in dss:
        s = df[df.dataset == ds].sort_values("F1", ascending=False)
        bm = b[b.dataset == ds].best_model.iloc[0] if len(b[b.dataset == ds]) else ""
        lines += [f"### {ds}", "",
                  "| best | model | P | R | F1 | PA-F1 | AUC | source |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in s.iterrows():
            mark = "*" if r.model == bm else ""
            lines.append(f"| {mark} | {r.model} | {r.P:.3f} | {r.R:.3f} | "
                         f"{r.F1:.3f} | {r.PA_F1:.3f} | {r.AUC:.3f} | {r.source} |")
        lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT, "lines", len(lines))


if __name__ == "__main__":
    main()
