# -*- coding: utf-8 -*-
"""Old DFA-COPOD fusion vs new robust DFA vs forecast/reconstruction."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "runs" / "recon_comparison"
OUT_CSV = RC / "OLD_VS_NEW_FUSION.csv"
OUT_MD = RC / "OLD_VS_NEW_FUSION.md"


def first_row(sub):
    if sub.empty:
        return None
    metric = "Normal99_F1" if "Normal99_F1" in sub.columns else "F1"
    return sub.loc[sub[metric].idxmax()]


def main():
    old = pd.read_csv(RC / "MULTIDATASET_DFA_FUSION.csv")
    old = old[old.threshold_source == "Normal99"]
    new = pd.read_csv(RC / "BACKBONE_SELECTED.csv")
    fr = pd.read_csv(RC / "FORECAST_RECON_COMPARE.csv")
    dfa = new[new.backbone.isin(["TCN", "ModernTCN", "MICN", "SE-TCN"])]

    rows = []
    datasets = sorted(set(old.dataset) | set(new.dataset))
    for ds in datasets:
        o = old[old.dataset == ds]
        n = dfa[dfa.dataset == ds]
        f = fr[(fr.dataset == ds) & (fr.family == "forecast")]
        r = fr[(fr.dataset == ds) & (fr.family == "reconstruction")]

        ob = first_row(o)
        npol = n[n.backbone == "TCN"]
        nbest = first_row(n)
        fb = first_row(f)
        rb = first_row(r)

        def add(kind, row):
            if row is None:
                return
            rows.append({
                "dataset": ds, "kind": kind,
                "method": row.get("config", row.get("method", "")),
                "P": row.get("Precision", row.get("Normal99_Precision")),
                "R": row.get("Recall", row.get("Normal99_Recall")),
                "F1": row.get("F1", row.get("Normal99_F1")),
                "PA_F1": row.get("PA_F1", row.get("Normal99_PA_F1")),
                "AUC": row.get("AUC", row.get("Normal99_AUC")),
            })

        add("OLD-DFA", ob)
        if not npol.empty:
            add("NEW-DFA-policy", npol.iloc[0])
        add("NEW-DFA-best", nbest)
        add("BEST-FC", fb)
        add("BEST-RC", rb)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    lines = [
        "# Old fusion COPOD vs new robust DFA vs forecast/reconstruction", "",
        "- OLD-DFA: best old configuration per dataset among raw COPOD / "
        "TCN-raw mean_z / TCN-whiten max_z (`MULTIDATASET_DFA_FUSION.csv`, "
        "Normal99).",
        "- NEW-DFA-policy: TCN backbone, policy-selected fusion (deployable).",
        "- NEW-DFA-best: best of TCN/ModernTCN/MICN/SE-TCN selected rows "
        "(post-hoc, diagnostic upper bound).",
        "- BEST-FC / BEST-RC: best forecast / reconstruction baseline per "
        "dataset (post-hoc, diagnostic).",
        "- All metrics at Normal99 threshold.",
        "",
        "## Full comparison", "",
        "| dataset | kind | method | P | R | F1 | PA-F1 | AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, x in out.iterrows():
        lines.append(
            f"| {x.dataset} | {x.kind} | {x.method} | {x.P:.4f} | "
            f"{x.R:.4f} | {x.F1:.4f} | {x.PA_F1:.4f} | {x.AUC:.4f} |")

    valid = ["batadal", "leakdb_hanoi1000", "leakdb_hanoi500",
             "leakdb_net1", "psm", "sml_machine-1-1",
             "sml_machine-1-2", "sml_machine-2-1", "swat"]
    v = out[out.dataset.isin(valid)]
    lines += ["", "## Mean over 9 completed datasets", "",
              "| kind | mean P | mean R | mean F1 | mean PA-F1 | mean AUC |",
              "|---|---|---|---|---|---|"]
    for k, sub in v.groupby("kind", sort=False):
        lines.append(
            f"| {k} | {sub.P.mean():.4f} | {sub.R.mean():.4f} | "
            f"{sub.F1.mean():.4f} | {sub.PA_F1.mean():.4f} | "
            f"{sub.AUC.mean():.4f} |")
    lines += ["", "Note: WADI has OLD/NEW DFA rows only (forecast/"
              "reconstruction baseline run was stopped by memory pressure). "
              "SMAP/BattLeDIM are excluded from the mean because their "
              "pointwise labels are degenerate."]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    print(out.to_string(index=False))
    print("\nMEANS (9 datasets)")
    print(v.groupby("kind", sort=False)[["P", "R", "F1", "PA_F1", "AUC"]]
          .mean().round(4).to_string())


if __name__ == "__main__":
    main()
