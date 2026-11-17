# -*- coding: utf-8 -*-
"""Merge per-dataset backbone comparison CSVs into one table + markdown."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "runs" / "backbone_compare"
OUT_CSV = ROOT / "runs" / "recon_comparison" / "BACKBONE_COMPARE.csv"
OUT_MD = ROOT / "runs" / "recon_comparison" / "BACKBONE_COMPARE.md"


def main():
    files = sorted(SRC.glob("*.csv"))
    if not files:
        print("no per-dataset csv")
        return
    frames = []
    for p in files:
        try:
            d = pd.read_csv(p)
        except Exception as exc:
            print(f"skip {p}: {exc}")
            continue
        if d.empty:
            continue
        frames.append(d)
    if not frames:
        print("no rows")
        return
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # selected policy row per dataset x backbone
    sel = df[df.selected].copy() if "selected" in df.columns else df.copy()
    sel = sel.sort_values(["dataset", "backbone"])
    metrics = ["Normal99_Precision", "Normal99_Recall", "Normal99_F1",
               "Normal99_FPR", "Normal99_AUC", "Normal99_PA_F1",
               "ValBestF1_Precision", "ValBestF1_Recall",
               "ValBestF1_F1", "ValBestF1_AUC", "val_AUC"]
    metrics = [m for m in metrics if m in sel.columns]
    keep = [c for c in ["dataset", "backbone", "config", "fusion", *metrics]
            if c in sel.columns]
    sel[keep].to_csv(
        ROOT / "runs" / "recon_comparison" / "BACKBONE_SELECTED.csv",
        index=False, encoding="utf-8-sig")

    lines = [
        "# Backbone comparison under robust DFA-COPOD fusion", "",
        "- Backbones: TCN, ModernTCN, MICN, SE-TCN",
        "- Each backbone is trained on clipped normal windows "
        "(10 epochs, max 800 batches/epoch).",
        "- Branches: raw COPOD + backbone latent COPOD (raw/whiten) + "
        "backbone forecast error.",
        "- One policy-selected row per dataset x backbone; policy uses "
        "training statistics only (no test labels).",
        "- Metrics at Normal99 threshold; ValBestF1 uses validation labels.",
        "",
        "## Selected configuration metrics", "",
        "| dataset | backbone | config | fusion | P | R | F1 | PA-F1 | "
        "AUC | ValF1 P | ValF1 R | ValF1 F1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in sel.iterrows():
        lines.append(
            f"| {r.dataset} | {r.backbone} | {r.config} | {r.fusion} | "
            f"{r.Normal99_Precision:.4f} | {r.Normal99_Recall:.4f} | "
            f"{r.Normal99_F1:.4f} | {r.Normal99_PA_F1:.4f} | "
            f"{r.Normal99_AUC:.4f} | {r.ValBestF1_Precision:.4f} | "
            f"{r.ValBestF1_Recall:.4f} | {r.ValBestF1_F1:.4f} |")

    # delta vs TCN
    lines += ["", "## Delta vs TCN (selected policy rows)", "",
              "| dataset | backbone | dF1 | dPA-F1 | dAUC |",
              "|---|---|---|---|---|"]
    for ds, sub in sel.groupby("dataset"):
        base = sub[sub.backbone == "TCN"]
        if base.empty:
            continue
        b = base.iloc[0]
        for _, r in sub.iterrows():
            lines.append(
                f"| {ds} | {r.backbone} | "
                f"{r.Normal99_F1 - b.Normal99_F1:+.4f} | "
                f"{r.Normal99_PA_F1 - b.Normal99_PA_F1:+.4f} | "
                f"{r.Normal99_AUC - b.Normal99_AUC:+.4f} |")

    # best per dataset
    lines += ["", "## Best backbone per dataset (selected rows)", "",
              "| dataset | best F1 | F1 | best AUC | AUC | best PA-F1 | PA-F1 |",
              "|---|---|---|---|---|---|---|"]
    for ds, sub in sel.groupby("dataset"):
        bf = sub.loc[sub.Normal99_F1.idxmax()]
        ba = (sub.loc[sub.Normal99_AUC.idxmax()]
              if sub.Normal99_AUC.notna().any() else sub.iloc[0])
        bp = sub.loc[sub.Normal99_PA_F1.idxmax()]
        lines.append(
            f"| {ds} | {bf.backbone} | {bf.Normal99_F1:.4f} | "
            f"{ba.backbone} | {ba.Normal99_AUC:.4f} | "
            f"{bp.backbone} | {bp.Normal99_PA_F1:.4f} |")

    # overall means
    lines += ["", "## Mean across datasets (selected rows)", "",
              "| backbone | mean F1 | mean PA-F1 | mean AUC | mean ValF1 F1 |",
              "|---|---|---|---|---|"]
    for b, sub in sel.groupby("backbone"):
        lines.append(
            f"| {b} | {sub.Normal99_F1.mean():.4f} | "
            f"{sub.Normal99_PA_F1.mean():.4f} | "
            f"{sub.Normal99_AUC.mean():.4f} | "
            f"{sub.ValBestF1_F1.mean():.4f} |")

    # Exclude degenerate label settings from the headline mean:
    # BattLeDIM labels are single-point leak onsets; SMAP A-1/A-2 have one
    # class after alignment; SMAP S-1 produces no alarms under the policy.
    valid = ["batadal", "leakdb_hanoi1000", "leakdb_hanoi500",
             "leakdb_net1", "psm", "sml_machine-1-1",
             "sml_machine-1-2", "sml_machine-2-1", "swat", "wadi"]
    sv = sel[sel.dataset.isin(valid)]
    lines += ["", "## Mean over 10 non-degenerate datasets", "",
              "| backbone | mean F1 | mean PA-F1 | mean AUC | "
              "mean ValF1 F1 |", "|---|---|---|---|---|"]
    for b, sub in sv.groupby("backbone"):
        lines.append(
            f"| {b} | {sub.Normal99_F1.mean():.4f} | "
            f"{sub.Normal99_PA_F1.mean():.4f} | "
            f"{sub.Normal99_AUC.mean():.4f} | "
            f"{sub.ValBestF1_F1.mean():.4f} |")
    lines += ["", "Note: BattLeDIM (onset labels) and SMAP (single-class / "
              "no-alarm) are excluded from the 10-dataset mean; their rows "
              "are kept above for completeness."]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    print(sel[["dataset", "backbone", *metrics]].to_string(index=False))


if __name__ == "__main__":
    main()
