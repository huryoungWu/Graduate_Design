# -*- coding: utf-8 -*-
"""Merge forecast/reconstruction baselines with the robust DFA fusion rows."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FR = ROOT / "runs" / "forecast_recon_benchmark"
RC = ROOT / "runs" / "recon_comparison"
OUT_CSV = RC / "FORECAST_RECON_COMPARE.csv"
OUT_MD = RC / "FORECAST_RECON_COMPARE.md"

FORECAST = {"TCN": "FC-TCN", "ModernTCN": "FC-ModernTCN",
            "MICN": "FC-MICN", "LSTM": "FC-LSTM",
            "Transformer": "FC-Transformer"}
RECON = {"AE": "RC-AE", "VAE": "RC-VAE",
         "LSTMVAE": "RC-LSTMVAE", "USAD": "RC-USAD"}
DFA = {"TCN": "DFA-TCN", "ModernTCN": "DFA-ModernTCN",
       "MICN": "DFA-MICN", "SE-TCN": "DFA-SE-TCN"}


def main():
    frames = []
    for p in sorted(FR.glob("*.csv")):
        d = pd.read_csv(p)
        if d.empty:
            continue
        d = d.copy()
        d["method"] = d.model.map(lambda m: FORECAST.get(m, RECON.get(m, m)))
        d["family"] = d.model.map(
            lambda m: "forecast" if m in FORECAST else "reconstruction")
        frames.append(d)
    if not frames:
        print("no forecast/recon results")
        return
    fr = pd.concat(frames, ignore_index=True)

    bs = pd.read_csv(RC / "BACKBONE_SELECTED.csv")
    bs = bs[bs.backbone.isin(DFA)].copy()
    bs["method"] = bs.backbone.map(DFA)
    bs["family"] = "robust fusion"
    # rename columns to match forecast/recon table
    rename = {
        "Normal99_Precision": "Normal99_Precision",
        "Normal99_Recall": "Normal99_Recall",
        "Normal99_F1": "Normal99_F1",
        "Normal99_FPR": "Normal99_FPR",
        "Normal99_AUC": "Normal99_AUC",
        "Normal99_PA_F1": "Normal99_PA_F1",
        "ValBestF1_Precision": "ValBestF1_Precision",
        "ValBestF1_Recall": "ValBestF1_Recall",
        "ValBestF1_F1": "ValBestF1_F1",
        "ValBestF1_AUC": "ValBestF1_AUC",
        "ValBestF1_PA_F1": "ValBestF1_PA_F1",
    }
    for c in rename:
        if c not in bs.columns:
            bs[c] = float("nan")
    bs = bs.rename(columns=rename)

    cols = ["dataset", "method", "family", "Normal99_Precision",
            "Normal99_Recall", "Normal99_F1", "Normal99_FPR",
            "Normal99_AUC", "Normal99_PA_F1", "ValBestF1_Precision",
            "ValBestF1_Recall", "ValBestF1_F1", "ValBestF1_AUC",
            "ValBestF1_PA_F1"]
    df = pd.concat([fr[cols], bs[cols]], ignore_index=True)
    # only datasets with completed forecast/reconstruction runs
    keep_ds = set(fr.dataset.unique())
    df = df[df.dataset.isin(keep_ds)].copy()
    order = ["batadal", "leakdb_hanoi500", "leakdb_hanoi1000",
             "leakdb_net1", "psm", "sml_machine-1-1",
             "sml_machine-1-2", "sml_machine-2-1", "swat", "wadi"]
    order = [d for d in order if d in keep_ds]
    df["dataset"] = pd.Categorical(df.dataset, order, ordered=True)
    df = df.sort_values(["dataset", "family", "method"])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    def table(sub, title):
        lines = [f"## {title}", "",
                 "| dataset | method | P | R | F1 | PA-F1 | AUC | "
                 "ValF1 P | ValF1 R | ValF1 F1 |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r.dataset} | {r.method} | "
                f"{r.Normal99_Precision:.4f} | {r.Normal99_Recall:.4f} | "
                f"{r.Normal99_F1:.4f} | {r.Normal99_PA_F1:.4f} | "
                f"{r.Normal99_AUC:.4f} | {r.ValBestF1_Precision:.4f} | "
                f"{r.ValBestF1_Recall:.4f} | {r.ValBestF1_F1:.4f} |")
        lines.append("")
        return lines

    lines = [
        "# Forecast / reconstruction baselines vs robust DFA fusion", "",
        "- Forecasting score: next-step MSE; reconstruction score: last-step "
        "reconstruction MSE.",
        "- All deep models: normal-only training, clipped inputs, "
        "10 epochs / max 800 batches per epoch.",
        "- Scores use the same 12-step alignment and Normal99/ValBestF1 "
        "protocol as the robust DFA fusion.",
        "- DFA rows are the policy-selected robust fusion results from "
        "`BACKBONE_SELECTED.csv`.",
        "- WADI is not included: its baseline run was stopped by system "
        "memory pressure before completion.",
        "- LSTM-VAE is missing for LeakDB hanoi500/hanoi1000 and SWaT: its "
        "input contains NaN on those datasets (reported and skipped).",
        "",
    ]
    lines += table(df[df.family == "forecast"], "Forecasting baselines")
    lines += table(df[df.family == "reconstruction"],
                   "Reconstruction baselines")
    lines += table(df[df.family == "robust fusion"],
                   "Robust DFA fusion (reference)")

    # overall means
    n_ds = len(keep_ds)
    lines += [f"## Mean across {n_ds} completed datasets", "",
              "| method | family | datasets | mean P | mean R | mean F1 | "
              "mean PA-F1 | mean AUC | mean ValF1 F1 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for (fam, m), sub in df.groupby(["family", "method"], sort=False):
        lines.append(
            f"| {m} | {fam} | {sub.dataset.nunique()} | "
            f"{sub.Normal99_Precision.mean():.4f} | "
            f"{sub.Normal99_Recall.mean():.4f} | "
            f"{sub.Normal99_F1.mean():.4f} | "
            f"{sub.Normal99_PA_F1.mean():.4f} | "
            f"{sub.Normal99_AUC.mean():.4f} | "
            f"{sub.ValBestF1_F1.mean():.4f} |")

    # best per dataset
    lines += ["", "## Best method per dataset (Normal99)", "",
              "| dataset | best F1 | F1 | best AUC | AUC | best PA-F1 | PA-F1 |",
              "|---|---|---|---|---|---|---|"]
    for ds, sub in df.groupby("dataset"):
        bf = sub.loc[sub.Normal99_F1.idxmax()]
        ba = sub.loc[sub.Normal99_AUC.idxmax()]
        bp = sub.loc[sub.Normal99_PA_F1.idxmax()]
        lines.append(
            f"| {ds} | {bf.method} | {bf.Normal99_F1:.4f} | "
            f"{ba.method} | {ba.Normal99_AUC:.4f} | "
            f"{bp.method} | {bp.Normal99_PA_F1:.4f} |")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    print(df[["dataset", "family", "method", "Normal99_Precision",
              "Normal99_Recall", "Normal99_F1", "Normal99_AUC"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
