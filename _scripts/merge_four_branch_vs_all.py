# -*- coding: utf-8 -*-
"""Final comparison: BASE-raw/whiten vs new multi-branch vs FC/RC."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "runs" / "recon_comparison"
OUT_CSV = RC / "FOUR_BRANCH_VS_ALL.csv"
OUT_MD = RC / "FOUR_BRANCH_VS_ALL.md"


def best_row(sub, col="Normal99_F1"):
    return sub.loc[sub[col].idxmax()]


def main():
    fb = pd.read_csv(RC / "FOUR_BRANCH_FUSION.csv")
    fr = pd.read_csv(RC / "FORECAST_RECON_COMPARE.csv")
    datasets = sorted(fb.dataset.unique())
    rows = []
    for ds in datasets:
        s = fb[fb.dataset == ds]
        base_r = s[s.method == "BASE-raw"].iloc[0]
        base_w = s[s.method == "BASE-whiten"].iloc[0]
        new = s[s.method.str.contains(
            "4B|5B|3B|cascade", regex=True)]
        nb = best_row(new)
        f = fr[(fr.dataset == ds) & (fr.family == "forecast")]
        r = fr[(fr.dataset == ds) & (fr.family == "reconstruction")]
        fbp = best_row(f) if len(f) else None
        rbp = best_row(r) if len(r) else None
        rows.append({
            "dataset": ds,
            "base_raw": base_r.method,
            "base_raw_P": base_r.Normal99_Precision,
            "base_raw_R": base_r.Normal99_Recall,
            "base_raw_F1": base_r.Normal99_F1,
            "base_raw_PA": base_r.Normal99_PA_F1,
            "base_raw_AUC": base_r.Normal99_AUC,
            "base_whiten": base_w.method,
            "base_whiten_P": base_w.Normal99_Precision,
            "base_whiten_R": base_w.Normal99_Recall,
            "base_whiten_F1": base_w.Normal99_F1,
            "base_whiten_PA": base_w.Normal99_PA_F1,
            "base_whiten_AUC": base_w.Normal99_AUC,
            "new_method": nb.method,
            "new_P": nb.Normal99_Precision,
            "new_R": nb.Normal99_Recall,
            "new_F1": nb.Normal99_F1,
            "new_PA": nb.Normal99_PA_F1,
            "new_AUC": nb.Normal99_AUC,
            "fc_method": fbp.method if fbp is not None else "—",
            "fc_P": fbp.Normal99_Precision if fbp is not None else float("nan"),
            "fc_R": fbp.Normal99_Recall if fbp is not None else float("nan"),
            "fc_F1": fbp.Normal99_F1 if fbp is not None else float("nan"),
            "fc_PA": fbp.Normal99_PA_F1 if fbp is not None else float("nan"),
            "fc_AUC": fbp.Normal99_AUC if fbp is not None else float("nan"),
            "rc_method": rbp.method if rbp is not None else "—",
            "rc_P": rbp.Normal99_Precision if rbp is not None else float("nan"),
            "rc_R": rbp.Normal99_Recall if rbp is not None else float("nan"),
            "rc_F1": rbp.Normal99_F1 if rbp is not None else float("nan"),
            "rc_PA": rbp.Normal99_PA_F1 if rbp is not None else float("nan"),
            "rc_AUC": rbp.Normal99_AUC if rbp is not None else float("nan"),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    lines = [
        "# BASE-raw/whiten vs multi-branch fusion vs forecast/reconstruction",
        "",
        "- BASE-raw = raw COPOD + TCN-raw COPOD, mean_z.",
        "- BASE-whiten = raw COPOD + TCN-whiten COPOD, max_z.",
        "- New = best of 3B/4B/5B (equal / validation weights) and cascade "
        "on this dataset (post-hoc best row; see `FOUR_BRANCH_FUSION.md`).",
        "- FC/RC = best forecast/reconstruction baseline on this dataset "
        "(post-hoc; existing results, not retrained).",
        "- All metrics at Normal99.",
        "",
        "## Per-dataset comparison", "",
        "| dataset | method class | method | P | R | F1 | PA-F1 | AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in out.iterrows():
        for tag, m, p, rec, f1, pa, auc in [
            ("BASE-raw", r.base_raw, r.base_raw_P, r.base_raw_R,
             r.base_raw_F1, r.base_raw_PA, r.base_raw_AUC),
            ("BASE-whiten", r.base_whiten, r.base_whiten_P,
             r.base_whiten_R, r.base_whiten_F1, r.base_whiten_PA,
             r.base_whiten_AUC),
            ("NEW", r.new_method, r.new_P, r.new_R, r.new_F1, r.new_PA,
             r.new_AUC),
            ("FC-best", r.fc_method, r.fc_P, r.fc_R, r.fc_F1, r.fc_PA,
             r.fc_AUC),
            ("RC-best", r.rc_method, r.rc_P, r.rc_R, r.rc_F1, r.rc_PA,
             r.rc_AUC),
        ]:
            lines.append(
                f"| {r.dataset} | {tag} | {m} | {p:.4f} | {rec:.4f} | "
                f"{f1:.4f} | {pa:.4f} | {auc:.4f} |")

    lines += ["", "## Mean over datasets", "",
              "| class | mean P | mean R | mean F1 | mean PA-F1 | mean AUC |",
              "|---|---|---|---|---|---|"]
    for tag, cols in [
        ("BASE-raw", ["base_raw_P", "base_raw_R", "base_raw_F1",
                      "base_raw_PA", "base_raw_AUC"]),
        ("BASE-whiten", ["base_whiten_P", "base_whiten_R",
                         "base_whiten_F1", "base_whiten_PA",
                         "base_whiten_AUC"]),
        ("NEW", ["new_P", "new_R", "new_F1", "new_PA", "new_AUC"]),
        ("FC-best", ["fc_P", "fc_R", "fc_F1", "fc_PA", "fc_AUC"]),
        ("RC-best", ["rc_P", "rc_R", "rc_F1", "rc_PA", "rc_AUC"]),
    ]:
        vals = [out[c].mean() for c in cols]
        lines.append(
            f"| {tag} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    print(out[["dataset", "new_method", "new_F1", "base_whiten_F1",
               "fc_F1", "rc_F1"]].to_string(index=False))


if __name__ == "__main__":
    main()
