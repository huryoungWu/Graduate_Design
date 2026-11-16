# -*- coding: utf-8 -*-
"""静态阈值完整报告：每个模型 × 每个数据集的 P/R/F1/FPR/AUC/PA-F1。

输入：
- runs/recon_comparison/MULTIDATASET.csv（8 个经典检测器，静态口径）
- runs/recon_comparison/MULTIDATASET_DFA_FUSION.csv（3 个 DFA 配置，静态口径）

输出：
- runs/recon_comparison/MULTIDATASET_STATIC_ALL.csv
- runs/recon_comparison/MULTIDATASET_STATIC_ALL.md
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "runs" / "recon_comparison"
CLASSICAL = RC / "MULTIDATASET.csv"
FUSION = RC / "MULTIDATASET_DFA_FUSION.csv"
OUT_CSV = RC / "MULTIDATASET_STATIC_ALL.csv"
OUT_MD = RC / "MULTIDATASET_STATIC_ALL.md"

SOURCES = ["ValFPR1", "Normal99", "ValBestF1"]
COLS = ["family", "dataset", "model", "threshold_source",
        "Precision", "Recall", "F1", "FPR", "AUC", "PR_AUC",
        "PA_F1", "n", "positives"]


def _load(path: Path, family: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.threshold_source.isin(SOURCES)].copy()
    d["family"] = family
    if family == "fusion":
        d = d.rename(columns={"config": "model"})
    return d[COLS]


def main() -> None:
    frames = []
    if CLASSICAL.exists():
        frames.append(_load(CLASSICAL, "classical"))
    if FUSION.exists():
        frames.append(_load(FUSION, "fusion"))
    if not frames:
        raise SystemExit("missing static result files")
    all_df = pd.concat(frames, ignore_index=True)
    order = {"classical": 0, "fusion": 1}
    all_df["_o"] = all_df.family.map(order)
    all_df = all_df.sort_values(
        ["_o", "threshold_source", "dataset", "model"]).drop(
        columns="_o")
    all_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    lines = [
        "# 静态阈值完整结果：每个模型 × 每个数据集", "",
        "- 阈值口径：ValFPR1 / Normal99 / ValBestF1（均为静态、无测试标签选择）",
        "- 指标：Precision / Recall / F1 / FPR / AUC / PR-AUC / PA-F1",
        "- classical：8 个经典检测器；fusion：raw COPOD + 两个 TCN 配置",
        "- SMAP A-1/A-2 的 fusion 因 60 窗对齐后无正例未评估", "",
    ]
    for source in SOURCES:
        sub = all_df[all_df.threshold_source == source]
        lines += [f"## {source}", ""]
        for family, fname in (("classical", "经典检测器"),
                              ("fusion", "DFA 融合配置")):
            s = sub[sub.family == family]
            if s.empty:
                continue
            lines += [f"### {fname}", "",
                      "| dataset | model | Precision | Recall | F1 | "
                      "FPR | AUC | PR-AUC | PA-F1 |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for _, r in s.iterrows():
                lines.append(
                    f"| {r['dataset']} | {r['model']} | "
                    f"{r['Precision']:.4f} | {r['Recall']:.4f} | "
                    f"{r['F1']:.4f} | {r['FPR']:.4f} | "
                    f"{r['AUC']:.4f} | {r['PR_AUC']:.4f} | "
                    f"{r['PA_F1']:.4f} |")
            lines.append("")

    # 每数据集最佳（ValFPR1）
    v = all_df[(all_df.threshold_source == "ValFPR1")
               & (all_df.family == "classical")]
    lines += ["## 各数据集最佳经典模型（ValFPR1，按 F1）", "",
              "| dataset | best model | Precision | Recall | F1 | PA-F1 |",
              "|---|---|---|---|---|---|"]
    for ds, g in v.groupby("dataset"):
        r = g.loc[g.F1.idxmax()]
        lines.append(
            f"| {ds} | {r['model']} | {r['Precision']:.4f} | "
            f"{r['Recall']:.4f} | {r['F1']:.4f} | {r['PA_F1']:.4f} |")
    lines.append("")

    # 模型平均
    lines += ["## 经典模型平均（ValFPR1）", "",
              "| model | mean Precision | mean Recall | mean F1 | "
              "mean PA-F1 | mean AUC |", "|---|---|---|---|---|---|"]
    agg = v.groupby("model").agg(
        P=("Precision", "mean"), R=("Recall", "mean"),
        F1=("F1", "mean"), PA=("PA_F1", "mean"),
        AUC=("AUC", "mean")).sort_values("F1", ascending=False)
    for m, r in agg.iterrows():
        lines.append(f"| {m} | {r.P:.4f} | {r.R:.4f} | {r.F1:.4f} | "
                     f"{r.PA:.4f} | {r.AUC:.4f} |")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
