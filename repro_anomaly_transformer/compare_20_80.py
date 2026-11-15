# -*- coding: utf-8 -*-
"""汇总 20/80 冻结测试协议下的模型对比。

数据来源：
  - 项目统一逐点表：runs/recon_comparison/comparison_unified_point.csv
  - IsolationForest 参考：runs/recon_comparison/comparison_reference_iforest.csv
  - Anomaly-Transformer（论文实现）：repro_anomaly_transformer/runs/swat_20_80/metrics.json
  - SBT：repro_anomaly_transformer/runs/sbt_20_80/metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "recon_comparison" / "COMPARISON_20_80.csv"
MD = ROOT / "runs" / "recon_comparison" / "COMPARISON_20_80.md"


def _row(model, level, score, thr, auc, pr, p, r, f1, fpr,
         pa_f1, note=""):
    return {
        "model": model, "level": level, "score": score,
        "threshold": thr, "AUC": auc, "PR_AUC": pr,
        "Precision": p, "Recall": r, "F1": f1, "FPR": fpr,
        "PA_F1": pa_f1, "note": note,
    }


def main():
    rows = []
    point_path = ROOT / "runs/recon_comparison/comparison_unified_point.csv"
    if point_path.exists():
        df = pd.read_csv(point_path)
        for _, x in df.iterrows():
            rows.append(_row(
                x["model"], "point", x["score"], x["threshold"],
                x["AUC"], x["PR_AUC"], x["Precision"], x["Recall"],
                x["F1"], x["FPR"], x.get("PA_F1", np.nan),
                "project unified point_recon_mse_last"))
    ifr_path = ROOT / "runs/recon_comparison/comparison_reference_iforest.csv"
    if ifr_path.exists():
        x = pd.read_csv(ifr_path).iloc[0]
        rows.append(_row(
            "IsolationForest", "point", x["score"], x["threshold"],
            x["AUC"], x["PR_AUC"], x["Precision"], x["Recall"],
            x["F1"], x["FPR"], x.get("PA_F1", np.nan),
            "IF reference (same timestamps/val_best_f1)"))

    at_path = (ROOT / "repro_anomaly_transformer" / "runs"
               / "swat_20_80" / "metrics.json")
    if at_path.exists():
        m = json.loads(at_path.read_text(encoding="utf-8"))
        score_csv = at_path.parent / "test_scores.csv"
        fpr = float("nan")
        if score_csv.exists():
            s = pd.read_csv(score_csv)
            neg = s["label"] == 0
            fpr = float(((s["pred"] == 1) & neg).sum() / max(neg.sum(), 1))
        rows.append(_row(
            "AnomalyTransformer(paper)", "point", "official_energy",
            m.get("threshold"), m.get("AUC"), m.get("PR_AUC"),
            m.get("precision_unadjusted"), m.get("recall_unadjusted"),
            m.get("f1_unadjusted"), fpr, m.get("f1"),
            "20/80 val_best_f1; PA_F1 is paper-style"))

    sbt_path = (ROOT / "repro_anomaly_transformer" / "runs"
                / "sbt_20_80" / "metrics.json")
    if sbt_path.exists():
        m = json.loads(sbt_path.read_text(encoding="utf-8"))
        rows.append(_row(
            "SBT", "window60", m.get("best_score"), m.get("threshold"),
            m.get("AUC"), m.get("PR_AUC"), m.get("Precision"),
            m.get("Recall"), m.get("F1"), m.get("FPR"),
            m.get("PA_F1"), "60-step window score; 20/80 val_best_f1"))

    if not rows:
        print("没有可汇总的结果")
        return
    df = pd.DataFrame(rows)
    # 逐点主表按 F1 排序，SBT 单独放窗口行
    df["order"] = df["level"].map({"point": 0, "window60": 1})
    df = df.sort_values(["order", "F1"], ascending=[True, False]) \
           .drop(columns="order")
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    cols = ["model", "level", "score", "threshold", "AUC", "PR_AUC",
            "Precision", "Recall", "F1", "FPR", "PA_F1", "note"]
    lines = ["# 20/80 冻结测试协议对比", "",
             "- 训练：Normal only",
             "- 验证：Attack 前 20%，用于 early stopping/阈值",
             "- 测试：Attack 后 80%，冻结，只在最后评估",
             "- 阈值：验证集 `val_best_f1`",
             "- `F1` 是标准 F1；`PA_F1` 是 point-adjusted F1，单独列",
             ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, x in df.iterrows():
        vals = []
        for c in cols:
            v = x[c]
            if isinstance(v, (float, np.floating)):
                vals.append("nan" if np.isnan(v) else f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines += ["", "注意：SBT 只输出 60 步窗口分数，因此 level=window60；"
              "其余模型是逐点评估。"]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"已保存: {OUT}")
    print(f"已保存: {MD}")


if __name__ == "__main__":
    main()
