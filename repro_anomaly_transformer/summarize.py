# -*- coding: utf-8 -*-
"""汇总 SWaT / WADI 复现结果，生成 RESULTS.md。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "repro_anomaly_transformer" / "runs"
OUT = ROOT / "repro_anomaly_transformer" / "RESULTS.md"

REFERENCE = {
    "swat": {
        "source": "Tencent Cloud 复现博客 (同一官方配置)",
        "precision": 0.8965, "recall": 0.9172, "f1": 0.9067,
        "threshold": 0.003219,
    },
    "wadi": {
        "source": "文献中 Anomaly-Transformer 的 WADI 参考值",
        "precision": 0.8226, "recall": 1.0000, "f1": 0.9026,
        "threshold": None,
    },
}


def main():
    rows = []
    for name in ("swat", "wadi"):
        path = RUNS / name / "metrics.json"
        if not path.exists():
            print(f"[skip] {path} 不存在")
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "dataset": name.upper(),
            "n_test_points": m.get("n_test_points"),
            "positives": m.get("positives"),
            "threshold": m.get("threshold"),
            "precision_PA": m.get("precision"),
            "recall_PA": m.get("recall"),
            "f1_PA": m.get("f1"),
            "precision_raw": m.get("precision_unadjusted"),
            "recall_raw": m.get("recall_unadjusted"),
            "f1_raw": m.get("f1_unadjusted"),
            "AUC": m.get("AUC"),
            "PR_AUC": m.get("PR_AUC"),
        })
    if not rows:
        print("没有结果")
        return
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    lines = ["# Anomaly-Transformer 复现结果", ""]
    lines.append("## 主表")
    lines.append("")
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## 参考值")
    lines.append("")
    for name, ref in REFERENCE.items():
        lines.append(f"- {name.upper()}：{ref['source']}；"
                     f"P={ref['precision']:.4f}, R={ref['recall']:.4f}, "
                     f"F1={ref['f1']:.4f}")
    lines.append("")
    lines.append("注意：主表 F1 是 point-adjusted F1；"
                 "`*_raw` 是未调整的逐点 P/R/F1。")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"已保存: {OUT}")


if __name__ == "__main__":
    main()
