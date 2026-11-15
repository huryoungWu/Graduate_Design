# -*- coding: utf-8 -*-
"""汇总 USAD 论文协议与统一框架协议的结果。

扫描 ``runs/usad_paper_*``（论文协议）和 ``runs/usad_*``（统一框架协议），
输出 ``runs/usad_summary.csv`` 并在终端打印 Markdown 表。

用法::

    python -m models.reconstruction.summarize_usad_runs
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def paper_rows(runs_root: Path) -> list[dict]:
    rows = []
    for d in sorted(runs_root.glob("usad_paper_*")):
        mpath = d / "metrics.json"
        if not mpath.exists():
            continue
        m = _load(mpath)
        tags = [
            ("official", m.get("official_testing_score")),
            ("paper_formula", m.get("paper_formula_score")),
        ]
        for tag, res in tags:
            if not res:
                continue
            w = res["window_level"]
            pa = res["point_level_PA"]
            rows.append({
                "protocol": "paper",
                "run": d.name,
                "dataset": m.get("dataset", ""),
                "score": tag,
                "level": "window",
                "hidden_size": m.get("hidden_size"),
                "z_size": m.get("z_size"),
                "AUC": w["AUC"],
                "PR_AUC": float("nan"),
                "Precision": w["Precision_eer"],
                "Recall": w["Recall_eer"],
                "F1": w["F1_eer"],
                "F1_oracle": w["F1_best_f1_oracle"],
                "PA_F1_oracle": pa["F1_oracle"],
                "PA_Precision_oracle": pa["Precision_oracle"],
                "PA_Recall_oracle": pa["Recall_oracle"],
            })
    return rows


def framework_rows(runs_root: Path) -> list[dict]:
    rows = []
    for d in sorted(runs_root.glob("usad_*")):
        if "paper" in d.name:
            continue
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        for cfg_path in sorted(d.glob("detection*/selected_config.json")):
            cfg = _load(cfg_path)
            for level, row in cfg.get("selected", {}).items():
                rows.append({
                    "protocol": "framework",
                    "run": d.name,
                    "config": cfg_path.parent.name,
                    "dataset": "wadi" if "wadi" in d.name else "swat",
                    "score": row.get("name", ""),
                    "level": level,
                    "hidden_size": None,
                    "z_size": None,
                    "AUC": row.get("AUC"),
                    "PR_AUC": row.get("PR_AUC"),
                    "Precision": row.get("Precision"),
                    "Recall": row.get("Recall"),
                    "F1": row.get("F1"),
                    "F1_oracle": float("nan"),
                    "PA_F1_oracle": float("nan"),
                    "PA_Precision_oracle": float("nan"),
                    "PA_Recall_oracle": float("nan"),
                })
    return rows


def main() -> None:
    rows = paper_rows(ROOT / "runs") + framework_rows(ROOT / "runs")
    if not rows:
        raise SystemExit("没有找到 USAD 结果")
    df = pd.DataFrame(rows)
    out = ROOT / "runs" / "usad_summary.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    show = ["protocol", "run", "config", "dataset", "score", "level",
            "AUC", "Precision", "Recall", "F1", "F1_oracle", "PA_F1_oracle"]
    print(df[show].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
