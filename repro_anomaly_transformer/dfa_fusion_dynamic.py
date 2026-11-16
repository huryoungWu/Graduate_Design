# -*- coding: utf-8 -*-
"""对已保存的 DFA 融合分数做动态阈值对比。

需要先运行 ``multidataset_dfa_fusion.py`` 生成
``runs/dfa_fusion_scores/<dataset>__<config>.npz``。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.common.dynamic_threshold import (make_thresholds,
                                             metrics_dynamic,
                                             rolling_quantile,
                                             static_threshold)
from models.common.protocol import select_threshold

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "runs" / "recon_comparison"
SCORE_DIR = ROOT / "runs" / "dfa_fusion_scores"
OUT_CSV = RC / "MULTIDATASET_DFA_FUSION_DYNAMIC.csv"
OUT_MD = RC / "MULTIDATASET_DFA_FUSION_DYNAMIC.md"


def main() -> None:
    files = sorted(SCORE_DIR.glob("*.npz"))
    rows = []
    for path in files:
        stem = path.stem
        if "__" not in stem:
            continue
        ds, config = stem.split("__", 1)
        z = np.load(path)
        normal = np.asarray(z["normal"], dtype=np.float64)
        val = np.asarray(z["val"], dtype=np.float64)
        test = np.asarray(z["test"], dtype=np.float64)
        yv = np.asarray(z["y_val"], dtype=np.int64)
        yt = np.asarray(z["y_test"], dtype=np.int64)
        if yt.sum() == 0:
            print(f"[skip] {stem}: no positives")
            continue
        warmup = static_threshold(normal, 0.99)
        nv = val[yv == 0]
        vfpr1 = float(np.percentile(nv, 99)) if len(nv) else warmup
        vbf1, _ = select_threshold(val, yv)
        thrs = {
            "static_N99": np.full(len(test), warmup),
            "static_VFPR1": np.full(len(test), vfpr1),
            "static_VBF1": np.full(len(test), float(vbf1)),
            **make_thresholds(test, warmup),
        }
        rq99_1000 = rolling_quantile(test, 0.99, 1000, warmup)
        rq995_1000 = rolling_quantile(test, 0.995, 1000, warmup)
        thrs["hybrid_max_rq99_w1000"] = np.maximum(warmup, rq99_1000)
        thrs["hybrid_min_rq99_w1000"] = np.minimum(warmup, rq99_1000)
        thrs["hybrid_max_rq995_w1000"] = np.maximum(warmup, rq995_1000)
        for strat, thr in thrs.items():
            met = metrics_dynamic(yt, test, thr)
            rows.append({
                "dataset": ds, "config": config, "strategy": strat,
                "warmup_N99": warmup,
                "thr_mean": float(np.mean(thr)),
                **met, "n": int(len(yt)),
                "positives": int(yt.sum()),
            })
    if not rows:
        raise SystemExit(f"no score files in {SCORE_DIR}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    lines = [
        "# 跨数据集 DFA 融合：动态阈值对比", "",
        "- 两个配置：TCN-raw mean_z / TCN-whiten max_z（含 raw COPOD 基线）",
        "- 动态阈值全部因果（只用过去分数）；热身使用 Normal99",
        "- 基线：static_N99 / static_VFPR1 / static_VBF1",
        "- AUC 与阈值无关；F1/PA/FPR 随阈值变化", "",
    ]
    lines += ["## 1. 策略平均（全部 dataset/config）", "",
              "| strategy | mean F1 | mean PA-F1 | mean FPR | mean Recall |",
              "|---|---|---|---|---|"]
    agg = out.groupby("strategy").agg(
        mean_F1=("F1", "mean"), mean_PA=("PA_F1", "mean"),
        mean_FPR=("FPR", "mean"), mean_R=("Recall", "mean"),
        n=("dataset", "nunique")).sort_values("mean_F1",
                                              ascending=False)
    for s, r in agg.iterrows():
        lines.append(f"| {s} | {r.mean_F1:.4f} | {r.mean_PA:.4f} | "
                     f"{r.mean_FPR:.4f} | {r.mean_R:.4f} |")
    lines.append("")

    lines += ["## 2. 各配置最佳策略（平均 F1）", "",
              "| config | static N99 F1 | best strategy | best F1 | ΔF1 | "
              "best PA-F1 |", "|---|---|---|---|---|---|"]
    for config in sorted(out.config.unique()):
        sub = out[out.config == config]
        base = sub[sub.strategy == "static_N99"]["F1"].mean()
        means = sub.groupby("strategy")["F1"].mean().sort_values(
            ascending=False)
        best = means.index[0]
        best_pa = sub[sub.strategy == best]["PA_F1"].mean()
        lines.append(f"| {config} | {base:.4f} | {best} | "
                     f"{means.iloc[0]:.4f} | {means.iloc[0] - base:+.4f} | "
                     f"{best_pa:.4f} |")
    lines.append("")

    lines += ["## 3. 各数据集最佳策略（仅分析）", "",
              "| dataset | config | static F1 | best strategy | best F1 | "
              "ΔF1 |", "|---|---|---|---|---|---|"]
    for (ds, config), sub in out.groupby(["dataset", "config"]):
        base = sub[sub.strategy == "static_N99"]["F1"].mean()
        means = sub.groupby("strategy")["F1"].mean().sort_values(
            ascending=False)
        lines.append(f"| {ds} | {config} | {base:.4f} | {means.index[0]} | "
                     f"{means.iloc[0]:.4f} | "
                     f"{means.iloc[0] - base:+.4f} |")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
