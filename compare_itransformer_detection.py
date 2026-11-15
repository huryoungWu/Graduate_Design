# -*- coding: utf-8 -*-
"""对比 iTransformer 的原始基线 vs 改进检测方案。

基线：mean 分数 + 验证集 max-F1 阈值（旧流程逻辑）。
改进：
  - val_fpr：验证集正常点固定 FPR 阈值
  - rolling_fpr：因果滚动分位数阈值
  - max / top3_mean / channel_norm_*：不同通道聚合方式
  - channel_norm_*：先用验证集正常点逐通道 99% 分位归一化，再聚合

输出：
  <artifact-dir>/comparison/
    comparison_overall.csv
    comparison_by_day.csv
    <combo>/...

运行：
  python -I compare_itransformer_detection.py
  python -I compare_itransformer_detection.py --artifact-dir runs/itransformer_swat
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.common.detection import DetectionConfig, run_detection_stage


COMBOS = [
    # name, score, threshold mode
    ("baseline_mean_valf1", "mean", "val_best_f1"),
    ("mean_valfpr", "mean", "val_fpr"),
    ("mean_rollfpr", "mean", "rolling_fpr"),
    ("max_rollfpr", "max", "rolling_fpr"),
    ("top3_rollfpr", "top3_mean", "rolling_fpr"),
    ("cnorm_max_valf1", "channel_norm_max", "val_best_f1"),
    ("cnorm_max_valfpr", "channel_norm_max", "val_fpr"),
    ("cnorm_max_rollfpr", "channel_norm_max", "rolling_fpr"),
    ("cnorm_top3_rollfpr", "channel_norm_top3", "rolling_fpr"),
    ("cnorm_mean_rollfpr", "channel_norm_mean", "rolling_fpr"),
]


def main() -> None:
    p = argparse.ArgumentParser(description="iTransformer 检测方案对比")
    p.add_argument("--artifact-dir", default="runs/itransformer_swat")
    p.add_argument("--out-dir", default="")
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--rolling-window", type=int, default=3600)
    p.add_argument("--plot", action="store_true",
                   help="为每个组合生成分数图和 True vs Pred 图（较慢）")
    args = p.parse_args()

    artifact_dir = Path(args.artifact_dir)
    out_dir = Path(args.out_dir) if args.out_dir else artifact_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, daily = [], []
    for name, score, mode in COMBOS:
        cfg = DetectionConfig(
            score_names=(score,), threshold=mode,
            target_fpr=args.target_fpr,
            rolling_window=args.rolling_window,
            no_plot=not args.plot, out_dir=str(out_dir / name))
        result = run_detection_stage(artifact_dir, cfg)
        row = result["selected"]["point"].copy()
        row.update({"combo": name, "score": score,
                    "threshold_mode": mode})

        score_csv = out_dir / name / f"scores_point_{score}.csv"
        sc = pd.read_csv(score_csv)
        row["Accuracy"] = float(
            (sc["pred_anomaly"].to_numpy() == sc["label"].to_numpy()).mean())
        rows.append(row)

        by_day = pd.read_csv(out_dir / name / "detection_metrics_by_day.csv")
        by_day["combo"] = name
        daily.append(by_day)

    summary = pd.DataFrame(rows)
    cols = ["combo", "score", "threshold_mode", "Precision", "Recall",
            "F1", "FPR", "AUC", "PR_AUC", "Accuracy", "threshold"]
    summary = summary[cols].sort_values("F1", ascending=False)
    summary.to_csv(out_dir / "comparison_overall.csv",
                   index=False, encoding="utf-8-sig")

    daily_df = pd.concat(daily, ignore_index=True)
    daily_df.to_csv(out_dir / "comparison_by_day.csv",
                    index=False, encoding="utf-8-sig")

    print("\n=== 总体对比 ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n结果已保存: {out_dir}")


if __name__ == "__main__":
    main()
