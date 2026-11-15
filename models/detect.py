# -*- coding: utf-8 -*-
"""独立的异常检测入口（Stage 2）。

只读取 Stage 1 的 artifact，不导入模型、不重新读取原始 CSV。

示例::

    python -m models.detect --artifact-dir runs/lstm_swat \
        --scores mean,max,top3_mean --threshold val_best_f1

    python -m models.detect --artifact-dir runs/ae_swat \
        --scores recon_mean,recon_max --threshold val_fpr --target-fpr 0.01

    python -m models.detect --artifact-dir runs/ae_swat \
        --normalize rolling --smooth-window 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.common.detection import DetectionConfig, run_detection_stage


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2 异常检测（读取 artifact）")
    p.add_argument("--artifact-dir", required=True,
                   help="Stage 1 保存的 artifact 目录")
    p.add_argument("--scores", default="",
                   help="只评估这些分数（逗号分隔；空=全部）")
    p.add_argument("--threshold", default="val_best_f1",
                   choices=["val_best_f1", "val_fpr", "rolling_fpr", "fixed"],
                   help="阈值选择方式")
    p.add_argument("--target-fpr", type=float, default=0.01,
                   help="val_fpr / rolling_fpr 的目标误报率")
    p.add_argument("--rolling-window", type=int, default=3600,
                   help="rolling_fpr 的因果滚动窗口（样本点/秒）")
    p.add_argument("--fixed-threshold", type=float, default=None,
                   help="threshold=fixed 时的阈值")
    p.add_argument("--normalize", default="none",
                   choices=["none", "rolling"],
                   help="分数归一化方式")
    p.add_argument("--normalize-window", type=int, default=3600,
                   help="rolling 归一化的窗口（样本点）")
    p.add_argument("--smooth-window", type=int, default=1,
                   help="滑动平均窗口（1=不平滑）")
    p.add_argument("--select-by", default="val_f1",
                   choices=["val_f1", "val_pr_auc", "val_auc"],
                   help="在验证集上选择评分配置的依据")
    p.add_argument("--out-dir", default="",
                   help="检测结果输出目录；为空时默认 <artifact>/detection")
    p.add_argument("--no-plot", action="store_true",
                   help="不生成按天分数图")
    return p


def main() -> None:
    args = build_parser().parse_args()
    names = None
    if args.scores.strip():
        names = tuple(s.strip() for s in args.scores.split(",") if s.strip())
    cfg = DetectionConfig(
        score_names=names,
        threshold=args.threshold,
        target_fpr=args.target_fpr,
        rolling_window=args.rolling_window,
        fixed_threshold=args.fixed_threshold,
        normalize=args.normalize,
        normalize_window=args.normalize_window,
        smooth_window=args.smooth_window,
        select_by=args.select_by,
        no_plot=args.no_plot,
        out_dir=args.out_dir,
    )
    run_detection_stage(args.artifact_dir, cfg)


if __name__ == "__main__":
    main()
