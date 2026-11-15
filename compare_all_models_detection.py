# -*- coding: utf-8 -*-
"""对所有已有 artifact 的模型做检测方案对比。

对比三种配置：
  1. baseline       : 原始分数 + 验证集 max-F1 阈值（旧流程逻辑）
  2. rolling        : 原始分数 + 因果滚动 FPR 阈值
  3. rolling_all    : 全部候选分数（含 channel_norm_*）+ 因果滚动 FPR 阈值

输出：
  runs/comparison_all/
    comparison_all_models.csv           # 所有模型/配置/level 的明细
    comparison_all_models_primary.csv   # 每个模型的主 level（point 优先）
    comparison_all_models_by_day.csv    # 主 level 的逐日指标

运行：
  python -I compare_all_models_detection.py
  python -I compare_all_models_detection.py --target-fpr 0.01 --rolling-window 3600
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.common.detection import DetectionConfig, run_detection_stage


ORIGINAL_SCORES = {
    "forecast": ("mean", "max", "top3_mean"),
    "AE": ("recon_max", "recon_mean",
           "point_recon_mae_last", "point_recon_mse_last"),
    "VAE": ("recon_max", "recon_mean", "kl",
            "point_recon_mae_last", "point_recon_mse_last", "point_kl"),
    "USAD": ("usad_score", "usad_score_paper", "usad_max",
             "recon_max", "recon_mean",
             "point_usad_last", "point_usad_paper_last", "point_usad_max",
             "point_recon_mae_last", "point_recon_mse_last"),
    "LSTMVAE": ("recon_mean", "recon_max", "recon_top3_mean", "kl",
                "point_recon_mse_last", "point_recon_mae_last", "point_kl"),
    "AnomalyTransformer": (
        "recon_max", "assoc_max", "fused",
        "point_recon_mae_last", "point_recon_mse_last", "point_assoc_last"),
    "DCdetector": ("score_max", "score_mean"),
    "SBT": ("score_max", "score_mean"),
}


def original_scores(meta: dict) -> tuple[str, ...] | None:
    task = meta.get("task", "forecast")
    if task == "forecast":
        return ORIGINAL_SCORES["forecast"]
    return ORIGINAL_SCORES.get(meta.get("model", ""), None)


def find_artifacts(root: Path) -> list[Path]:
    out = []
    for meta_path in sorted(root.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "task" not in meta:
            continue
        out.append(meta_path.parent)
    return out


def pick_primary(selected: dict) -> str:
    """按验证集 F1 选择主 level；没有 val_F1 时优先 point。"""
    best, best_v = None, float("-inf")
    for level, row in selected.items():
        v = row.get("val_F1")
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v == v and v > best_v:  # 非 NaN
            best, best_v = level, v
    if best is not None:
        return best
    return "point" if "point" in selected else next(iter(selected))


def main() -> None:
    p = argparse.ArgumentParser(description="全部模型检测方案对比")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--out-dir", default="runs/comparison_all")
    p.add_argument("--target-fpr", type=float, default=0.01)
    p.add_argument("--rolling-window", type=int, default=3600)
    p.add_argument("--no-plot", action="store_true", default=True)
    args = p.parse_args()

    root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, primary_rows, daily_rows = [], [], []
    for artifact in find_artifacts(root):
        meta = json.loads((artifact / "meta.json").read_text(encoding="utf-8"))
        model = meta.get("model", artifact.name)
        task = meta.get("task", "forecast")
        orig = original_scores(meta)
        configs = [
            ("baseline", "val_best_f1", orig),
            ("rolling", "rolling_fpr", orig),
            ("rolling_all", "rolling_fpr", None),
        ]
        print(f"\n===== {model} ({artifact.name}) =====")
        for cfg_name, threshold, scores in configs:
            if scores is not None and len(scores) == 0:
                continue
            cfg = DetectionConfig(
                score_names=scores,
                threshold=threshold,
                target_fpr=args.target_fpr,
                rolling_window=args.rolling_window,
                no_plot=True,
                out_dir=str(artifact / "comparison_all" / cfg_name),
            )
            try:
                result = run_detection_stage(artifact, cfg)
            except Exception as exc:
                print(f"  [skip] {cfg_name}: {exc}")
                continue

            selected = result["selected"]
            if not selected:
                continue
            primary = pick_primary(selected)
            for level, row in selected.items():
                r = dict(row)
                r.update({"model": model, "artifact": artifact.name,
                          "task": task, "config": cfg_name,
                          "primary": (level == primary)})
                rows.append(r)

            prow = dict(selected[primary])
            prow.update({"model": model, "artifact": artifact.name,
                         "task": task, "config": cfg_name})
            primary_rows.append(prow)

            by_day_path = (artifact / "comparison_all"
                           / cfg_name / "detection_metrics_by_day.csv")
            if by_day_path.exists():
                bd = pd.read_csv(by_day_path)
                bd = bd[bd.level == primary].copy()
                bd["model"] = model
                bd["artifact"] = artifact.name
                bd["task"] = task
                bd["config"] = cfg_name
                daily_rows.append(bd)

    if not rows:
        raise RuntimeError("没有可对比的 artifact")

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "comparison_all_models.csv",
                  index=False, encoding="utf-8-sig")
    primary = pd.DataFrame(primary_rows)
    primary.to_csv(out_dir / "comparison_all_models_primary.csv",
                   index=False, encoding="utf-8-sig")
    if daily_rows:
        pd.concat(daily_rows, ignore_index=True).to_csv(
            out_dir / "comparison_all_models_by_day.csv",
            index=False, encoding="utf-8-sig")

    show_cols = ["model", "task", "config", "level", "name",
                 "Precision", "Recall", "F1", "FPR", "AUC", "PR_AUC"]
    print("\n=== 主 level 对比 ===")
    print(primary[show_cols].sort_values(
        ["model", "config"]).to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n结果已保存: {out_dir}")


if __name__ == "__main__":
    main()
