# -*- coding: utf-8 -*-
"""跨数据集动态阈值对比：16 数据集 × 8 经典检测器。

动态阈值全部因果（只用过去分数），不使用测试标签。
基线：Normal99 静态阈值；动态策略见
``models.common.dynamic_threshold``。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models.common.config import resolve_device
from models.common.protocol import select_threshold
from models.common.dynamic_threshold import (make_thresholds,
                                             metrics_dynamic,
                                             static_threshold)
from models.classical.isolation_peers import _window, _window_labels

from .multidataset_benchmark import (DATASETS, MODELS, _fit_score,
                                     _load)

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "runs" / "recon_comparison"
OUT_CSV = RC / "MULTIDATASET_DYNAMIC.csv"
OUT_MD = RC / "MULTIDATASET_DYNAMIC.md"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    device = resolve_device(args.device)
    datasets = args.datasets or DATASETS
    models = args.models or MODELS
    rows = []
    print(f"[dyn] device={device} datasets={len(datasets)} "
          f"models={len(models)}", flush=True)

    for ds in datasets:
        if not (ROOT / "runs" / "multidataset" / ds).exists():
            print(f"[skip] {ds}: cache missing")
            continue
        Xtr, Xv, yv, Xt, yt, meta = _load(ds)
        print(f"\n[dyn] {ds}: train={len(Xtr):,} val={len(Xv):,} "
              f"test={len(Xt):,}", flush=True)
        for seed in args.seeds:
            for model in models:
                try:
                    sn, sv, st, fit_s, score_s, nfit = _fit_score(
                        model, Xtr, Xv, Xt, seed, device)
                    nw = _window(sn)
                    vw = _window(sv)
                    tw = _window(st)
                    vy = _window_labels(yv)
                    ty = _window_labels(yt)
                    nv = min(len(vw), len(vy))
                    nt = min(len(tw), len(ty))
                    vw, vy = vw[:nv], vy[:nv]
                    tw, ty = tw[:nt], ty[:nt]
                    warmup = static_threshold(nw, 0.99)
                    nv = vw[vy == 0]
                    vfpr1 = (float(np.percentile(nv, 99))
                             if len(nv) else warmup)
                    vbf1, _ = select_threshold(vw, vy)
                    thrs = {
                        "static_N99": np.full(nt, warmup),
                        "static_VFPR1": np.full(nt, vfpr1),
                        "static_VBF1": np.full(nt, float(vbf1)),
                        **make_thresholds(tw, warmup),
                    }
                    for strat, thr in thrs.items():
                        met = metrics_dynamic(ty, tw, thr)
                        rows.append({
                            "dataset": ds, "model": model,
                            "seed": seed, "strategy": strat,
                            "warmup_N99": warmup,
                            "thr_mean": float(np.mean(thr)),
                            **met, "n": int(nt),
                            "positives": int(ty.sum()),
                            "fit_s": float(fit_s),
                            "score_s": float(score_s),
                            "train_n": int(nfit),
                        })
                    print(f"  {model:6s} {nt} windows, "
                          f"{len(thrs)} strategies", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[fail] {ds} {model}: {exc}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # ---------------- Markdown ----------------
    lines = [
        "# 跨数据集动态阈值对比", "",
        "- 16 数据集 × 8 经典检测器；动态阈值全部因果（只用过去分数）",
        "- 基线：Normal99 静态阈值；动态策略：滚动分位数 / "
        "滚动 mean+k·std / EWMA-Z",
        "- AUC 与阈值策略无关；F1/PA/FPR 会随阈值变化",
        "- 热身阶段使用 Normal99 静态阈值", "",
    ]
    v = out[out.strategy == "static_N99"]
    # 按策略平均
    lines += ["## 1. 各策略平均（全部数据集）", "",
              "| strategy | mean F1 | mean PA-F1 | mean FPR | mean Recall |",
              "|---|---|---|---|---|"]
    agg = out.groupby("strategy").agg(
        mean_F1=("F1", "mean"), mean_PA=("PA_F1", "mean"),
        mean_FPR=("FPR", "mean"), mean_R=("Recall", "mean"),
        datasets=("dataset", "nunique")).sort_values("mean_F1",
                                                     ascending=False)
    for s, r in agg.iterrows():
        lines.append(f"| {s} | {r.mean_F1:.4f} | {r.mean_PA:.4f} | "
                     f"{r.mean_FPR:.4f} | {r.mean_R:.4f} |")
    lines.append("")

    # 每个模型的最佳策略
    lines += ["## 2. 各模型最佳策略（平均 F1）", "",
              "| model | static F1 | best strategy | best F1 | ΔF1 | "
              "best PA-F1 |", "|---|---|---|---|---|---|"]
    for model in models:
        sub = out[out.model == model]
        if sub.empty:
            continue
        base = sub[sub.strategy == "static_N99"]["F1"].mean()
        means = sub.groupby("strategy")["F1"].mean().sort_values(
            ascending=False)
        best = means.index[0]
        best_pa = sub[sub.strategy == best]["PA_F1"].mean()
        lines.append(f"| {model} | {base:.4f} | {best} | "
                     f"{means.iloc[0]:.4f} | "
                     f"{means.iloc[0] - base:+.4f} | {best_pa:.4f} |")
    lines.append("")

    # 每数据集最佳策略
    lines += ["## 3. 各数据集最佳策略（平均 F1，仅分析）", "",
              "| dataset | static F1 | best strategy | best F1 | ΔF1 |",
              "|---|---|---|---|---|"]
    for ds in datasets:
        sub = out[out.dataset == ds]
        if sub.empty:
            continue
        base = sub[sub.strategy == "static_N99"]["F1"].mean()
        means = sub.groupby("strategy")["F1"].mean().sort_values(
            ascending=False)
        lines.append(f"| {ds} | {base:.4f} | {means.index[0]} | "
                     f"{means.iloc[0]:.4f} | "
                     f"{means.iloc[0] - base:+.4f} |")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
