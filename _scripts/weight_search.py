# -*- coding: utf-8 -*-
"""在已保存的分支 z 分数上搜索全局融合权重（只用验证集选择）。"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.common.protocol import select_threshold  # noqa: E402
from repro_anomaly_transformer.benchmark_all_models import _metrics  # noqa: E402

SCORE_DIR = ROOT / "runs" / "robust_dfa_scores"


def load(ds):
    z = np.load(SCORE_DIR / f"{ds}__branches.npz")
    return z


def metric(y, s, thr):
    return _metrics(np.asarray(y), np.asarray(s), float(thr))[0]


def eval_weights(ds, z, w):
    keys = ["r", "e", "t", "w"]
    fn = sum(wi * z[f"{k}_n"] for k, wi in zip(keys, w))
    fv = sum(wi * z[f"{k}_v"] for k, wi in zip(keys, w))
    ft = sum(wi * z[f"{k}_t"] for k, wi in zip(keys, w))
    yv, yt = z["y_val"], z["y_test"]
    from sklearn.metrics import roc_auc_score
    vthr, _ = select_threshold(fv, yv)
    vm = metric(yv, fv, vthr)
    n99 = float(np.percentile(fn, 99))
    mt = metric(yt, ft, n99)
    vt = metric(yt, ft, vthr)
    return {
        "dataset": ds,
        "weights": w,
        "val_AUC": (float(roc_auc_score(yv, fv))
                    if len(np.unique(yv)) > 1 else np.nan),
        "val_F1": vm["F1"], "val_PA_F1": vm["PA_F1"],
        "test_AUC": mt["AUC"],
        "N99_F1": mt["F1"], "N99_PA_F1": mt["PA_F1"],
        "ValF1_F1": vt["F1"], "ValF1_PA_F1": vt["PA_F1"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", required=True)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    data = {ds: load(ds) for ds in args.datasets if
            (SCORE_DIR / f"{ds}__branches.npz").exists()}
    if not data:
        print("no branch files")
        return
    steps = int(round(1.0 / args.step))
    grid = []
    for a in range(steps + 1):
        for b in range(steps + 1 - a):
            for c in range(steps + 1 - a - b):
                d = steps - a - b - c
                grid.append((a / steps, b / steps, c / steps, d / steps))
    rows = []
    for w in grid:
        per = [eval_weights(ds, z, w) for ds, z in data.items()]
        rows.append({
            "weights": w,
            "mean_val_AUC": np.mean([p["val_AUC"] for p in per]),
            "mean_val_F1": np.mean([p["val_F1"] for p in per]),
            "mean_val_PA": np.mean([p["val_PA_F1"] for p in per]),
            "mean_test_AUC": np.mean([p["test_AUC"] for p in per]),
            "mean_N99_F1": np.mean([p["N99_F1"] for p in per]),
            "mean_ValF1_F1": np.mean([p["ValF1_F1"] for p in per]),
            "per": per,
        })
    df = pd.DataFrame(rows)
    print("=== top by mean validation AUC ===")
    for _, r in df.sort_values("mean_val_AUC", ascending=False).head(args.top).iterrows():
        w = r.weights
        print(f"w(r,e,t,w)={w} valAUC={r.mean_val_AUC:.4f} "
              f"valF1={r.mean_val_F1:.4f} valPA={r.mean_val_PA:.4f} | "
              f"testAUC={r.mean_test_AUC:.4f} N99F1={r.mean_N99_F1:.4f} "
              f"ValF1={r.mean_ValF1_F1:.4f}")
        for p in r.per:
            print(f"    {p['dataset']:18s} valAUC={p['val_AUC']:.3f} "
                  f"testAUC={p['test_AUC']:.3f} N99F1={p['N99_F1']:.4f} "
                  f"ValF1={p['ValF1_F1']:.4f}")
        print()
    print("=== top by mean validation PA-F1 ===")
    for _, r in df.sort_values("mean_val_PA", ascending=False).head(5).iterrows():
        print(f"w(r,e,t,w)={r.weights} valPA={r.mean_val_PA:.4f} "
              f"valAUC={r.mean_val_AUC:.4f} testAUC={r.mean_test_AUC:.4f} "
              f"N99F1={r.mean_N99_F1:.4f} ValF1={r.mean_ValF1_F1:.4f}")


if __name__ == "__main__":
    main()
