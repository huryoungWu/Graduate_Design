# -*- coding: utf-8 -*-
"""对已保存的 train/test energy 做 anomaly_ratio 敏感性分析。

注意：这里在测试集上扫描多个阈值，只用于分析阈值敏感性；
论文/主表应使用 `run.py` 中固定的 `--anomaly-ratio`。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)

from .run import Solver


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ratios", nargs="*", type=float,
                   default=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    train_energy = np.load(run_dir / "train_energy.npy")
    test_energy = np.load(run_dir / "test_energy.npy")
    test_y = np.load(Path(meta.get("data_dir", "")) / "test_y.npy") \
        if meta.get("data_dir") else None
    if test_y is None:
        # prepare_data 的输出目录与 run 目录的约定
        test_y = np.load(
            Path(__file__).resolve().parent / "data"
            / meta["dataset"] / "test_y.npy")
    n = min(len(test_energy), len(test_y))
    test_energy, test_y = test_energy[:n], test_y[:n].astype(int)
    combined = np.concatenate([train_energy, test_energy])
    auc = float(roc_auc_score(test_y, test_energy))
    ap = float(average_precision_score(test_y, test_energy))

    rows = []
    for r in args.ratios:
        thr = float(np.percentile(combined, 100 - r))
        pred = (test_energy > thr).astype(int)
        pred_adj = Solver._point_adjust(test_y, pred)
        p0, r0, f0, _ = precision_recall_fscore_support(
            test_y, pred, average="binary", zero_division=0)
        p1, r1, f1, _ = precision_recall_fscore_support(
            test_y, pred_adj, average="binary", zero_division=0)
        rows.append({
            "anomaly_ratio": r, "threshold": thr,
            "precision_unadjusted": float(p0),
            "recall_unadjusted": float(r0),
            "f1_unadjusted": float(f0),
            "precision_PA": float(p1),
            "recall_PA": float(r1),
            "f1_PA": float(f1),
            "AUC": auc, "PR_AUC": ap,
        })
    df = pd.DataFrame(rows)
    out = run_dir / "threshold_sweep.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
