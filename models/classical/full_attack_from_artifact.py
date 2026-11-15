# -*- coding: utf-8 -*-
"""从统一框架 artifact（val.npz + test.npz）计算全 Attack 段指标。

适用场景：模型训练/推理已经完成，但重新加载原始数据做全 Attack 段
评估会占用过多内存（例如 WADI）。val（Attack 前 20%）+ test（后 80%）
的窗口分数拼接起来即覆盖整个 Attack 段；本脚本只读取：

- ``<artifact>/val.npz``、``<artifact>/test.npz``（窗口/逐点分数、标签、
  时间戳）；
- 原始 CSV 的 ``timestamp, label, period`` 三列（用于逐点标签）。

输出与 ``usad_paper.evaluate_scores`` 相近的窗口/逐点/PA 指标，
以及一份窗口分数 CSV。

用法::

    python -m models.classical.full_attack_from_artifact \
        --artifact-dir runs/dagmm_wadi \
        --csv 09_WADI/WADI_all_1s.csv \
        --out-dir runs/dagmm_paper_wadi
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from models.classical.isolation_forest import _dataset_name
from models.reconstruction.usad_extended_metrics import (
    best_f1, eer_threshold, point_adjust, prf, sweep_best_pa_f1)


def _load_split(artifact_dir: Path, split: str):
    with np.load(artifact_dir / f"{split}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    p = argparse.ArgumentParser(description="从 artifact 计算全 Attack 段指标")
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-dir", default="",
                   help="为空时默认 runs/full_attack_<artifact名>")
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--no-pa-sweep", action="store_true",
                   help="跳过 PA oracle 阈值扫描（更快）")
    args = p.parse_args()

    artifact = Path(args.artifact_dir)
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path("runs") / f"full_attack_{artifact.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    val = _load_split(artifact, "val")
    test = _load_split(artifact, "test")
    win_names = sorted(k[len("win_"):] for k in val if k.startswith("win_"))
    if not win_names:
        raise RuntimeError("artifact 中没有 win_* 窗口分数")
    ts = np.concatenate([val["timestamp"], test["timestamp"]])
    win_labels = np.concatenate(
        [val["label_window"], test["label_window"]]).astype(np.int8)

    # 逐点标签（只读三列，内存很小）
    df = pd.read_csv(args.csv, usecols=["timestamp", "label", "period"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    attack = df[df["period"].astype(str).str.lower() == "attack"].copy()
    attack = attack.reset_index(drop=True)
    attack["row"] = np.arange(len(attack))
    attack_y = attack["label"].to_numpy(np.int8)
    row_of_ts = {t: int(r) for t, r in zip(attack["timestamp"], attack["row"])}

    rows = np.array([row_of_ts.get(pd.Timestamp(t), -1) for t in ts])
    if (rows < 0).any():
        raise RuntimeError("有窗口时间戳无法在 Attack 段中对齐")

    results = {}
    score_table = {"timestamp": pd.to_datetime(ts)}
    for name in win_names:
        score = np.concatenate([val[f"win_{name}"],
                                test[f"win_{name}"]]).astype(np.float64)
        score_table[f"score_{name}"] = score
        # 逐点分数：窗口结束时刻赋值，其余点零阶保持（跨界缺口也填上）
        point_score = np.full(len(attack_y), np.nan, dtype=np.float64)
        point_score[rows] = score
        point_score = pd.Series(point_score).ffill().bfill().to_numpy()

        # 窗口级
        auc = float(roc_auc_score(win_labels, score))
        ap = float(average_precision_score(win_labels, score))
        thr_eer = eer_threshold(win_labels, score)
        p_eer, r_eer, f1_eer = prf(win_labels, score >= thr_eer)
        thr_best, f1_best = best_f1(win_labels, score)
        p_best, r_best, _ = prf(win_labels, score >= thr_best)

        # 逐点 raw
        p_raw, r_raw, f1_raw = prf(attack_y, point_score >= thr_eer)
        thr_pt, f1_pt = best_f1(attack_y, point_score)
        p_pt, r_pt, _ = prf(attack_y, point_score >= thr_pt)

        # 逐点 PA
        p_pa, r_pa, f1_pa = prf(
            attack_y, point_adjust(attack_y, point_score >= thr_eer))
        p_pab, r_pab, f1_pab = prf(
            attack_y, point_adjust(attack_y, point_score >= thr_best))
        if args.no_pa_sweep:
            f1_pao, thr_pao, p_pao, r_pao = (float("nan"),) * 4
        else:
            f1_pao, thr_pao, p_pao, r_pao = sweep_best_pa_f1(
                attack_y, point_score)

        results[name] = {
            "window_level": {
                "AUC": auc, "PR_AUC": ap,
                "threshold_eer": thr_eer,
                "Precision_eer": p_eer, "Recall_eer": r_eer,
                "F1_eer": f1_eer,
                "threshold_best_f1_oracle": thr_best,
                "Precision_best_f1_oracle": p_best,
                "Recall_best_f1_oracle": r_best,
                "F1_best_f1_oracle": f1_best,
            },
            "point_level_raw": {
                "threshold_eer": thr_eer,
                "Precision_eer": p_raw, "Recall_eer": r_raw,
                "F1_eer": f1_raw,
                "threshold_best_f1_oracle": thr_pt,
                "Precision_best_f1_oracle": p_pt,
                "Recall_best_f1_oracle": r_pt,
                "F1_best_f1_oracle": f1_pt,
            },
            "point_level_PA": {
                "threshold_eer": thr_eer,
                "Precision_eer": p_pa, "Recall_eer": r_pa,
                "F1_eer": f1_pa,
                "threshold_best_window_f1": thr_best,
                "Precision_best_window_f1": p_pab,
                "Recall_best_window_f1": r_pab,
                "F1_best_window_f1": f1_pab,
                "threshold_oracle": thr_pao,
                "Precision_oracle": p_pao,
                "Recall_oracle": r_pao,
                "F1_oracle": f1_pao,
            },
        }

    metrics = {
        "dataset": _dataset_name(args.csv),
        "protocol": "full attack segment from val+test artifact",
        "artifact_dir": str(artifact),
        "n_windows": int(len(win_labels)),
        "n_points": int(len(attack_y)),
        "scores": results,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame(score_table).to_csv(
        out_dir / "window_scores.csv", index=False, encoding="utf-8-sig")

    for name, res in results.items():
        w = res["window_level"]
        pa = res["point_level_PA"]
        print(f"[artifact-eval] {name}: AUC {w['AUC']:.4f} | "
              f"F1@EER {w['F1_eer']:.4f} | "
              f"F1@oracle {w['F1_best_f1_oracle']:.4f} | "
              f"PA-F1@oracle {pa['F1_oracle']:.4f}")
    print(f"[artifact-eval] 已保存: {out_dir}")


if __name__ == "__main__":
    main()
