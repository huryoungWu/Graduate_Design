# -*- coding: utf-8 -*-
"""把 ADT 的逐窗口预测映射到逐点，计算 point-wise raw / PA-F1。

ADT 本身输出的是每个窗口的动作（阈值 0/1）。本脚本加载已训练的
``dqn.pt`` 与 ``ae_scores.npy``，重建每个窗口的预测，然后：

- **last-point**：窗口预测赋给窗口最后一点（因果对齐，与项目其它
  模型的 point 级评估一致）；
- **dilated**：窗口预测为 1 时，把窗口内 L 个点都标为 1（更宽松）。

分别计算逐点 raw P/R/F1 与 point-adjusted (PA) P/R/F1，并给出
AE 分数在逐点上的 AUC，便于和窗口级指标对照。

用法::

    python -m models.classical.adt_point_eval \
        --run-dir runs/adt_paper_swat \
        --csv 08_SWaT/SWaT_all_1s.csv
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
import torch
from sklearn.metrics import roc_auc_score

from models.classical.adt import (
    DQNAgent, Env, _numpy_q_params, _numpy_q_values)
from models.reconstruction.usad_extended_metrics import point_adjust, prf


def _window_predictions(model_path, scores, window_labels,
                        l_action=1, k_state=1):
    agent = DQNAgent(state_size=6, action_size=2, device="cpu")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    agent.q_net.load_state_dict(state)
    agent.q_net.eval()
    params = _numpy_q_params(agent)

    env = Env([0, 1], window_labels, scores, verbose=False)
    preds = []
    action_hist = []
    for t in range(len(window_labels)):
        if t % max(1, int(l_action)) == 0:
            a_idx = int(np.argmax(_numpy_q_values(env.state, params)))
            action = [0, 1][a_idx]
            action_hist.append(action)
        else:
            action = action_hist[-1]
        preds.append(1 if scores[t] >= action else 0)
        _, next_state, done = env.do_step(action, t, k_state, 0, 1)
        env.state = next_state
        if done:
            break
    return np.asarray(preds, dtype=np.int8)


def _attack_points(csv_path):
    df = pd.read_csv(csv_path, usecols=["timestamp", "label", "period"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    attack = df[df["period"].astype(str).str.lower() == "attack"]
    attack = attack.reset_index(drop=True)
    attack["row"] = np.arange(len(attack))
    return attack


def _metrics(y, pred):
    p, r, f1 = prf(y, pred)
    return {"Precision": p, "Recall": r, "F1": f1}


def main() -> None:
    ap = argparse.ArgumentParser(description="ADT 逐点评估")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--l-action-infer", type=int, default=1)
    ap.add_argument("--k-state", type=int, default=1)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    scores = np.load(run_dir / "ae_scores.npy").astype(np.float64)
    win_labels = np.load(run_dir / "window_labels.npy").astype(np.int8)
    ts = np.load(run_dir / "window_timestamps.npy")
    preds = _window_predictions(
        run_dir / "dqn.pt", scores, win_labels,
        l_action=args.l_action_infer, k_state=args.k_state)

    attack = _attack_points(args.csv)
    attack_y = attack["label"].to_numpy(np.int8)
    attack_ts = attack["timestamp"].to_numpy()
    L = int(args.seq_len)
    T = len(attack_y)
    if len(preds) != len(scores):
        preds = preds[:len(scores)]

    # 窗口级 sanity check
    win_metrics = _metrics(win_labels[:len(preds)], preds)

    # last-point：窗口 i 的预测赋给第 i+L-1 个点
    point_pred_last = np.zeros(T, dtype=np.int8)
    point_pred_last[L - 1:L - 1 + len(preds)] = preds
    valid = np.arange(T) >= L - 1
    y_valid = attack_y[valid]
    raw_last = _metrics(y_valid, point_pred_last[valid])
    pa_last = _metrics(attack_y, point_adjust(attack_y, point_pred_last))

    # dilated：窗口预测为 1 -> 窗口内 L 个点都标 1
    point_pred_dil = np.zeros(T, dtype=np.int8)
    for i, p in enumerate(preds):
        if p:
            point_pred_dil[i:i + L] = 1
    raw_dil = _metrics(attack_y, point_pred_dil)
    pa_dil = _metrics(attack_y, point_adjust(attack_y, point_pred_dil))

    # AE 分数的逐点 AUC（last-point 对齐）
    point_score = np.full(T, np.nan, dtype=np.float64)
    point_score[L - 1:L - 1 + len(scores)] = scores
    ae_auc = float(roc_auc_score(
        attack_y[valid], point_score[valid]))

    out = {
        "run_dir": str(run_dir),
        "n_windows": int(len(preds)),
        "n_points": int(T),
        "seq_len": L,
        "window_level": win_metrics,
        "point_raw_last_point": raw_last,
        "point_PA_last_point": pa_last,
        "point_raw_dilated": raw_dil,
        "point_PA_dilated": pa_dil,
        "ae_score_point_auc": ae_auc,
        "note": "last-point=窗口预测赋给窗口最后一点；"
                "dilated=窗口报警则窗口内所有点报警；"
                "PA=点级 point-adjust。",
    }
    with open(run_dir / "point_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[ADT-point] window F1          : "
          f"{win_metrics['F1']:.4f}")
    print(f"[ADT-point] point raw (last)   : "
          f"P {raw_last['Precision']:.4f} R {raw_last['Recall']:.4f} "
          f"F1 {raw_last['F1']:.4f}")
    print(f"[ADT-point] point PA  (last)   : "
          f"P {pa_last['Precision']:.4f} R {pa_last['Recall']:.4f} "
          f"F1 {pa_last['F1']:.4f}")
    print(f"[ADT-point] point raw (dilated): "
          f"P {raw_dil['Precision']:.4f} R {raw_dil['Recall']:.4f} "
          f"F1 {raw_dil['F1']:.4f}")
    print(f"[ADT-point] point PA  (dilated): "
          f"P {pa_dil['Precision']:.4f} R {pa_dil['Recall']:.4f} "
          f"F1 {pa_dil['F1']:.4f}")
    print(f"[ADT-point] AE point AUC       : {ae_auc:.4f}")
    print(f"[ADT-point] 已保存: {run_dir / 'point_metrics.json'}")


if __name__ == "__main__":
    main()
