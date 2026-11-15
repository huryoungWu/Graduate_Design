# -*- coding: utf-8 -*-
"""Normal 99% 分位阈值 + 全 Attack 段评估（统一可部署协议）。

协议
----
1. 各模型按自己的方式在 Normal 段上训练/拟合（使用已训练权重）；
2. 对 **Normal 段** 的窗口分数取 99% 分位作为固定阈值
   （目标 FPR = 1%，不使用任何 Attack 标签）；
3. 在整个 Attack 段上评估窗口级 P/R/F1/FPR、AUC/PR-AUC，
   并给出固定阈值下的逐点 PA-F1。

这解决了“全 Attack + oracle 阈值”不可部署、以及“20/80 验证集”
只用 80% Attack 的问题。ADT 的 DQN 依赖推理时真实标签反馈，
无法用 Normal 阈值校准，因此本表使用 **ADT 的 AE-only 主干**。

用法::

    python -m models.classical.normal99_full_attack
    python -m models.classical.normal99_full_attack --models USAD LSTM-VAE
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader

from models.classical.adt import AE, WindowDataset
from models.classical.dagmm import DAGMM
from models.common.config import resolve_device, set_seed
from models.common.data import ReconstructionDataset
from models.reconstruction.lstm_vae import LSTMVAE
from models.reconstruction.usad import UsadModel
from models.reconstruction.usad_extended_metrics import point_adjust, prf
from models.reconstruction.usad_paper import load_periods

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = ["USAD", "LSTM-VAE", "IsolationForest", "DAGMM", "ADT-AE"]
DATASETS = {"SWaT": "08_SWaT/SWaT_all_1s.csv",
            "WADI": "09_WADI/WADI_all_1s.csv"}

MODEL_PATHS = {
    ("USAD", "SWaT"): ("runs/usad_paper_swat_z120/model.pt", 10),
    ("USAD", "WADI"): ("runs/usad_paper_wadi_e25/model.pt", 10),
    ("LSTM-VAE", "SWaT"): ("runs/lstmvae_swat/model.pt", None),
    ("LSTM-VAE", "WADI"): ("runs/lstmvae_wadi/model.pt", None),
    ("DAGMM", "SWaT"): ("runs/dagmm_swat/model.pt", None),
    ("DAGMM", "WADI"): ("runs/dagmm_wadi/model.pt", None),
    ("ADT-AE", "SWaT"): ("runs/adt_swat/ae.pt", None),
    ("ADT-AE", "WADI"): ("runs/adt_wadi/ae.pt", None),
}


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def _scale(normal, attack, cols, kind: str, clip: bool):
    if kind == "minmax":
        scaler = MinMaxScaler()
        lo, hi = 0.0, 1.0
    elif kind == "minmax11":
        scaler = MinMaxScaler(feature_range=(-1, 1))
        lo, hi = -1.0, 1.0
    else:
        scaler = StandardScaler()
        lo, hi = None, None
    nx = scaler.fit_transform(
        normal[cols].to_numpy(np.float64)).astype(np.float32)
    ax = scaler.transform(
        attack[cols].to_numpy(np.float64)).astype(np.float32)
    if clip and lo is not None:
        nx = np.clip(nx, lo, hi)
        ax = np.clip(ax, lo, hi)
    return nx, ax, scaler


def _win_mean(point_scores, seq_len: int):
    ps = np.asarray(point_scores, dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(
        ps, seq_len).mean(axis=1)


def _win_labels(labels, seq_len: int):
    return np.lib.stride_tricks.sliding_window_view(
        np.asarray(labels).astype(np.int8), seq_len).max(axis=1)


@torch.no_grad()
def _score_window_model(model, X, labels, seq_len, device,
                        score_key, batch_size=8192):
    ds = ReconstructionDataset(X, labels, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    model.eval()
    parts = []
    for x, _ in loader:
        out = model.score_batch(x.to(device, non_blocking=True))
        parts.append(out["window"][score_key].cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


@torch.no_grad()
def _score_ae_windows(model, X, seq_len, device, batch_size=8192):
    ds = WindowDataset(X, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    model.eval()
    parts = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        recon = model(x)
        parts.append(((recon - x) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


# ----------------------------------------------------------------------
# 各模型打分
# ----------------------------------------------------------------------
def score_usad(dataset, normal, attack, cols, L, device):
    clip = dataset == "WADI"
    nx, ax, _ = _scale(normal, attack, cols, "minmax", clip)
    path, hidden = MODEL_PATHS[("USAD", dataset)]
    model = UsadModel(L, len(cols), hidden_size=hidden,
                      clip_input=clip).to(device)
    model.load_state_dict(torch.load(
        ROOT / path, map_location=device, weights_only=True))
    y = attack["label"].to_numpy(np.int8)
    ns = _score_window_model(model, nx, np.zeros(len(nx), np.int8),
                             L, device, "usad_score")
    as_ = _score_window_model(model, ax, y, L, device, "usad_score")
    del nx, ax
    return ns, as_


def score_lstmvae(dataset, normal, attack, cols, L, device):
    clip = dataset == "WADI"
    nx, ax, _ = _scale(normal, attack, cols, "minmax11", clip)
    path, _ = MODEL_PATHS[("LSTM-VAE", dataset)]
    model = LSTMVAE(len(cols), L, intermediate_dim=15, z_dim=3,
                    clip_input=clip).to(device)
    model.load_state_dict(torch.load(
        ROOT / path, map_location=device, weights_only=True))
    y = attack["label"].to_numpy(np.int8)
    ns = _score_window_model(model, nx, np.zeros(len(nx), np.int8),
                             L, device, "recon_mean")
    as_ = _score_window_model(model, ax, y, L, device, "recon_mean")
    del nx, ax
    return ns, as_


def score_iforest(dataset, normal, attack, cols, L, device):
    nx, ax, _ = _scale(normal, attack, cols, "standard", False)
    model = IsolationForest(
        n_estimators=100, max_samples="auto", contamination="auto",
        max_features=1.0, bootstrap=False, random_state=42, n_jobs=-1)
    model.fit(nx)
    ns = _win_mean(-model.score_samples(nx), L)
    as_ = _win_mean(-model.score_samples(ax), L)
    del nx, ax
    return ns, as_


def score_dagmm(dataset, normal, attack, cols, L, device):
    nx, ax, _ = _scale(normal, attack, cols, "standard", False)
    path, _ = MODEL_PATHS[("DAGMM", dataset)]
    model = DAGMM(len(cols), comp_hiddens=(60, 30, 10, 1),
                  est_hiddens=(10, 4), dropout=0.5,
                  lambda1=0.1, lambda2=1e-4).to(device)
    model.load_state_dict(torch.load(
        ROOT / path, map_location=device, weights_only=True))
    model.fixed = True
    ns = _win_mean(model.score_points(nx, batch_size=8192), L)
    as_ = _win_mean(model.score_points(ax, batch_size=8192), L)
    del nx, ax
    return ns, as_


def score_adt_ae(dataset, normal, attack, cols, L, device):
    nx, ax, _ = _scale(normal, attack, cols, "minmax", True)
    path, _ = MODEL_PATHS[("ADT-AE", dataset)]
    model = AE(L * len(cols), latent_dim=100).to(device)
    model.load_state_dict(torch.load(
        ROOT / path, map_location=device, weights_only=True))
    ns = _score_ae_windows(model, nx, L, device)
    as_ = _score_ae_windows(model, ax, L, device)
    del nx, ax
    return ns, as_


SCORERS = {
    "USAD": score_usad,
    "LSTM-VAE": score_lstmvae,
    "IsolationForest": score_iforest,
    "DAGMM": score_dagmm,
    "ADT-AE": score_adt_ae,
}

SCALERS = {
    "USAD": "minmax",
    "LSTM-VAE": "minmax11",
    "IsolationForest": "standard",
    "DAGMM": "standard",
    "ADT-AE": "minmax",
}

NOTES = {
    ("USAD", "WADI"): "WADI 使用输入裁剪；模型 e25（未崩溃）",
    ("ADT-AE", "SWaT"): "AE-only 主干；DQN 依赖标签反馈，不适用",
    ("ADT-AE", "WADI"): "AE-only 主干；DQN 依赖标签反馈，不适用",
}


# ----------------------------------------------------------------------
# 评估
# ----------------------------------------------------------------------
# 模型训练时是否留出了 Normal：USAD/LSTM-VAE 用 80% Normal 训练，
# ADT-AE 用 70% Normal 训练。其余模型（IF/DAGMM）用全部 Normal。
HELDOUT_FRAC = {"USAD": 0.80, "LSTM-VAE": 0.80, "ADT-AE": 0.70}


def _row_for_threshold(dataset, model_name, method, thr,
                       as_, win_y, attack_y, L, n, auc, ap,
                       n_normal, n_holdout):
    pred = (as_ >= thr).astype(np.int8)
    p, r, f1 = prf(win_y, pred)
    tp = int(((pred == 1) & (win_y == 1)).sum())
    fp = int(((pred == 1) & (win_y == 0)).sum())
    fn = int(((pred == 0) & (win_y == 1)).sum())
    tn = int(((pred == 0) & (win_y == 0)).sum())
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    # 固定阈值下的逐点 PA-F1：窗口分数对齐到窗口最后一点
    T = len(attack_y)
    point_pred = np.zeros(T, dtype=np.int8)
    m = min(n, T - L + 1)
    point_pred[L - 1:L - 1 + m] = pred[:m]
    pa_pred = point_adjust(attack_y, point_pred)
    p_pa, r_pa, f1_pa = prf(attack_y, pa_pred)

    return {
        "model": model_name,
        "dataset": dataset,
        "protocol": "calibrated threshold + full Attack",
        "method": method,
        "threshold": float(thr),
        "normal_windows": int(n_normal),
        "heldout_normal_windows": int(n_holdout),
        "attack_windows": int(n),
        "AUC": auc,
        "PR_AUC": ap,
        "Precision": p,
        "Recall": r,
        "F1": f1,
        "FPR": fpr,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "PA_F1_fixed_threshold": f1_pa,
        "PA_Precision": p_pa,
        "PA_Recall": r_pa,
        "note": NOTES.get((model_name, dataset), ""),
    }


def evaluate_methods(dataset, model_name, ns, as_, attack_y, L):
    """同一份分数上比较多种阈值校准方法。"""
    win_y = _win_labels(attack_y, L)
    n = min(len(as_), len(win_y))
    as_ = as_[:n]
    win_y = win_y[:n]
    auc = (float(roc_auc_score(win_y, as_))
           if len(np.unique(win_y)) > 1 else float("nan"))
    ap = (float(average_precision_score(win_y, as_))
          if len(np.unique(win_y)) > 1 else float("nan"))

    rows = []
    frac = HELDOUT_FRAC.get(model_name)
    n_hold = int(len(ns) * frac) if frac else 0

    methods = [
        ("q90_all", float(np.quantile(ns, 0.90))),
        ("q95_all", float(np.quantile(ns, 0.95))),
        ("q99_all", float(np.quantile(ns, 0.99))),
        ("q99.9_all", float(np.quantile(ns, 0.999))),
    ]
    if frac and n_hold < len(ns):
        hold = ns[n_hold:]
        methods += [
            ("q99_heldout", float(np.quantile(hold, 0.99))),
            ("q99.9_heldout", float(np.quantile(hold, 0.999))),
        ]
    # 部署流上的报警预算（不使用标签）：阈值 = 分数的 (1 - 预算) 分位
    methods += [
        ("alarm5", float(np.quantile(as_, 0.95))),
        ("alarm5.86", float(np.quantile(as_, 1.0 - 0.0586))),
        ("alarm9.28", float(np.quantile(as_, 1.0 - 0.0928))),
        ("alarm10", float(np.quantile(as_, 0.90))),
        ("alarm12", float(np.quantile(as_, 0.88))),
        ("alarm15", float(np.quantile(as_, 0.85))),
    ]

    for name, thr in methods:
        rows.append(_row_for_threshold(
            dataset, model_name, name, thr, as_, win_y, attack_y, L,
            n, auc, ap, len(ns), n_hold))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Normal 99% 阈值 + 全 Attack 统一评估")
    p.add_argument("--models", nargs="*", default=MODEL_NAMES,
                   choices=MODEL_NAMES)
    p.add_argument("--datasets", nargs="*", default=list(DATASETS),
                   choices=list(DATASETS))
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", default="runs/normal99_full_attack")
    args = p.parse_args()

    set_seed(42)
    device = resolve_device(args.device)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for dataset in args.datasets:
        csv_path = str(ROOT / DATASETS[dataset])
        print(f"\n===== {dataset} =====", flush=True)
        normal, attack, cols = load_periods(csv_path)
        print(f"[{dataset}] Normal {len(normal):,} | "
              f"Attack {len(attack):,} | 变量 {len(cols)}", flush=True)
        attack_y = attack["label"].to_numpy(np.int8)
        L = 12

        for model_name in args.models:
            print(f"[{dataset}] scoring {model_name} ...", flush=True)
            try:
                ns, as_ = SCORERS[model_name](
                    dataset, normal, attack, cols, L, device)
                model_rows = evaluate_methods(
                    dataset, model_name, ns, as_, attack_y, L)
                rows.extend(model_rows)
                base = next(r for r in model_rows if r["method"] == "q99_all")
                f1s = " | ".join(
                    f"{r['method']}={r['F1']:.3f}" for r in model_rows)
                print(f"  {model_name}: AUC={base['AUC']:.4f} | {f1s}",
                      flush=True)
            except Exception as exc:
                print(f"  [ERROR] {model_name}: {exc}", flush=True)
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del normal, attack, cols
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison_methods.csv", index=False,
              encoding="utf-8-sig")
    # 保留原来的 q99 baseline 文件，兼容旧报告/脚本
    df[df["method"] == "q99_all"].to_csv(
        out_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    show = ["dataset", "model", "method", "threshold", "AUC", "PR_AUC",
            "Precision", "Recall", "F1", "FPR", "PA_F1_fixed_threshold"]
    print("\n=== 阈值校准方法对比：全 Attack ===")
    print(df[show].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n已保存: {out_dir / 'comparison_methods.csv'}")


if __name__ == "__main__":
    main()
