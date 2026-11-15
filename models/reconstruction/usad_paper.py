# -*- coding: utf-8 -*-
"""USAD 论文协议复现（SWaT / WADI）。

与 ``models.reconstruction.usad``（项目统一两阶段协议）的区别：

``usad.py``（统一基准）
    训练：完整 Normal 段；验证/测试：Attack 段前 20% / 后 80%；
    阈值在验证集上选，测试集只报告最终指标。

``usad_paper.py``（论文/官方 notebook 协议）
    1. MinMaxScaler 只在 Normal 段上 fit；
    2. Normal 窗口按时间顺序 80% 训练 / 20% 验证（官方 notebook）；
    3. 整个 Attack 段作为测试集；
    4. 窗口标签 = 窗口内是否含任意攻击点；
    5. 阈值取 ROC 上 TPR = 1 - FPR 的等错误率点（官方 ``ROC()``），
       在该阈值下报告 Precision/Recall/F1；同时给出 oracle 最优 F1；
    6. 额外给出逐点 raw F1 与 PA-F1（point-adjusted F1）。

两种异常分数都评估：

- ``usad_score``：官方 ``testing()``
  ``alpha*MSE(W,AE1(W)) + beta*MSE(W,AE2(AE1(W)))``
- ``usad_score_paper``：论文式 (5)
  ``alpha*MSE(W,AE1(W)) + beta*MSE(W,AE2(W))``

官方 notebook 的 SWaT 默认超参：
    window_size=12, hidden_size=100（z_size=window*100）,
    batch=7919, epochs=100, Adam 默认 lr=1e-3。

注意：仓库提交的 ``model.pth`` 潜变量维度是 120（即 hidden_size=10），
而 notebook 源码当前写的是 100（z=1200）。可用 ``--hidden-size`` 复现
两种配置；通常 z=120 更接近论文的表征瓶颈设置。

用法::

    python -m models.reconstruction.usad_paper \
        --csv 08_SWaT/SWaT_all_1s.csv

    python -m models.reconstruction.usad_paper \
        --csv 09_WADI/WADI_all_1s.csv

    # 只加载已训练权重重新评估（不训练）
    python -m models.reconstruction.usad_paper --eval-only \
        --csv 08_SWaT/SWaT_all_1s.csv \
        --out-dir runs/usad_paper_swat --hidden-size 10

    # 快速自检（1 epoch + 截断数据）
    python -m models.reconstruction.usad_paper --smoke \
        --csv 08_SWaT/SWaT_all_1s.csv
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (auc, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Subset

from models.common.config import resolve_device, set_seed
from models.common.data import ReconstructionDataset
from models.reconstruction.usad import UsadModel

EXCLUDE = ("timestamp", "label", "period")


# ----------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------
def load_periods(csv_path: str | Path):
    """按 period 列返回 (normal_df, attack_df, sensor_cols)。"""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    sensor_cols = [c for c in df.columns if c not in EXCLUDE]
    if not sensor_cols:
        raise ValueError("未找到传感器列")

    if "period" in df.columns:
        period = df["period"].astype(str).str.lower()
        normal = df[period == "normal"].reset_index(drop=True)
        attack = df[period == "attack"].reset_index(drop=True)
    else:
        labels = df["label"].astype(int).to_numpy()
        first = int(np.flatnonzero(labels == 1)[0]) if (labels == 1).any() else len(df)
        normal = df.iloc[:first].reset_index(drop=True)
        attack = df.iloc[first:].reset_index(drop=True)
    if len(attack) == 0:
        raise ValueError("没有 attack 段")

    # 与统一框架 load_swat 一致：缺失点前向填充。
    # WADI 有 4 列共 34 个 NaN（约 3.6e-5，README 四舍五入为 0%），
    # 若不处理，MinMaxScaler 会保留 NaN 并让训练损失变成 NaN。
    n_nan = int(normal[sensor_cols].isna().sum().sum()
                + attack[sensor_cols].isna().sum().sum())
    if n_nan:
        normal[sensor_cols] = normal[sensor_cols].ffill().fillna(0.0)
        attack[sensor_cols] = attack[sensor_cols].ffill().fillna(0.0)
        print(f"[USAD-paper] 传感器列缺失值 {n_nan} 个，已 ffill/填 0")
    return normal, attack, sensor_cols


def _make_loader(dataset, batch_size: int, device, shuffle: bool = False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=(device.type == "cuda"))


# ----------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------
def eer_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """官方 ROC() 的等错误率阈值：TPR = 1 - FPR 的交点。"""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    diff = tpr - (1.0 - fpr)
    idx = np.flatnonzero(np.diff(np.sign(diff)))
    if len(idx) == 0:
        idx = np.array([int(np.argmin(np.abs(diff)))])
    i = int(idx[0])
    return float(thresholds[i])


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray):
    """ORACLE：在测试标签上直接搜 F1 最优阈值（仅作上界参考）。"""
    prec, rec, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    k = int(np.nanargmax(f1[:-1]))
    return float(thresholds[k]), float(f1[k])


def _prf(labels, pred) -> tuple[float, float, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        labels.astype(int), pred.astype(int), average="binary",
        zero_division=0)
    return float(p), float(r), float(f1)


def point_adjust(labels: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """经典 point-adjust：攻击段内只要有一个报警，整段判为检出。"""
    labels = np.asarray(labels).astype(int)
    out = np.asarray(pred).astype(int).copy()
    edges = np.flatnonzero(np.diff(np.r_[0, labels, 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if out[s:e].any():
            out[s:e] = 1
    return out


def sweep_best_pa_f1(labels: np.ndarray, scores: np.ndarray,
                     n_grid: int = 200):
    """在分位点阈值上扫描 PA-F1 的最大值（oracle）。"""
    candidates = np.unique(np.quantile(
        scores, np.linspace(0.0, 100.0, n_grid) / 100.0))
    best = (-1.0, float("nan"), 0.0, 0.0)
    for thr in candidates:
        adj = point_adjust(labels, scores >= thr)
        p, r, f1 = _prf(labels, adj)
        if f1 > best[0]:
            best = (f1, float(thr), p, r)
    return best


def evaluate_scores(name: str, scores: np.ndarray, win_labels: np.ndarray,
                    attack_y: np.ndarray, seq_len: int) -> dict:
    """窗口级 + 逐点级（raw / PA）完整指标。"""
    n_points = len(attack_y)
    valid = np.arange(n_points) >= (seq_len - 1)
    point_score = np.full(n_points, np.nan, dtype=np.float64)
    point_score[seq_len - 1:] = scores[:n_points - seq_len + 1]
    y = attack_y[valid]
    s = point_score[valid]

    # ---- 窗口级 ----
    auc_val = float(roc_auc_score(win_labels, scores))
    thr_eer = eer_threshold(win_labels, scores)
    p_eer, r_eer, f1_eer = _prf(win_labels, scores >= thr_eer)
    thr_best, f1_best = best_f1_threshold(win_labels, scores)
    p_best, r_best, _ = _prf(win_labels, scores >= thr_best)

    # ---- 逐点 raw ----
    p_raw, r_raw, f1_raw = _prf(y, s >= thr_eer)
    thr_pt_best, f1_pt_best = best_f1_threshold(y, s)
    p_pt_best, r_pt_best, _ = _prf(y, s >= thr_pt_best)

    # ---- 逐点 PA ----
    p_pa, r_pa, f1_pa = _prf(y, point_adjust(y, s >= thr_eer))
    p_pa_bw, r_pa_bw, f1_pa_bw = _prf(
        y, point_adjust(y, s >= thr_best))
    f1_pa_or, thr_pa_or, p_pa_or, r_pa_or = sweep_best_pa_f1(y, s)

    return {
        "score_name": name,
        "window_level": {
            "AUC": auc_val,
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
            "Precision_eer": p_raw, "Recall_eer": r_raw, "F1_eer": f1_raw,
            "threshold_best_f1_oracle": thr_pt_best,
            "Precision_best_f1_oracle": p_pt_best,
            "Recall_best_f1_oracle": r_pt_best,
            "F1_best_f1_oracle": f1_pt_best,
        },
        "point_level_PA": {
            "threshold_eer": thr_eer,
            "Precision_eer": p_pa, "Recall_eer": r_pa, "F1_eer": f1_pa,
            "threshold_best_window_f1": thr_best,
            "Precision_best_window_f1": p_pa_bw,
            "Recall_best_window_f1": r_pa_bw,
            "F1_best_window_f1": f1_pa_bw,
            "threshold_oracle": thr_pa_or,
            "Precision_oracle": p_pa_or,
            "Recall_oracle": r_pa_or,
            "F1_oracle": f1_pa_or,
        },
    }


# ----------------------------------------------------------------------
# 绘图
# ----------------------------------------------------------------------
def _plot_roc(labels, scores, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.4f}")
    ax.plot([0, 1], [0, 1], "r:", linewidth=0.8, label="chance")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_score_hist(labels, scores, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bins = np.histogram_bin_edges(scores, bins=80)
    ax.hist(scores[labels == 0], bins=bins, alpha=0.65,
            label="normal windows", color="#4c9be8")
    ax.hist(scores[labels == 1], bins=bins, alpha=0.65,
            label="attack windows", color="#e8574c")
    ax.set_yscale("log")
    ax.set_xlabel("USAD anomaly score")
    ax.set_ylabel("count (log)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="USAD 论文协议复现（SWaT/WADI）")
    p.add_argument("--csv", required=True,
                   help="SWaT_all_1s.csv 或 WADI_all_1s.csv")
    p.add_argument("--seq-len", type=int, default=12, help="官方 window_size=12")
    p.add_argument("--downsample", type=int, default=1,
                   help="降采样步长（1=不降采样；部分文献对 WADI 用 5）")
    p.add_argument("--clip-input", action="store_true",
                   help="把 MinMax 后的输入裁剪到 [0,1]。USAD 的 Decoder "
                        "末端是 Sigmoid，WADI 攻击期存在大量超出 Normal "
                        "范围的值，不裁剪会产生极端重构误差。")
    p.add_argument("--hidden-size", type=int, default=100,
                   help="z_size = seq_len * hidden_size；官方 notebook 100，"
                        "仓库 model.pth 对应 10（z=120）")
    p.add_argument("--batch-size", type=int, default=7919,
                   help="官方 notebook BATCH_SIZE=7919")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="官方使用 torch.optim.Adam 默认学习率 1e-3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", default="",
                   help="为空时默认 runs/usad_paper_<swat|wadi>")
    p.add_argument("--zeroing", choices=["official", "clean"],
                   default="official",
                   help="official=严格复刻官方 training() 的梯度清零方式；"
                        "clean=每个子步骤清空全部梯度（消融）")
    p.add_argument("--val-every", type=int, default=10,
                   help="每多少个 epoch 计算一次验证损失（仅日志用，"
                        "不影响权重；1=每个 epoch）")
    p.add_argument("--eval-only", action="store_true",
                   help="跳过训练，从 <out-dir>/model.pt 加载权重后重新评估")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch + 截断数据，用于流程自检")
    args = p.parse_args()

    if args.smoke:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 2048)

    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset_name = "wadi" if "wadi" in str(args.csv).lower() else "swat"
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path("runs") / f"usad_paper_{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[USAD-paper] dataset={dataset_name} | device={device}")
    print(f"[USAD-paper] window={args.seq_len} | hidden={args.hidden_size} | "
          f"batch={args.batch_size} | epochs={args.epochs} | lr={args.lr}")

    # ---- 读取 + MinMax 归一化（只在 Normal 上 fit）----
    normal, attack, sensor_cols = load_periods(args.csv)
    if args.downsample > 1:
        normal = normal.iloc[::args.downsample].reset_index(drop=True)
        attack = attack.iloc[::args.downsample].reset_index(drop=True)
        print(f"[USAD-paper] 降采样步长={args.downsample}（窗口覆盖 "
              f"{args.seq_len * args.downsample} 秒）")
    if args.smoke:
        normal = normal.iloc[:100_000].reset_index(drop=True)
        attack = attack.iloc[:50_000].reset_index(drop=True)
    n_normal_rows, n_attack_rows = len(normal), len(attack)
    print(f"[USAD-paper] Normal {n_normal_rows:,} 行 | "
          f"Attack {n_attack_rows:,} 行 | 变量 {len(sensor_cols)}")

    # 先取数组/标签再释放 DataFrame（WADI 约 0.8 GB），降低内存峰值
    normal_vals = normal[sensor_cols].to_numpy(np.float64)
    attack_vals = attack[sensor_cols].to_numpy(np.float64)
    normal_y = normal["label"].to_numpy(np.int8, copy=True)
    attack_y = attack["label"].to_numpy(np.int8, copy=True)
    attack_timestamps = attack["timestamp"].to_numpy(copy=True)
    del normal, attack

    scaler = MinMaxScaler()
    normal_X = scaler.fit_transform(normal_vals)
    del normal_vals
    attack_X = scaler.transform(attack_vals)
    del attack_vals
    normal_X = normal_X.astype(np.float32, copy=False)
    attack_X = attack_X.astype(np.float32, copy=False)
    if args.clip_input:
        normal_X = np.clip(normal_X, 0.0, 1.0)
        attack_X = np.clip(attack_X, 0.0, 1.0)
        print("[USAD-paper] 输入已裁剪到 [0,1]")

    # ---- 窗口：Normal 80% 训练 / 20% 验证；Attack 全部测试 ----
    normal_ds = ReconstructionDataset(normal_X, normal_y, args.seq_len)
    n_total = len(normal_ds)
    n_train = int(np.floor(0.8 * n_total))
    train_ds = Subset(normal_ds, range(0, n_train))
    val_ds = Subset(normal_ds, range(n_train, n_total))
    test_ds = ReconstructionDataset(attack_X, attack_y, args.seq_len)
    del normal_X, attack_X

    train_loader = _make_loader(train_ds, args.batch_size, device, shuffle=False)
    val_loader = _make_loader(val_ds, args.batch_size, device)
    test_loader = _make_loader(test_ds, args.batch_size, device)
    print(f"[USAD-paper] windows: train={len(train_ds):,} | "
          f"val={len(val_ds):,} | test={len(test_ds):,}")

    # ---- 模型 + 官方双优化器训练 ----
    model = UsadModel(args.seq_len, len(sensor_cols),
                      hidden_size=args.hidden_size).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"[USAD-paper] w_size={model.w_size} | z_size={model.z_size} | "
          f"参数量={n_params:,}")

    history: list[dict] = []
    train_time = 0.0
    t0 = time.time()
    if args.eval_only:
        ckpt = out_dir / "model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"--eval-only 需要已训练权重: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))
        print(f"[USAD-paper] 已加载权重，跳过训练: {ckpt}")
    else:
        train_time = model.fit(
            train_loader, args.epochs, args.lr, device,
            name="USAD", val_loader=val_loader, history=history,
            zeroing=args.zeroing, val_every=args.val_every)
        print(f"[USAD-paper] 训练耗时: {train_time:.1f}s")

    # ---- 测试集打分（两种分数一次算完）----
    model.eval()
    parts = {"usad_score": [], "usad_score_paper": []}
    label_parts = []
    with torch.no_grad():
        for x, lab in test_loader:
            out = model.score_batch(x.to(device, non_blocking=True))
            for key in parts:
                parts[key].append(
                    out["window"][key].detach().cpu().numpy())
            label_parts.append(np.asarray(lab))
    scores = {k: np.concatenate(v).astype(np.float64)
              for k, v in parts.items()}
    win_labels = np.concatenate(label_parts).astype(np.int8)
    for k, v in scores.items():
        if len(v) != len(win_labels) or len(v) != len(test_ds):
            raise RuntimeError(f"{k}: 窗口数与分数/标签长度不一致")

    # ---- 指标（两种分数）----
    results = {
        name: evaluate_scores(name, arr, win_labels, attack_y,
                              args.seq_len)
        for name, arr in scores.items()
    }
    official = results["usad_score"]
    paper_formula = results["usad_score_paper"]

    metrics = {
        "dataset": dataset_name,
        "protocol": "USAD paper / official notebook",
        "n_features": int(len(sensor_cols)),
        "seq_len": int(args.seq_len),
        "hidden_size": int(args.hidden_size),
        "w_size": int(model.w_size),
        "z_size": int(model.z_size),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "downsample": int(args.downsample),
        "clip_input": bool(args.clip_input),
        "alpha": 0.5,
        "beta": 0.5,
        "zeroing": args.zeroing,
        "n_train_windows": int(len(train_ds)),
        "n_val_windows": int(len(val_ds)),
        "n_test_windows": int(len(test_ds)),
        "train_time_s": float(train_time),
        "official_testing_score": official,
        "paper_formula_score": paper_formula,
        # 兼容旧字段：默认官方 testing() 分数
        "window_level": official["window_level"],
        "point_level": {
            "F1_raw_at_eer_thr":
                official["point_level_raw"]["F1_eer"],
            "Precision_raw_at_eer_thr":
                official["point_level_raw"]["Precision_eer"],
            "Recall_raw_at_eer_thr":
                official["point_level_raw"]["Recall_eer"],
            "F1_point_adjusted_at_eer_thr":
                official["point_level_PA"]["F1_eer"],
            "Precision_point_adjusted_at_eer_thr":
                official["point_level_PA"]["Precision_eer"],
            "Recall_point_adjusted_at_eer_thr":
                official["point_level_PA"]["Recall_eer"],
            "F1_point_adjusted_oracle":
                official["point_level_PA"]["F1_oracle"],
            "n_valid_points": int(len(attack_y) - args.seq_len + 1),
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": float(time.time() - t0),
    }

    # ---- 保存 ----
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    if history:
        pd.DataFrame(history).to_csv(
            out_dir / "training_history.csv", index=False,
            encoding="utf-8-sig")
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "csv": str(args.csv), "seed": args.seed,
            "sensor_cols": sensor_cols,
            "normal_rows": int(n_normal_rows),
            "attack_rows": int(n_attack_rows),
            "scaler": "minmax (fit on normal)",
            "eval_only": bool(args.eval_only),
        }, f, ensure_ascii=False, indent=2)
    np.savez_compressed(
        out_dir / "scaler.npz",
        min_=scaler.min_, scale_=scaler.scale_,
        data_min_=scaler.data_min_, data_max_=scaler.data_max_)
    torch.save(model.state_dict(), out_dir / "model.pt")

    timestamps = attack_timestamps[args.seq_len - 1:]
    pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "score": scores["usad_score"],
        "score_paper": scores["usad_score_paper"],
        "label_window": win_labels.astype(int),
    }).to_csv(out_dir / "window_scores.csv", index=False,
              encoding="utf-8-sig")

    if not args.no_plot:
        try:
            _plot_roc(win_labels, scores["usad_score"],
                      out_dir / "roc.png",
                      f"[USAD] {dataset_name.upper()} window-level ROC")
            _plot_score_hist(win_labels, scores["usad_score"],
                             out_dir / "score_hist.png",
                             f"[USAD] {dataset_name.upper()} score distribution")
        except Exception as exc:
            print(f"[USAD-paper] 绘图跳过: {exc}")

    for tag, res in (("official", official),
                     ("paper-formula", paper_formula)):
        w = res["window_level"]
        pa = res["point_level_PA"]
        print(f"\n[USAD-paper] {tag} 窗口级: "
              f"AUC={w['AUC']:.4f} | EER-thr={w['threshold_eer']:.6g} | "
              f"P={w['Precision_eer']:.4f} | R={w['Recall_eer']:.4f} | "
              f"F1={w['F1_eer']:.4f}")
        print(f"  [oracle] best-window-F1={w['F1_best_f1_oracle']:.4f} | "
              f"PA-F1(oracle)={pa['F1_oracle']:.4f} "
              f"(P={pa['Precision_oracle']:.4f}, "
              f"R={pa['Recall_oracle']:.4f})")
    print(f"[USAD-paper] 结果已保存: {out_dir}")


if __name__ == "__main__":
    main()
