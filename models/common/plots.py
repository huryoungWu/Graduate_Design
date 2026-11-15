# -*- coding: utf-8 -*-
"""共享按天绘图：预测类逐通道对比图、重构类异常分数图。

TP/FP/TN/FN 不再用散点压在曲线上，而是画成坐标区四色背景色带；
True/Pred 用“粗浅色底线 + 细深色虚线”的双层画法，
即使两条曲线几乎重合也能分辨。
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.dates import date2num
from matplotlib.patches import Patch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 0=TN（正常且判正常），1=TP（攻击且判攻击），
# 2=FP（正常但误报），3=FN（攻击但漏报）
_STATE_COLORS = [
    (0.12, 0.47, 0.71, 0.10),   # TN: 浅蓝
    (0.00, 0.60, 0.00, 0.18),   # TP: 绿
    (0.85, 0.00, 0.00, 0.16),   # FP: 红
    (0.55, 0.20, 0.70, 0.18),   # FN: 紫
]
_STATE_CMAP = ListedColormap(_STATE_COLORS)


def _safe(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


def _state_handles(tp=None, fp=None, tn=None, fn=None):
    """图例里的 TN/TP/FP/FN 色块；传入当天数组时附带计数。"""
    def _count(arr) -> str:
        if arr is None:
            return ""
        return f" ({int(np.asarray(arr).sum()):,})"

    return [
        Patch(facecolor=_STATE_COLORS[0][:3] + (0.30,), edgecolor="none",
              label=f"TN (normal){_count(tn)}"),
        Patch(facecolor=_STATE_COLORS[1][:3] + (0.35,), edgecolor="none",
              label=f"TP (detected){_count(tp)}"),
        Patch(facecolor=_STATE_COLORS[2][:3] + (0.30,), edgecolor="none",
              label=f"FP (false alarm){_count(fp)}"),
        Patch(facecolor=_STATE_COLORS[3][:3] + (0.35,), edgecolor="none",
              label=f"FN (missed){_count(fn)}"),
    ]


def _add_state_background(ax, timestamps, tp, fp, fn, tn=None) -> None:
    """把 TN/TP/FP/FN 画成坐标区四色背景色带（不遮挡曲线）。"""
    ts = pd.to_datetime(timestamps)
    n = len(ts)
    if n == 0:
        return

    tp = np.asarray(tp, dtype=bool).reshape(-1)
    fp = np.asarray(fp, dtype=bool).reshape(-1)
    fn = np.asarray(fn, dtype=bool).reshape(-1)
    if tn is None:
        tn = ~(tp | fp | fn)
    else:
        tn = np.asarray(tn, dtype=bool).reshape(-1)

    # 0=TN 是默认状态，其余状态覆盖在它上面
    state = np.zeros(n, dtype=np.int8)
    state[tp] = 1
    state[fp] = 2
    state[fn] = 3

    x = date2num(ts.to_numpy())
    x0, x1 = float(x[0]), float(x[-1])
    if x1 <= x0:
        x1 = x0 + 1.0 / 86400.0
    pad = 0.5 / 86400.0
    y0, y1 = ax.get_ylim()

    ax.imshow(
        state[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        extent=[x0 - pad, x1 + pad, y0, y1],
        cmap=_STATE_CMAP,
        # 固定 0/1/2/3 的颜色映射；否则某类状态缺失时
        # matplotlib 会把 2(FP) 映射到 3(FN) 的灰色，导致误读。
        vmin=0, vmax=3,
        zorder=0,
    )
    # imshow 会改坐标范围，恢复成曲线原本的范围
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0, y1)


def _plot_true_overlay(ax, timestamps, true, pred, pred_label: str):
    """粗浅色 True 底线 + 细深色 Pred 虚线，重合时仍可分辨。"""
    line_true, = ax.plot(
        timestamps, true, label="True", color="#1f77b4",
        linewidth=2.8, alpha=0.35, zorder=2, solid_capstyle="round")
    line_pred, = ax.plot(
        timestamps, pred, label=pred_label, color="#ff7f0e",
        linewidth=1.2, linestyle="--", alpha=1.0, zorder=3)
    return line_true, line_pred


def plot_forecast_daily(timestamps, true, pred, residual, tp, fp, fn,
                        sensor_cols, out_dir, model_name: str,
                        score_name: str, threshold: float,
                        horizon: int = 1, tn=None) -> None:
    """每个通道每天一张图：True/Pred + TN/TP/FP/FN 背景 + 残差子图。"""
    tp = np.asarray(tp, dtype=bool).reshape(-1)
    fp = np.asarray(fp, dtype=bool).reshape(-1)
    fn = np.asarray(fn, dtype=bool).reshape(-1)
    tn = (~(tp | fp | fn) if tn is None
          else np.asarray(tn, dtype=bool).reshape(-1))

    df = pd.DataFrame(pred, columns=[f"pred_{c}" for c in sensor_cols])
    df.insert(0, "timestamp", pd.to_datetime(timestamps).values)
    for j, c in enumerate(sensor_cols):
        df[f"true_{c}"] = true[:, j]
        df[f"res_{c}"] = residual[:, j]
    df["tp"] = tp
    df["fp"] = fp
    df["fn"] = fn
    df["tn"] = tn
    df["date"] = df["timestamp"].dt.date

    os.makedirs(out_dir, exist_ok=True)
    for d in sorted(df["date"].unique()):
        day = df[df["date"] == d].reset_index(drop=True)
        if len(day) == 0:
            continue
        ds = str(d)
        tp_arr = day["tp"].to_numpy()
        fp_arr = day["fp"].to_numpy()
        fn_arr = day["fn"].to_numpy()
        tn_arr = day["tn"].to_numpy()

        for c in sensor_cols:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(14, 5), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]})

            line_true, line_pred = _plot_true_overlay(
                ax1, day["timestamp"], day[f"true_{c}"],
                day[f"pred_{c}"], "Pred")
            _add_state_background(
                ax1, day["timestamp"], tp_arr, fp_arr, fn_arr, tn_arr)

            title = (f"[{model_name}] {c} | {ds} | "
                     f"val-selected={score_name}")
            if horizon > 1:
                title += f" | H={horizon}"
            ax1.set_title(title)
            ax1.set_ylabel("Value")
            ax1.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
            ax1.legend(handles=[line_true, line_pred]
                       + _state_handles(tp_arr, fp_arr, tn_arr, fn_arr),
                       loc="upper right", fontsize=7, ncol=2)

            ax2.plot(day["timestamp"], day[f"res_{c}"], color="purple",
                     linewidth=0.8, alpha=0.8, label="Residual")
            _add_state_background(
                ax2, day["timestamp"], tp_arr, fp_arr, fn_arr, tn_arr)
            ax2.axhline(0, color="gray", linewidth=0.5)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("|Pred - True|")
            ax2.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
            ax2.legend(loc="upper right", fontsize=7)

            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{ds}_{_safe(c)}.png"), dpi=110)
            plt.close(fig)
        print(f"  {ds}: 已保存 {len(sensor_cols)} 张通道图")


def plot_lead_curves(best_rows, out_path: str, model_name: str = "") -> None:
    """逐 lead 检测指标曲线：AUC/PR-AUC 与 F1/FPR 随预测提前量的变化。"""
    if not best_rows:
        return
    leads = [int(r["lead"]) for r in best_rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(leads, [r["AUC"] for r in best_rows],
             marker="o", ms=2, label="AUC")
    ax1.plot(leads, [r["PR_AUC"] for r in best_rows],
             marker="s", ms=2, label="PR-AUC")
    ax1.set_xlabel("Lead (steps)")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax1.legend()

    ax2.plot(leads, [r["F1"] for r in best_rows],
             marker="o", ms=2, label="F1")
    ax2.plot(leads, [r["FPR"] for r in best_rows],
             marker="s", ms=2, label="FPR")
    ax2.set_xlabel("Lead (steps)")
    ax2.set_ylabel("Value")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax2.legend()

    if model_name:
        fig.suptitle(f"[{model_name}] detection metrics vs lead")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_reconstruction_daily(timestamps, true, recon, sensor_cols, out_dir,
                              model_name: str, score_name: str = "",
                              tp=None, fp=None, fn=None, tn=None) -> None:
    """每个通道每天一张图：True vs Recon + TN/TP/FP/FN 背景 + 重构残差。"""
    true = np.asarray(true)
    recon = np.asarray(recon)
    n = len(true)
    tp = np.zeros(n, dtype=bool) if tp is None else np.asarray(tp, dtype=bool)
    fp = np.zeros(n, dtype=bool) if fp is None else np.asarray(fp, dtype=bool)
    fn = np.zeros(n, dtype=bool) if fn is None else np.asarray(fn, dtype=bool)
    tn = (~(tp | fp | fn) if tn is None
          else np.asarray(tn, dtype=bool).reshape(-1))

    df = pd.DataFrame(recon, columns=[f"recon_{c}" for c in sensor_cols])
    df.insert(0, "timestamp", pd.to_datetime(timestamps).values)
    for j, c in enumerate(sensor_cols):
        df[f"true_{c}"] = true[:, j]
        df[f"res_{c}"] = np.abs(recon[:, j] - true[:, j])
    df["tp"] = tp
    df["fp"] = fp
    df["fn"] = fn
    df["tn"] = tn
    df["date"] = df["timestamp"].dt.date

    os.makedirs(out_dir, exist_ok=True)
    for d in sorted(df["date"].unique()):
        day = df[df["date"] == d].reset_index(drop=True)
        if len(day) == 0:
            continue
        ds = str(d)
        tp_arr = day["tp"].to_numpy()
        fp_arr = day["fp"].to_numpy()
        fn_arr = day["fn"].to_numpy()
        tn_arr = day["tn"].to_numpy()

        for c in sensor_cols:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(14, 5), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]})

            line_true, line_recon = _plot_true_overlay(
                ax1, day["timestamp"], day[f"true_{c}"],
                day[f"recon_{c}"], "Recon")
            _add_state_background(
                ax1, day["timestamp"], tp_arr, fp_arr, fn_arr, tn_arr)

            title = f"[{model_name}] {c} | {ds} | True vs Recon"
            if score_name:
                title += f" | val-selected={score_name}"
            ax1.set_title(title)
            ax1.set_ylabel("Value")
            ax1.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
            ax1.legend(handles=[line_true, line_recon]
                       + _state_handles(tp_arr, fp_arr, tn_arr, fn_arr),
                       loc="upper right", fontsize=7, ncol=2)

            ax2.plot(day["timestamp"], day[f"res_{c}"], color="purple",
                     linewidth=0.8, alpha=0.8, label="Reconstruction residual")
            _add_state_background(
                ax2, day["timestamp"], tp_arr, fp_arr, fn_arr, tn_arr)
            ax2.axhline(0, color="gray", linewidth=0.5)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("|Recon - True|")
            ax2.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
            ax2.legend(loc="upper right", fontsize=7)

            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{ds}_{_safe(c)}_recon.png"),
                        dpi=110)
            plt.close(fig)
        print(f"  {ds}: 已保存 {len(sensor_cols)} 张重构对比图")


def plot_score_daily(timestamps, score, labels, tp, fp, fn,
                     threshold, out_dir, model_name: str,
                     score_name: str, extra_scores: dict | None = None,
                     tn=None) -> None:
    """重构/对比类：每天一张异常分数图，TN/TP/FP/FN 用四色背景带标注。

    ``threshold`` 可以是标量，也可以是与 ``score`` 等长的逐时刻阈值数组；
    后者会画成虚线阈值曲线（用于 rolling_fpr）。
    """
    tp = np.asarray(tp, dtype=bool).reshape(-1)
    fp = np.asarray(fp, dtype=bool).reshape(-1)
    fn = np.asarray(fn, dtype=bool).reshape(-1)
    tn = (~(tp | fp | fn) if tn is None
          else np.asarray(tn, dtype=bool).reshape(-1))

    thr_arr = np.asarray(threshold, dtype=np.float64)
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps).values,
        "label": labels,
        "score": score,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    })
    if thr_arr.ndim == 0:
        df["threshold"] = float(thr_arr)
    else:
        if len(thr_arr) != len(df):
            raise ValueError("threshold 数组长度与 score 不一致")
        df["threshold"] = thr_arr
    for name, values in (extra_scores or {}).items():
        df[name] = values
    df["date"] = df["timestamp"].dt.date

    os.makedirs(out_dir, exist_ok=True)
    for d in sorted(df["date"].unique()):
        day = df[df["date"] == d].reset_index(drop=True)
        if len(day) == 0:
            continue
        ds = str(d)

        fig, ax = plt.subplots(figsize=(14, 4))
        line_score, = ax.plot(day["timestamp"], day["score"],
                              label=score_name, color="purple",
                              linewidth=1.2, zorder=3)
        extra_lines = []
        for name in (extra_scores or {}):
            line, = ax.plot(day["timestamp"], day[name], label=name,
                            linewidth=1.0, alpha=0.75, zorder=2)
            extra_lines.append(line)

        _add_state_background(
            ax, day["timestamp"],
            day["tp"].to_numpy(), day["fp"].to_numpy(),
            day["fn"].to_numpy(), day["tn"].to_numpy())

        if thr_arr.ndim == 0:
            thr_line = ax.axhline(
                float(thr_arr), color="gray", linestyle="--",
                linewidth=0.8, label=f"thr={float(thr_arr):.4f}",
                zorder=2)
        else:
            thr_line, = ax.plot(
                day["timestamp"], day["threshold"].to_numpy(),
                color="gray", linestyle="--", linewidth=0.9,
                label=f"rolling thr (med={np.nanmedian(thr_arr):.4f})",
                zorder=2)

        ax.set_title(f"[{model_name}] {score_name} | {d} | "
                     f"val-selected={score_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel(score_name)
        ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
        ax.legend(handles=[line_score] + extra_lines + [thr_line]
                  + _state_handles(
                      day["tp"].to_numpy(), day["fp"].to_numpy(),
                      day["tn"].to_numpy(), day["fn"].to_numpy()),
                  loc="upper right", fontsize=8, ncol=3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{d}_anomaly_score.png"), dpi=110)
        plt.close(fig)
        print(f"  {d}: 已保存异常分数图")
