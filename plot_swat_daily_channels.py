# -*- coding: utf-8 -*-
"""按天绘制 SWaT 原始数据每个通道的曲线，并用背景色标出正常/异常。

输出结构：
    SWaT_Daily_Channel_Plots/
        2015-12-22/
            FIT101.png
            LIT101.png
            ...
            day_summary.csv
        2015-12-23/
        ...
        2016-01-02/

背景色：
    - 浅绿色：label=0（正常点）
    - 浅红色：label=1（真实异常点）

其他：
    - 连续通道（FIT/LIT/AIT/PIT/DPIT）按 5 秒降采样绘制；
    - 离散通道（泵/阀/UV 等）保持 1 秒 step 绘制；
    - 标题给出当天异常秒数和占比。

运行：
    python -I plot_swat_daily_channels.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from matplotlib.dates import date2num
from matplotlib.patches import Patch

# ============================================================
# 配置
# ============================================================
DATA_ROOT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"
CSV_PATH = os.path.join(DATA_ROOT, "08_SWaT", "SWaT_all_1s.csv")
OUTPUT_ROOT = os.path.join(DATA_ROOT, "SWaT_Daily_Channel_Plots")

TIME_COLUMN = "timestamp"
LABEL_COLUMN = "label"
PERIOD_COLUMN = "period"

# 连续通道绘图降采样步长（秒）；离散通道固定为 1 秒
CONTINUOUS_STRIDE = 5

# 图片参数
DPI = 120
FIGSIZE = (13, 4.8)

# 0=正常，1=异常；颜色带透明度，避免遮挡曲线
LABEL_CMAP = ListedColormap([
    (0.0, 0.55, 0.0, 0.10),   # normal：浅绿
    (0.85, 0.0, 0.0, 0.20),   # anomaly：浅红
])


# ============================================================
# 工具函数
# ============================================================
def runs_of_true(mask):
    """返回 True 连续区间的 (start_idx, end_idx) 列表。"""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return [(int(r[0]), int(r[-1])) for r in np.split(idx, splits)]


def is_discrete_channel(series):
    """泵/阀/UV 等离散状态量：取值很少且为整数。"""
    values = pd.unique(series.dropna())
    if len(values) > 10:
        return False
    return np.allclose(values, np.round(values))


def format_period_title(periods_day):
    counts = pd.Series(periods_day).value_counts().to_dict()
    parts = []
    for key in ["normal", "attack"]:
        if key in counts:
            parts.append(f"{key}: {counts[key]}s")
    return " | ".join(parts) if parts else "period: unknown"


def add_label_background(ax, timestamps, labels):
    """用浅绿/浅红背景标出正常点和真实异常点。

    直接画一个 1×N 的状态图，比逐段 axvspan 更稳，也不会
    因为某一天没有异常而出现颜色映射错位。
    """
    ts = pd.to_datetime(timestamps)
    x = date2num(ts.to_numpy())
    x0, x1 = float(x[0]), float(x[-1])
    if x1 <= x0:
        x1 = x0 + 1.0 / 86400.0
    pad = 0.5 / 86400.0
    y0, y1 = ax.get_ylim()
    state = np.asarray(labels, dtype=np.int8)
    ax.imshow(
        state[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        extent=[x0 - pad, x1 + pad, y0, y1],
        cmap=LABEL_CMAP,
        vmin=0, vmax=1,
        zorder=0,
    )
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0, y1)


# ============================================================
# 主流程
# ============================================================
def main():
    print("读取数据：", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")

    if df[TIME_COLUMN].isna().any():
        raise ValueError("时间列存在无法解析的值。")

    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)

    feature_cols = [
        c for c in df.columns
        if c not in (TIME_COLUMN, LABEL_COLUMN, PERIOD_COLUMN)
    ]

    labels = pd.to_numeric(df[LABEL_COLUMN], errors="coerce").to_numpy()
    periods = df[PERIOD_COLUMN].astype(str).str.strip().str.lower().to_numpy()
    dates = df[TIME_COLUMN].dt.date.to_numpy()

    if np.isnan(labels).any():
        raise ValueError("label 列存在非数值。")

    labels = labels.astype(np.int64)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    readme = """SWaT 每日通道曲线图
====================

目录结构：
    日期/
        通道名.png         每个通道一张图
        day_summary.csv    当天行数、label/period 统计

背景色：
    浅绿色 = label=0（正常点）
    浅红色 = label=1（真实异常点）

标题：
    SWaT <日期> | <通道> | anomaly=<异常秒数>s (<占比>%) | ...
    - 连续通道按 5 秒降采样绘制
    - 离散通道（泵/阀/UV）保持 1 秒 step 绘制

说明：
    背景色来自原始数据集的 label 列，表示真实标签；
    period 列只表示数据录制批次，已放在标题中。
"""
    with open(os.path.join(OUTPUT_ROOT, "README.txt"),
              "w", encoding="utf-8") as f:
        f.write(readme)

    unique_dates = sorted(pd.unique(dates))
    print(f"共 {len(unique_dates)} 个自然日，每天 {len(feature_cols)} 个通道")

    for day_idx, date in enumerate(unique_dates, start=1):
        day = df.loc[dates == date].reset_index(drop=True)
        day_periods = day[PERIOD_COLUMN].astype(str).str.strip().str.lower()
        day_labels = day[LABEL_COLUMN].to_numpy(dtype=np.int64)
        day_times = day[TIME_COLUMN].to_numpy()

        out_dir = Path(OUTPUT_ROOT) / str(date)
        out_dir.mkdir(parents=True, exist_ok=True)

        label_runs = runs_of_true(day_labels == 1)
        period_title = format_period_title(day_periods.to_numpy())
        n_label1 = int((day_labels == 1).sum())
        label1_pct = 100.0 * n_label1 / len(day) if len(day) else 0.0

        summary = {
            "date": str(date),
            "rows": len(day),
            "first_time": str(day[TIME_COLUMN].iloc[0]),
            "last_time": str(day[TIME_COLUMN].iloc[-1]),
            "label0_seconds": int((day_labels == 0).sum()),
            "label1_seconds": n_label1,
            "label1_pct": label1_pct,
            "period_normal_seconds": int((day_periods == "normal").sum()),
            "period_attack_seconds": int((day_periods == "attack").sum()),
            "attack_segments": len(label_runs),
        }
        pd.DataFrame([summary]).to_csv(
            out_dir / "day_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"[{day_idx:02d}/{len(unique_dates)}] {date} | "
            f"{len(day)}s | label1={n_label1}s ({label1_pct:.1f}%) | "
            f"{len(label_runs)} 段异常"
        )

        for ch_idx, ch in enumerate(feature_cols, start=1):
            series = day[ch]
            discrete = is_discrete_channel(series)
            stride = 1 if discrete else CONTINUOUS_STRIDE

            t = day_times[::stride]
            y = series.to_numpy(dtype=np.float64)[::stride]

            fig, ax = plt.subplots(figsize=FIGSIZE)
            ax.plot(
                t, y,
                linewidth=0.8 if not discrete else 1.0,
                color="tab:blue" if not discrete else "tab:orange",
                drawstyle="steps-post" if discrete else "default",
                zorder=2,
            )

            # 先让 y 轴范围确定，再铺背景
            ax.set_xlim(day_times[0], day_times[-1])
            add_label_background(ax, day_times, day_labels)

            ax.set_ylabel(ch, fontsize=9)
            ax.grid(alpha=0.22)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.tick_params(axis="x", labelsize=8)
            ax.set_xlabel("time of day", fontsize=9)

            handles = [
                Patch(facecolor=(0.0, 0.55, 0.0, 0.30),
                      edgecolor="none", label="normal (label=0)"),
                Patch(facecolor=(0.85, 0.0, 0.0, 0.35),
                      edgecolor="none", label="anomaly (label=1)"),
            ]
            ax.legend(handles=handles, loc="upper right",
                      fontsize=7, framealpha=0.9)

            title = (
                f"SWaT {date} | {ch} | "
                f"anomaly={n_label1}s ({label1_pct:.1f}%) | "
                f"{period_title}"
            )
            if discrete:
                title += " | discrete (1s step)"
            else:
                title += f" | {CONTINUOUS_STRIDE}s resolution"
            ax.set_title(title, fontsize=10, pad=6)

            fig.tight_layout()
            fig.savefig(out_dir / f"{ch}.png",
                        dpi=DPI, bbox_inches="tight")
            plt.close(fig)

            if ch_idx % 10 == 0 or ch_idx == len(feature_cols):
                print(f"    {ch_idx}/{len(feature_cols)} channels")

    print("\n完成。输出目录：", OUTPUT_ROOT)
    print(f"每个日期文件夹内包含 {len(feature_cols)} 张通道图和 "
          "day_summary.csv")


if __name__ == "__main__":
    main()
