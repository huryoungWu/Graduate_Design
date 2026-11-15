# -*- coding: utf-8 -*-
"""把本地 SWaT / WADI CSV 转成 Anomaly-Transformer 复现所需的 .npy。

划分与官方代码一致：
  train = period == "normal" 的全部时间点（只用 Normal 训练）
  test  = period == "attack" 的全部时间点（带 label）

标准化：StandardScaler 只在 Normal 训练段上 fit，再 transform 两段。
本地 CSV 已经删掉了恒定列；为了让输入维数与官方 SWaT(51)/WADI(127)
一致，这里把被删掉的恒定列补成 0（常量列标准化后本来也是 0）。

用法::

    python -m repro_anomaly_transformer.prepare_data --dataset swat
    python -m repro_anomaly_transformer.prepare_data --dataset wadi
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

DEFAULT = {
    "swat": {
        "csv": ROOT / "08_SWaT" / "SWaT_all_1s.csv",
        "dropped": ROOT / "08_SWaT" / "SWaT_dropped_constant_columns.txt",
        "out": ROOT / "repro_anomaly_transformer" / "data" / "swat",
    },
    "wadi": {
        "csv": ROOT / "09_WADI" / "WADI_all_1s.csv",
        "dropped": ROOT / "09_WADI" / "WADI_dropped_constant_columns.txt",
        "out": ROOT / "repro_anomaly_transformer" / "data" / "wadi",
    },
}


def _load_dropped(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="准备 SWaT/WADI 复现数据")
    p.add_argument("--dataset", choices=["swat", "wadi"], required=True)
    p.add_argument("--csv", default="")
    p.add_argument("--dropped", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--winsorize", type=float, default=0.0,
                   help="按 Normal 训练段的 0.1%%/99.9%% 分位裁剪；"
                        "WADI 建议 0.001，SWaT 可保持 0")
    args = p.parse_args()

    cfg = DEFAULT[args.dataset]
    csv_path = Path(args.csv) if args.csv else cfg["csv"]
    dropped_path = Path(args.dropped) if args.dropped else cfg["dropped"]
    out_dir = Path(args.out_dir) if args.out_dir else cfg["out"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] 读取 {csv_path} ...", flush=True)
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "period" not in df.columns:
        raise ValueError("CSV 缺少 period 列，无法区分 Normal/Attack")

    feature_cols = [c for c in df.columns
                    if c not in ("timestamp", "label", "period")]
    dropped_cols = _load_dropped(dropped_path)

    period = df["period"].astype(str).str.lower()
    normal_mask = period == "normal"
    attack_mask = period == "attack"
    if not normal_mask.any() or not attack_mask.any():
        raise ValueError("period 中没有 normal/attack 段")

    train = df.loc[normal_mask]
    test = df.loc[attack_mask]
    print(f"[prepare] Normal {len(train):,} | Attack {len(test):,} | "
          f"特征 {len(feature_cols)} | 补充恒定列 {len(dropped_cols)}")

    train_x = np.nan_to_num(
        train[feature_cols].to_numpy(np.float32),
        nan=0.0, posinf=0.0, neginf=0.0)
    test_x = np.nan_to_num(
        test[feature_cols].to_numpy(np.float32),
        nan=0.0, posinf=0.0, neginf=0.0)
    test_y = test["label"].to_numpy(np.int8)
    test_ts = test["timestamp"].to_numpy("datetime64[s]")
    train_ts = train["timestamp"].to_numpy("datetime64[s]")

    # 只使用 Normal 训练段的统计量做 winsorization：
    # WADI 的 2B_AIT_002_PV 在 Normal 标准差约 0.16，Attack 值却到 8128，
    # 标准化后 ~49k，会让验证损失直接爆炸。裁剪不改变 Normal 主体分布，
    # 但把 Attack 极端值压到 Normal 的 99.9% 分位。
    clip_low = clip_high = None
    if args.winsorize and args.winsorize > 0:
        q = float(args.winsorize)
        clip_low = np.quantile(train_x, q, axis=0).astype(np.float32)
        clip_high = np.quantile(train_x, 1.0 - q, axis=0).astype(np.float32)
        train_x = np.clip(train_x, clip_low, clip_high)
        test_x = np.clip(test_x, clip_low, clip_high)
        print(f"[prepare] winsorize {q:.4f} -> "
              f"scaled max {float(np.abs(test_x).max()):.3f} (raw)")

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x).astype(np.float32)
    test_x = scaler.transform(test_x).astype(np.float32)

    # 补回被删除的恒定列（标准化后为 0）
    if dropped_cols:
        z_train = np.zeros((len(train_x), len(dropped_cols)), np.float32)
        z_test = np.zeros((len(test_x), len(dropped_cols)), np.float32)
        train_x = np.concatenate([train_x, z_train], axis=1)
        test_x = np.concatenate([test_x, z_test], axis=1)

    np.save(out_dir / "train_X.npy", train_x)
    np.save(out_dir / "test_X.npy", test_x)
    np.save(out_dir / "test_y.npy", test_y)
    np.save(out_dir / "train_timestamp.npy", train_ts)
    np.save(out_dir / "test_timestamp.npy", test_ts)

    meta = {
        "dataset": args.dataset,
        "csv": str(csv_path),
        "feature_cols": feature_cols,
        "dropped_constant_cols": dropped_cols,
        "channels": feature_cols + dropped_cols,
        "n_features": int(train_x.shape[1]),
        "n_train": int(len(train_x)),
        "n_test": int(len(test_x)),
        "n_attack_points": int(test_y.sum()),
        "attack_ratio": float(test_y.mean()),
        "train_period": [str(train_ts[0]), str(train_ts[-1])],
        "test_period": [str(test_ts[0]), str(test_ts[-1])],
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "winsorize": float(args.winsorize),
        "clip_low": None if clip_low is None else clip_low.astype(float).tolist(),
        "clip_high": None if clip_high is None else clip_high.astype(float).tolist(),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[prepare] 已保存: {out_dir}")
    print(f"[prepare] train_X {train_x.shape} | test_X {test_x.shape} | "
          f"attack_ratio {test_y.mean():.4f}")


if __name__ == "__main__":
    main()
