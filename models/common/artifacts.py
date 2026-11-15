# -*- coding: utf-8 -*-
"""训练/推理产物的统一保存与加载。

设计原则
--------
Stage 1（训练 + 推理）只负责把模型输出和预测/重构精度落盘；
Stage 2（异常检测）只读取这里的产物，不再依赖模型类、DataLoader 或原始 CSV。

一个 artifact 目录的结构::

    runs/<prefix>/
      meta.json          # 模型、切分、通道、超参数、scaler 统计量
      model.pt           # 可选
      val.npz            # 验证集输出
      test.npz           # 测试集输出
      prediction_metrics.csv   # 预测类
      recon_metrics.csv        # 重构类

npz 中数组的命名约定：

预测类
  pred      [N, C]  或 [N, P, C]   预测值（默认第一步）
  true      [N, C]  或 [N, P, C]   真值
  label     [N]     或 [N, P]      标签
  timestamp [N]     预测时刻
  score_<name> [N]  模型特有的分数（可选）

重构类
  recon_last [N, C]  最后一个时刻的重构值
  true_last  [N, C]  最后一个时刻的真值
  abs_last   [N, C]  最后一个时刻的绝对误差
  sq_last    [N, C]  最后一个时刻的平方误差
  abs_ts     [N, L]  逐时刻的通道平均绝对误差
  sq_ts      [N, L]  逐时刻的通道平均平方误差
  label_window [N]   窗口内是否含攻击
  label_point  [N]   窗口最后一点的标签
  timestamp    [N]   窗口最后时刻
  score_<name> [N]   模型特有分数（window/point/assoc/fused 等）

对比类（DCdetector/SBT）
  score_<name> [N]   窗口级分数
  label_window [N]
  timestamp    [N]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACT_VERSION = 1


# ---------------------------------------------------------------------------
# 基础读写
# ---------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_meta(artifact_dir: str | Path, meta: dict[str, Any]) -> None:
    p = ensure_dir(artifact_dir) / "meta.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_meta(artifact_dir: str | Path) -> dict[str, Any]:
    with open(Path(artifact_dir) / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _save_npz(path: str | Path, arrays: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    payload = {}
    for key, value in arrays.items():
        if value is None:
            continue
        arr = _to_numpy(value)
        if arr.dtype.kind in "OUS":
            arr = arr.astype(str)
        payload[key] = arr
    np.savez_compressed(path, **payload)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# 预测类
# ---------------------------------------------------------------------------
def save_forecast_split(artifact_dir: str | Path, split: str, *,
                        pred, true, label, timestamp,
                        scores: dict[str, Any] | None = None) -> None:
    arrays = {
        "pred": np.asarray(pred, dtype=np.float32),
        "true": np.asarray(true, dtype=np.float32),
        "label": np.asarray(label, dtype=np.int8),
        # datetime64[s] 比字符串省 ~10 倍空间，且 pandas 可直接识别
        "timestamp": np.asarray(timestamp, dtype="datetime64[s]"),
    }
    for name, value in (scores or {}).items():
        arrays[f"score_{name}"] = value
    _save_npz(Path(artifact_dir) / f"{split}.npz", arrays)


def load_forecast_split(artifact_dir: str | Path,
                        split: str) -> dict[str, Any]:
    z = _load_npz(Path(artifact_dir) / f"{split}.npz")
    return {
        "pred": z["pred"],
        "true": z["true"],
        "label": z["label"],
        "timestamp": z["timestamp"],
        "scores": {k[len("score_"):]: z[k] for k in z
                   if k.startswith("score_")},
    }


# ---------------------------------------------------------------------------
# 重构 / 对比类
# ---------------------------------------------------------------------------
_RECON_SCALAR_KEYS = (
    "recon_last", "true_last", "abs_last", "sq_last",
    "abs_ts", "sq_ts", "label_window", "label_point", "timestamp",
)


def save_recon_split(artifact_dir: str | Path, split: str, *,
                     recon_last=None, true_last=None,
                     abs_last=None, sq_last=None,
                     abs_ts=None, sq_ts=None,
                     label_window=None, label_point=None,
                     timestamp,
                     window_scores: dict[str, Any] | None = None,
                     point_scores: dict[str, Any] | None = None,
                     ts_scores: dict[str, Any] | None = None,
                     channel_arrays: dict[str, Any] | None = None) -> None:
    arrays = {
        "recon_last": recon_last,
        "true_last": true_last,
        "abs_last": abs_last,
        "sq_last": sq_last,
        "abs_ts": abs_ts,
        "sq_ts": sq_ts,
        "label_window": (None if label_window is None
                         else np.asarray(label_window, dtype=np.int8)),
        "label_point": (None if label_point is None
                        else np.asarray(label_point, dtype=np.int8)),
        "timestamp": np.asarray(timestamp, dtype="datetime64[s]"),
    }
    for name, value in (window_scores or {}).items():
        arrays[f"win_{name}"] = value
    for name, value in (point_scores or {}).items():
        arrays[f"pt_{name}"] = value
    for name, value in (ts_scores or {}).items():
        arrays[f"ts_{name}"] = value
    for name, value in (channel_arrays or {}).items():
        arrays[f"ch_{name}"] = value
    _save_npz(Path(artifact_dir) / f"{split}.npz", arrays)


def load_recon_split(artifact_dir: str | Path,
                     split: str) -> dict[str, Any]:
    z = _load_npz(Path(artifact_dir) / f"{split}.npz")
    out: dict[str, Any] = {
        "window_scores": {k[len("win_"):]: z[k] for k in z
                          if k.startswith("win_")},
        "point_scores": {k[len("pt_"):]: z[k] for k in z
                         if k.startswith("pt_")},
        "ts_scores": {k[len("ts_"):]: z[k] for k in z
                      if k.startswith("ts_")},
        "channel": {k[len("ch_"):]: z[k] for k in z
                    if k.startswith("ch_")},
    }
    for key in _RECON_SCALAR_KEYS:
        out[key] = z.get(key)
    return out


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def load_artifact(artifact_dir: str | Path) -> dict[str, Any]:
    """加载整个 artifact 目录，返回 meta + val + test。"""
    meta = load_meta(artifact_dir)
    task = meta.get("task", "forecast")
    if task == "forecast":
        return {
            "meta": meta,
            "val": load_forecast_split(artifact_dir, "val"),
            "test": load_forecast_split(artifact_dir, "test"),
        }
    return {
        "meta": meta,
        "val": load_recon_split(artifact_dir, "val"),
        "test": load_recon_split(artifact_dir, "test"),
    }


def make_meta(*, model: str, prefix: str, task: str, channels: list[str],
              seq_len: int, pred_len: int, horizon: int, val_frac: float,
              split: dict[str, Any], train: dict[str, Any],
              scaler_mean=None, scaler_scale=None,
              scaler_type: str = "standard",
              extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = {
        "artifact_version": ARTIFACT_VERSION,
        "model": model,
        "prefix": prefix,
        "task": task,
        "channels": list(channels),
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "horizon": int(horizon),
        "val_frac": float(val_frac),
        "split": split,
        "train": train,
        "scaler": {
            # standard: X' = (X - mean) / scale
            # minmax:   X' = X * scale + mean（mean 字段存 min_）
            "type": str(scaler_type),
            "mean": None if scaler_mean is None
            else np.asarray(scaler_mean, dtype=np.float64).tolist(),
            "scale": None if scaler_scale is None
            else np.asarray(scaler_scale, dtype=np.float64).tolist(),
        },
    }
    if extra:
        meta.update(extra)
    return meta
