# -*- coding: utf-8 -*-
"""预测类模型的两阶段管线。

Stage 1（train）:
    训练 -> 在 val/test 上推理 -> 保存 artifact（pred/true/label/timestamp）
    -> 保存逐通道预测精度 prediction_metrics.csv。

Stage 2（detect）:
    读取 artifact -> 计算 |pred-true| -> 通道聚合 -> 选阈值 -> 测试集评估。
    该阶段不导入模型，也不重新读取原始 CSV。

用法::

    python -m models.forecasting.lstm --stage train --epochs 50
    python -m models.detect --artifact-dir runs/lstm_swat --scores mean
    # 或者一条命令完成两阶段：
    python -m models.forecasting.lstm --stage all --epochs 50
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .artifacts import ensure_dir, make_meta, save_forecast_split, save_meta
from .config import RunConfig, parse_config, resolve_device, set_seed
from .data import load_swat, make_forecast_loaders
from .detection import DetectionConfig, run_detection_stage
from .eval import channel_metrics
from .inference import collect_forecast_outputs
from .train import train_forecaster


def resolve_artifact_dir(cfg: RunConfig, prefix: str) -> Path:
    if cfg.artifact_dir:
        return Path(cfg.artifact_dir)
    return Path(cfg.out_dir) / "runs" / prefix


def _first_step(array: np.ndarray) -> np.ndarray:
    return array[:, 0, :] if array.ndim == 3 else array


def _first_label(array: np.ndarray) -> np.ndarray:
    return array[:, 0] if array.ndim == 2 else array


def _prediction_rows(bundle, pred: np.ndarray, true: np.ndarray) -> tuple[list, list]:
    rows, mapes = [], []
    for j, col in enumerate(bundle.sensor_cols):
        mae_j, rmse_j, mape_j, vr_j = channel_metrics(true[:, j], pred[:, j])
        rows.append({"channel": col, "MAE": mae_j, "RMSE": rmse_j,
                     "MAPE(%)": mape_j, "valid_ratio": vr_j})
        if np.isfinite(mape_j):
            mapes.append(mape_j)
    return rows, mapes


def _split_timestamps(df, n: int, cfg: RunConfig) -> np.ndarray:
    start = cfg.seq_len + cfg.horizon - 1
    return df["timestamp"].iloc[start:start + n].astype(str).to_numpy()


def run_forecast_stage(model_name: str, prefix: str,
                       build_model: Callable, cfg: RunConfig) -> Path:
    """Stage 1：训练/推理/保存 artifact + 预测精度。"""
    set_seed(cfg.seed, cfg.deterministic)
    device = resolve_device(cfg.device)
    print(f"设备: {device} | 模型: {model_name} | epochs: {cfg.epochs} | "
          f"batch: {cfg.batch_size} | seq_len: {cfg.seq_len} | "
          f"horizon: {cfg.horizon} | pred_len: {cfg.pred_len}")

    bundle = load_swat(cfg.csv, val_frac=cfg.val_frac)
    train_loader, val_loader, test_loader = make_forecast_loaders(bundle, cfg)

    model = build_model(bundle.n_channels, cfg.seq_len, cfg.pred_len,
                        bundle).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    if cfg.load_model:
        state = torch.load(cfg.load_model, map_location=device)
        model.load_state_dict(state)
        train_time = 0.0
        print(f"[{model_name}] 已加载权重: {cfg.load_model}（跳过训练）")
    else:
        train_time = train_forecaster(
            model, train_loader, cfg.epochs, cfg.lr, device, model_name)
        print(f"[{model_name}] 训练总耗时: {train_time:.1f}s")

    first_only = not cfg.save_all_steps
    val_pred, val_true, val_label = collect_forecast_outputs(
        model, val_loader, device, first_only=first_only)
    test_pred, test_true, test_label = collect_forecast_outputs(
        model, test_loader, device, first_only=first_only)

    val_pred_f, val_true_f = _first_step(val_pred), _first_step(val_true)
    test_pred_f, test_true_f = _first_step(test_pred), _first_step(test_true)
    val_label_f = _first_label(val_label)
    test_label_f = _first_label(test_label)

    val_ts = _split_timestamps(bundle.val_df, len(val_pred), cfg)
    test_ts = _split_timestamps(bundle.test_df, len(test_pred), cfg)

    # ---- 预测精度（标准化空间） ----
    artifact_dir = ensure_dir(resolve_artifact_dir(cfg, prefix))
    val_rows, val_mapes = _prediction_rows(bundle, val_pred_f, val_true_f)
    test_rows, test_mapes = _prediction_rows(bundle, test_pred_f, test_true_f)
    pd.DataFrame(val_rows).to_csv(
        artifact_dir / "prediction_metrics_val.csv",
        index=False, encoding="utf-8-sig")
    pd.DataFrame(test_rows).to_csv(
        artifact_dir / "prediction_metrics.csv",
        index=False, encoding="utf-8-sig")

    val_mae = float(np.mean(np.abs(val_pred_f - val_true_f)))
    val_rmse = float(np.sqrt(np.mean((val_pred_f - val_true_f) ** 2)))
    test_mae = float(np.mean(np.abs(test_pred_f - test_true_f)))
    test_rmse = float(np.sqrt(np.mean((test_pred_f - test_true_f) ** 2)))

    print(f"\n[{model_name}] 预测精度（测试集，标准化空间）: "
          f"MAE {test_mae:.6f} | RMSE {test_rmse:.6f}")

    # ---- 保存 artifact ----
    meta = make_meta(
        model=model_name, prefix=prefix, task="forecast",
        channels=bundle.sensor_cols, seq_len=cfg.seq_len,
        pred_len=cfg.pred_len, horizon=cfg.horizon,
        val_frac=cfg.val_frac,
        split={
            "split_idx": int(bundle.split_idx),
            "val_end": int(bundle.val_end),
            "train_rows": len(bundle.train_df),
            "val_rows": len(bundle.val_df),
            "test_rows": len(bundle.test_df),
        },
        train={
            "epochs": int(cfg.epochs), "batch_size": int(cfg.batch_size),
            "lr": float(cfg.lr), "seed": int(cfg.seed),
            "deterministic": bool(cfg.deterministic),
            "train_time_s": float(train_time),
            "params": int(n_params),
        },
        scaler_mean=bundle.scaler.mean_, scaler_scale=bundle.scaler.scale_,
        extra={
            "prediction": {
                "val": {"MAE": val_mae, "RMSE": val_rmse,
                        "MAPE_mean_channels": float(np.mean(val_mapes))
                        if val_mapes else None},
                "test": {"MAE": test_mae, "RMSE": test_rmse,
                         "MAPE_mean_channels": float(np.mean(test_mapes))
                         if test_mapes else None},
            },
            "prediction_metrics": "prediction_metrics.csv",
        },
    )
    save_meta(artifact_dir, meta)

    save_forecast_split(
        artifact_dir, "val", pred=val_pred, true=val_true,
        label=val_label, timestamp=val_ts)
    save_forecast_split(
        artifact_dir, "test", pred=test_pred, true=test_true,
        label=test_label, timestamp=test_ts)

    torch.save(model.state_dict(), artifact_dir / "model.pt")

    print(f"[{model_name}] artifact 已保存: {artifact_dir}")
    return artifact_dir


def run_forecasting(model_name: str, prefix: str,
                    build_model: Callable, description: str = "",
                    batch_size: int = 256, epochs: int = 10,
                    lr: float = 1e-4,
                    extra_fields: dict | None = None) -> dict:
    """兼容旧模型入口的两阶段包装。"""
    cfg: RunConfig = parse_config(
        description or f"{model_name} 预测类模型",
        batch_size=batch_size, epochs=epochs, lr=lr)
    artifact_dir = resolve_artifact_dir(cfg, prefix)

    summary = {
        "model": model_name, "prefix": prefix,
        "artifact_dir": str(artifact_dir), "stage": cfg.stage,
    }
    if extra_fields:
        summary["extra"] = extra_fields

    if cfg.stage in ("train", "all"):
        artifact_dir = run_forecast_stage(
            model_name, prefix, build_model, cfg)
        summary["artifact_dir"] = str(artifact_dir)

    if cfg.stage in ("detect", "all"):
        detection = run_detection_stage(
            artifact_dir, DetectionConfig.from_run_config(cfg))
        summary["detection"] = detection

    return summary
