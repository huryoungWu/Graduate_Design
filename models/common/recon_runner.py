# -*- coding: utf-8 -*-
"""重构/对比类模型的两阶段管线。

Stage 1（train）:
    训练 -> 收集窗口/点/通道分数与重构误差 -> 保存 artifact
    -> 保存逐通道重构精度 recon_metrics.csv（AE/VAE/AnomalyTransformer）。

Stage 2（detect）:
    读取 artifact -> 构造窗口级/点级候选分数 -> 选阈值 -> 测试集评估。
    该阶段不导入模型，也不重新读取原始 CSV。

用法::

    python -m models.reconstruction.ae --stage train --epochs 50
    python -m models.detect --artifact-dir runs/ae_swat --scores recon_mean
    # 或者：
    python -m models.reconstruction.ae --stage all --epochs 50
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .artifacts import ensure_dir, make_meta, save_meta, save_recon_split
from .config import RunConfig, parse_config, resolve_device, set_seed
from .data import load_swat, make_reconstruction_loaders, scaler_stats
from .detection import DetectionConfig, run_detection_stage
from .eval import channel_metrics
from .inference import collect_recon_outputs, fuse_val_test
from .train import train_reconstructor


def resolve_artifact_dir(cfg: RunConfig, prefix: str) -> Path:
    if cfg.artifact_dir:
        return Path(cfg.artifact_dir)
    return Path(cfg.out_dir) / "runs" / prefix


def _split_timestamps(df, n: int, cfg: RunConfig) -> np.ndarray:
    start = cfg.seq_len - 1
    return df["timestamp"].iloc[start:start + n].astype(str).to_numpy()


def _split_labels(labels: np.ndarray, n: int, cfg: RunConfig) -> np.ndarray:
    start = cfg.seq_len - 1
    return np.asarray(labels[start:start + n]).astype(np.int8)


def _recon_rows(bundle, abs_last: np.ndarray, sq_last: np.ndarray) -> list[dict]:
    rows = []
    for j, col in enumerate(bundle.sensor_cols):
        a = abs_last[:, j]
        s = sq_last[:, j]
        # 与旧 runner 一致：MAPE 用 |true| 的有效位置，这里 true 来自标准化后的最后一点
        # 由调用方传入 true_last 时再用；没有 true_last 时仅报告 MAE/RMSE。
        rows.append({"channel": col, "MAE": float(a.mean()),
                     "RMSE": float(np.sqrt(s.mean())),
                     "MAPE(%)": float("nan"),
                     "valid_ratio": 1.0})
    return rows


def _recon_rows_with_true(bundle, abs_last: np.ndarray, sq_last: np.ndarray,
                          true_last: np.ndarray) -> list[dict]:
    rows = []
    for j, col in enumerate(bundle.sensor_cols):
        t = true_last[:, j]
        a = abs_last[:, j]
        s = sq_last[:, j]
        mask = np.abs(t) > 1e-6
        mape = (float(np.mean(a[mask] / np.abs(t[mask])) * 100)
                if mask.any() else float("nan"))
        rows.append({"channel": col, "MAE": float(a.mean()),
                     "RMSE": float(np.sqrt(s.mean())),
                     "MAPE(%)": mape,
                     "valid_ratio": float(mask.mean())})
    return rows


def run_recon_stage(model_name: str, prefix: str,
                    build_model: Callable, cfg: RunConfig,
                    save_model: bool = False) -> Path:
    """Stage 1：训练/推理/保存 artifact + 重构精度。"""
    set_seed(cfg.seed, cfg.deterministic)
    device = resolve_device(cfg.device)
    print(f"设备: {device} | 模型: {model_name} | epochs: {cfg.epochs} | "
          f"batch: {cfg.batch_size} | seq_len: {cfg.seq_len}")

    bundle = load_swat(cfg.csv, val_frac=cfg.val_frac, scaler=cfg.scaler)
    train_loader, val_loader, test_loader = make_reconstruction_loaders(
        bundle, cfg)

    model = build_model(bundle.n_channels, cfg.seq_len, bundle).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    if cfg.load_model:
        state = torch.load(cfg.load_model, map_location=device)
        model.load_state_dict(state)
        train_time = 0.0
        print(f"[{model_name}] 已加载权重: {cfg.load_model}（跳过训练）")
    else:
        train_time = train_reconstructor(
            model, train_loader, cfg.epochs, cfg.lr, device, model_name)
        print(f"[{model_name}] 训练总耗时: {train_time:.1f}s")

    val_out = collect_recon_outputs(model, val_loader, device)
    test_out = collect_recon_outputs(model, test_loader, device)

    # AnomalyTransformer 等模型用验证集统计量做融合
    val_win, test_win = fuse_val_test(
        model, val_out["window_scores"], test_out["window_scores"])
    val_out["window_scores"] = val_win
    test_out["window_scores"] = test_win

    n_val = len(next(iter(val_out["window_scores"].values())))
    n_test = len(next(iter(test_out["window_scores"].values())))
    val_ts = _split_timestamps(bundle.val_df, n_val, cfg)
    test_ts = _split_timestamps(bundle.test_df, n_test, cfg)
    val_point_lab = _split_labels(bundle.val_label, n_val, cfg)
    test_point_lab = _split_labels(bundle.test_label, n_test, cfg)

    # ---- 重构精度（AE/VAE/AnomalyTransformer）----
    artifact_dir = ensure_dir(resolve_artifact_dir(cfg, prefix))
    has_recon = (val_out["channel"].get("recon_last") is not None
                 and val_out["channel"].get("abs_last") is not None)
    if has_recon:
        for split, out in (("val", val_out), ("test", test_out)):
            abs_last = out["channel"]["abs_last"]
            sq_last = out["channel"]["sq_last"]
            true_last = out["channel"].get("true_last")
            if true_last is not None:
                rows = _recon_rows_with_true(
                    bundle, abs_last, sq_last, true_last)
            else:
                rows = _recon_rows(bundle, abs_last, sq_last)
            name = ("recon_metrics_val.csv" if split == "val"
                    else "recon_metrics.csv")
            pd.DataFrame(rows).to_csv(
                artifact_dir / name, index=False, encoding="utf-8-sig")

    # ---- meta ----
    task = "reconstruction" if has_recon else "score"
    scaler_mean, scaler_scale = scaler_stats(bundle.scaler)
    meta = make_meta(
        model=model_name, prefix=prefix, task=task,
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
        scaler_mean=scaler_mean, scaler_scale=scaler_scale,
        scaler_type=cfg.scaler,
        extra={"recon_metrics": ("recon_metrics.csv" if has_recon else None)},
    )
    save_meta(artifact_dir, meta)

    # ---- 保存 val/test artifact ----
    for split, out, ts, point_lab in (
            ("val", val_out, val_ts, val_point_lab),
            ("test", test_out, test_ts, test_point_lab)):
        channel = out["channel"]
        save_recon_split(
            artifact_dir, split,
            label_window=out["label_window"],
            label_point=point_lab,
            timestamp=ts,
            window_scores=out["window_scores"],
            point_scores=out["point_scores"],
            # 只保存 sq_ts 和最后一点 recon/true；
            # abs_last/sq_last 等可由 Stage 2 从 recon/true 重新计算。
            ts_scores={"sq_ts": out["ts_scores"].get("sq_ts")},
            channel_arrays={
                "recon_last": channel.get("recon_last"),
                "true_last": channel.get("true_last"),
            },
        )

    # 统一保存 checkpoint，便于复现和后续只做推理
    torch.save(model.state_dict(), artifact_dir / "model.pt")

    print(f"[{model_name}] artifact 已保存: {artifact_dir}")
    return artifact_dir


def run_reconstruction(model_name: str, prefix: str,
                       build_model: Callable, description: str = "",
                       batch_size: int = 256, epochs: int = 10,
                       lr: float = 1e-4, save_model: bool = False,
                       plot_level: str = "window",
                       has_channel_metrics: bool = False,
                       plot_reconstruction: bool = False) -> dict:
    """兼容旧模型入口的两阶段包装。"""
    cfg: RunConfig = parse_config(
        description or f"{model_name} 重构/对比类模型",
        batch_size=batch_size, epochs=epochs, lr=lr)
    artifact_dir = resolve_artifact_dir(cfg, prefix)
    summary = {
        "model": model_name, "prefix": prefix,
        "artifact_dir": str(artifact_dir), "stage": cfg.stage,
    }

    if cfg.stage in ("train", "all"):
        artifact_dir = run_recon_stage(
            model_name, prefix, build_model, cfg, save_model=save_model)
        summary["artifact_dir"] = str(artifact_dir)

    if cfg.stage in ("detect", "all"):
        detection = run_detection_stage(
            artifact_dir, DetectionConfig.from_run_config(cfg))
        summary["detection"] = detection

    return summary
