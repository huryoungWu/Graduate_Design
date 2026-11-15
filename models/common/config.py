# -*- coding: utf-8 -*-
"""共享配置、命令行参数与设备/随机种子工具。"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

# 使用确定性 cuBLAS 算法时必须设置（需要 CUDA >= 10.2）；
# 必须在第一次 cuBLAS 调用前设置。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "08_SWaT" / "SWaT_all_1s.csv"


@dataclass
class RunConfig:
    csv: str = str(DEFAULT_CSV)
    seq_len: int = 60
    pred_len: int = 1          # 每个窗口输出的预测步数（评估只取第一步）
    horizon: int = 1           # 预测提前量：输入末时刻为 t，预测从 t+horizon 开始
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-4
    seed: int = 42
    device: str = "auto"
    scaler: str = "standard"   # standard | minmax（USAD 官方实现使用 MinMaxScaler）
    out_dir: Path = field(default_factory=lambda: PROJECT_ROOT)
    no_plot: bool = False
    smoke: bool = False
    num_workers: int = 0
    val_frac: float = 0.20    # Attack 段中划作验证集的比例（建议保持 0.20）
    deterministic: bool = False  # 强制确定性算法（可复现，但可能稍慢）
    load_model: str = ""       # 加载已训练权重（跳过训练），用于同权重对比

    # ---- 两阶段流程 ----
    stage: str = "all"         # all | train | detect
    artifact_dir: str = ""     # 为空时默认 <out_dir>/runs/<prefix>
    scores: str = ""           # 检测阶段只评估这些分数（逗号分隔；空=全部）
    threshold: str = "val_best_f1"  # val_best_f1 | val_fpr | rolling_fpr | fixed
    target_fpr: float = 0.01
    rolling_window: int = 3600   # rolling_fpr 的因果窗口（秒）
    fixed_threshold: float | None = None
    normalize: str = "none"    # none | rolling
    smooth_window: int = 1
    select_by: str = "val_f1"  # val_f1 | val_pr_auc | val_auc
    save_all_steps: bool = False  # 预测类是否保存全部预测步（默认只存第一步）


def build_parser(description: str, batch_size: int = 256,
                 epochs: int = 10, lr: float = 1e-4) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="SWaT CSV 路径")
    p.add_argument("--seq-len", type=int, default=RunConfig.seq_len,
                   help="输入窗口长度（默认取 RunConfig.seq_len）")
    p.add_argument("--pred-len", type=int, default=RunConfig.pred_len,
                   help="每个窗口输出的预测步数，仅预测类模型使用（评估只取第一步）")
    p.add_argument("--horizon", type=int, default=RunConfig.horizon,
                   help="预测提前量，仅预测类模型使用：输入末时刻为 t，预测从 "
                        "t+horizon 开始；1=下一时刻，60=预测 60 秒后")
    p.add_argument("--batch-size", type=int, default=batch_size)
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--seed", type=int, default=RunConfig.seed)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--scaler", default=RunConfig.scaler,
                   choices=["standard", "minmax", "minmax11"],
                   help="标准化方式：standard=StandardScaler；"
                        "minmax=MinMaxScaler [0,1]（USAD 官方设置）；"
                        "minmax11=MinMaxScaler [-1,1]（LSTM-VAE 官方设置）")
    p.add_argument("--out-dir", default=str(PROJECT_ROOT),
                   help="结果输出目录（默认项目根目录，保持原文件名）")
    p.add_argument("--no-plot", action="store_true", help="不生成按天图片")
    p.add_argument("--smoke", action="store_true", help="小数据快速自检")
    p.add_argument("--num-workers", type=int, default=RunConfig.num_workers)
    p.add_argument("--val-frac", type=float, default=RunConfig.val_frac,
                   help="Attack 段中划作验证集的比例")
    p.add_argument("--deterministic", action="store_true",
                   help="启用确定性算法：关闭 cuDNN autotune、使用确定性 cuBLAS，"
                        "同一条命令重复运行结果完全一致（可能稍慢）")
    p.add_argument("--load-model", default=RunConfig.load_model,
                   help="加载已训练模型权重并跳过训练（支持基线或三分支 checkpoint），"
                        "用于同权重对比；需保证 seq_len/pred_len/horizon 一致")
    p.add_argument("--stage", choices=["all", "train", "detect"],
                   default=RunConfig.stage,
                   help="all=训练+检测；train=只训练并保存 artifact；"
                        "detect=只读 artifact 做检测")
    p.add_argument("--artifact-dir", default=RunConfig.artifact_dir,
                   help="artifact 目录；为空时默认 <out-dir>/runs/<prefix>")
    p.add_argument("--scores", default=RunConfig.scores,
                   help="检测阶段只评估这些分数（逗号分隔，空=全部）")
    p.add_argument("--threshold",
                   choices=["val_best_f1", "val_fpr", "rolling_fpr", "fixed"],
                   default=RunConfig.threshold,
                   help="阈值选择方式")
    p.add_argument("--target-fpr", type=float, default=RunConfig.target_fpr,
                   help="val_fpr / rolling_fpr 的目标误报率")
    p.add_argument("--rolling-window", type=int,
                   default=RunConfig.rolling_window,
                   help="rolling_fpr 的因果滚动窗口（样本点/秒）")
    p.add_argument("--fixed-threshold", type=float,
                   default=RunConfig.fixed_threshold,
                   help="threshold=fixed 时使用的固定阈值")
    p.add_argument("--normalize", choices=["none", "rolling"],
                   default=RunConfig.normalize,
                   help="检测阶段是否对分数做滚动鲁棒归一化")
    p.add_argument("--smooth-window", type=int, default=RunConfig.smooth_window,
                   help="检测前对分数做滑动平均的窗口（1=不平滑）")
    p.add_argument("--select-by",
                   choices=["val_f1", "val_pr_auc", "val_auc"],
                   default=RunConfig.select_by,
                   help="在验证集上选择评分配置的依据")
    p.add_argument("--save-all-steps", action="store_true",
                   default=RunConfig.save_all_steps,
                   help="预测类保存全部预测步（默认只保存第一步）")
    return p


def parse_config(description: str, batch_size: int = 256,
                 epochs: int = 10, lr: float = 1e-4) -> RunConfig:
    args = build_parser(description, batch_size, epochs, lr).parse_args()
    cfg = RunConfig(**vars(args))
    cfg.out_dir = Path(cfg.out_dir)
    if cfg.smoke:
        cfg.epochs = 1
    return cfg


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[config] 未检测到 CUDA，改用 CPU")
        return torch.device("cpu")
    return torch.device(device)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # 让 GPU 算子也走确定性实现；同一命令重复运行结果一致。
        # 注意：这只能保证"同一段代码"可复现，不能保证不同模型实现
        # 得到同一组初始权重。
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "mha"):
            torch.backends.mha.set_fastpath_enabled(False)
        torch.use_deterministic_algorithms(True, warn_only=True)
