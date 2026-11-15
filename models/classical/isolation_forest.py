# -*- coding: utf-8 -*-
"""Isolation Forest 异常检测基线（scikit-learn）。

参考：Liu, Ting & Zhou, *Isolation Forest*, ICDM 2008；
实现：``sklearn.ensemble.IsolationForest``。

做法
----
1. 只用 Normal 段拟合 IsolationForest（无监督）；
2. 对每个时刻的传感器向量打异常分：``score = -model.score_samples(X)``
   （score_samples 越小越异常，取负后越大越异常）；
3. 用滑动窗口把逐点分数聚合成窗口分数（mean / max / top3_mean），
   与项目统一评估协议对齐；
4. 输出两种协议：
   - framework：Normal 训练、Attack 前 20% 验证、后 80% 测试，
     阈值在验证集上选（与其它模型一致）；
   - paper：整个 Attack 段测试，报告 AUC / F1@EER / oracle F1 / PA-F1。

IsolationForest 是树模型，对特征的单调缩放不敏感；这里仍用
StandardScaler（与框架一致），便于保存/反变换。

用法::

    # 统一框架协议：SWaT / WADI
    python -m models.classical.isolation_forest --protocol framework \
        --csv 08_SWaT/SWaT_all_1s.csv
    python -m models.classical.isolation_forest --protocol framework \
        --csv 09_WADI/WADI_all_1s.csv

    # 全 Attack 段评估
    python -m models.classical.isolation_forest --protocol paper \
        --csv 08_SWaT/SWaT_all_1s.csv
    python -m models.classical.isolation_forest --protocol paper \
        --csv 09_WADI/WADI_all_1s.csv

    # 复用已拟合的 IF 只做检测（framework）
    python -m models.classical.isolation_forest --protocol framework \
        --stage detect --artifact-dir runs/iforest_swat
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from models.common.artifacts import (ensure_dir, make_meta, save_meta,
                                     save_recon_split)
from models.common.config import (PROJECT_ROOT, RunConfig, build_parser,
                                  set_seed)
from models.common.data import load_swat
from models.common.detection import DetectionConfig, run_detection_stage
from models.reconstruction.usad_paper import (
    _plot_roc, _plot_score_hist, evaluate_scores, load_periods)

MODEL_NAME = "IsolationForest"


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def _window_outputs(point_scores: np.ndarray, labels: np.ndarray,
                    timestamps: np.ndarray, seq_len: int,
                    prefix: str = "iforest"):
    """把逐点分数聚合成窗口分数，并返回窗口/逐点标签与时间戳。

    ``prefix`` 决定分数名（IF：iforest_*；DAGMM：dagmm_*），
    这样同一个 artifact 格式可以复用于不同模型。
    """
    point_scores = np.asarray(point_scores, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int8)
    L = int(seq_len)
    if len(point_scores) < L:
        raise ValueError("序列长度小于窗口长度")
    sw = np.lib.stride_tricks.sliding_window_view(point_scores, L)  # [N,L]
    k = min(3, L)
    win_scores = {
        f"{prefix}_mean": sw.mean(axis=1),
        f"{prefix}_max": sw.max(axis=1),
        f"{prefix}_top3_mean": np.sort(sw, axis=1)[:, -k:].mean(axis=1),
    }
    win_labels = np.lib.stride_tricks.sliding_window_view(
        labels, L).max(axis=1)
    ts = np.asarray(timestamps)[L - 1:]
    point_scores_out = {
        f"point_{prefix}": point_scores[L - 1:],
        f"point_{prefix}_window_max": win_scores[f"{prefix}_max"],
    }
    point_labels = labels[L - 1:]
    return win_scores, win_labels, ts, point_scores_out, point_labels


def _fit_iforest(X_train: np.ndarray, args) -> tuple[IsolationForest, float]:
    t0 = time.time()
    model = IsolationForest(
        n_estimators=int(args.n_estimators),
        max_samples=args.max_samples,
        contamination=args.contamination,
        max_features=1.0,
        bootstrap=False,
        random_state=int(args.seed),
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_train)
    return model, time.time() - t0


def _score(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    # sklearn: score_samples 越低越异常；取负使分数越高越异常
    return (-model.score_samples(X)).astype(np.float64)


def _dataset_name(csv: str) -> str:
    return "wadi" if "wadi" in str(csv).lower() else "swat"


# ----------------------------------------------------------------------
# 统一框架协议
# ----------------------------------------------------------------------
def run_framework(cfg: RunConfig, args) -> None:
    prefix = f"iforest_{_dataset_name(cfg.csv)}"
    artifact_dir = (Path(cfg.artifact_dir) if cfg.artifact_dir
                    else Path(cfg.out_dir) / "runs" / prefix)

    if cfg.stage == "detect":
        run_detection_stage(
            artifact_dir, DetectionConfig.from_run_config(cfg))
        return

    print(f"[{MODEL_NAME}] 读取数据...")
    bundle = load_swat(cfg.csv, val_frac=cfg.val_frac, scaler="standard")
    n_train, n_val, n_test = (len(bundle.train_X), len(bundle.val_X),
                              len(bundle.test_X))
    print(f"[{MODEL_NAME}] 传感器数={bundle.n_channels} | "
          f"train={n_train:,} | val={n_val:,} | test={n_test:,}")
    # 只保留时间戳与数组，释放带传感器列的 DataFrame（WADI 约 0.8 GB）
    val_ts_all = bundle.val_df["timestamp"].to_numpy()
    test_ts_all = bundle.test_df["timestamp"].to_numpy()
    split_info = {
        "split_idx": int(bundle.split_idx),
        "val_end": int(bundle.val_end),
        "train_rows": n_train,
        "val_rows": n_val,
        "test_rows": n_test,
    }
    sensor_cols = bundle.sensor_cols
    bundle.train_df = bundle.val_df = None
    bundle.test_df = None

    print(f"[{MODEL_NAME}] 拟合 IsolationForest "
          f"(n_estimators={args.n_estimators}, "
          f"max_samples={args.max_samples}, "
          f"contamination={args.contamination}) ...")
    model, fit_time = _fit_iforest(bundle.train_X, args)
    bundle.train_X = None   # 拟合后不再需要
    print(f"[{MODEL_NAME}] 拟合耗时 {fit_time:.1f}s")

    artifact_dir = ensure_dir(artifact_dir)
    meta = make_meta(
        model=MODEL_NAME, prefix=prefix, task="score",
        channels=sensor_cols, seq_len=cfg.seq_len,
        pred_len=1, horizon=1, val_frac=cfg.val_frac,
        split=split_info,
        train={
            "fit_time_s": float(fit_time),
            "n_estimators": int(args.n_estimators),
            "max_samples": str(args.max_samples),
            "contamination": str(args.contamination),
            "feature_mode": "point",
            "score_aggregation": ["mean", "max", "top3_mean"],
        },
        scaler_mean=bundle.scaler.mean_,
        scaler_scale=bundle.scaler.scale_,
        scaler_type="standard",
        extra={"score_aggregation": "point -> sliding window"},
    )
    save_meta(artifact_dir, meta)

    for split, X, labels, ts in (
            ("val", bundle.val_X, bundle.val_label, val_ts_all),
            ("test", bundle.test_X, bundle.test_label, test_ts_all)):
        ps = _score(model, X)
        win, wl, wts, pts, pl = _window_outputs(
            ps, labels, ts, cfg.seq_len)
        save_recon_split(
            artifact_dir, split,
            label_window=wl, label_point=pl, timestamp=wts,
            window_scores=win, point_scores=pts)
        print(f"[{MODEL_NAME}] {split}: {len(wl):,} 个窗口，"
              f"正例率 {float(wl.mean()):.4f}")

    print(f"[{MODEL_NAME}] artifact 已保存: {artifact_dir}")
    if cfg.stage in ("all", "detect"):
        run_detection_stage(
            artifact_dir, DetectionConfig.from_run_config(cfg))


# ----------------------------------------------------------------------
# 全 Attack 段协议
# ----------------------------------------------------------------------
def run_paper(cfg: RunConfig, args) -> None:
    dataset = _dataset_name(cfg.csv)
    out_arg = Path(args.out_dir) if args.out_dir else None
    if out_arg is None or out_arg.resolve() == PROJECT_ROOT.resolve():
        out_dir = Path("runs") / f"iforest_paper_{dataset}"
    else:
        out_dir = out_arg
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{MODEL_NAME}-paper] dataset={dataset}")
    normal, attack, sensor_cols = load_periods(cfg.csv)
    print(f"[{MODEL_NAME}-paper] Normal {len(normal):,} | "
          f"Attack {len(attack):,} | 变量 {len(sensor_cols)}")

    scaler = StandardScaler()
    normal_X = scaler.fit_transform(
        normal[sensor_cols].to_numpy(np.float64)).astype(np.float32)
    attack_X = scaler.transform(
        attack[sensor_cols].to_numpy(np.float64)).astype(np.float32)
    attack_y = attack["label"].to_numpy(np.int8)
    attack_ts = attack["timestamp"].to_numpy()
    del normal, attack

    model, fit_time = _fit_iforest(normal_X, args)
    print(f"[{MODEL_NAME}-paper] 拟合耗时 {fit_time:.1f}s")
    point_scores = _score(model, attack_X)
    del attack_X, normal_X

    win, wl, wts, pts, pl = _window_outputs(
        point_scores, attack_y, attack_ts, cfg.seq_len)
    results = {
        name: evaluate_scores(name, arr, wl, attack_y, cfg.seq_len)
        for name, arr in win.items()
    }

    metrics = {
        "dataset": dataset,
        "protocol": "full attack segment (oracle thresholds)",
        "model": MODEL_NAME,
        "seq_len": int(cfg.seq_len),
        "n_estimators": int(args.n_estimators),
        "max_samples": str(args.max_samples),
        "contamination": str(args.contamination),
        "fit_time_s": float(fit_time),
        "n_test_windows": int(len(wl)),
        "scores": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "timestamp": pd.to_datetime(wts),
        **{f"score_{k}": v for k, v in win.items()},
        "score_point_iforest": pts["point_iforest"],
        "label_window": wl.astype(int),
    }).to_csv(out_dir / "window_scores.csv", index=False,
              encoding="utf-8-sig")

    if not args.no_plot:
        try:
            _plot_roc(wl, win["iforest_mean"], out_dir / "roc.png",
                      f"[{MODEL_NAME}] {dataset.upper()} window-level ROC")
            _plot_score_hist(wl, win["iforest_mean"],
                             out_dir / "score_hist.png",
                             f"[{MODEL_NAME}] {dataset.upper()} "
                             f"score distribution")
        except Exception as exc:
            print(f"[{MODEL_NAME}-paper] 绘图跳过: {exc}")

    for name, res in results.items():
        w = res["window_level"]
        pa = res["point_level_PA"]
        print(f"[{MODEL_NAME}-paper] {name}: AUC {w['AUC']:.4f} | "
              f"F1@EER {w['F1_eer']:.4f} | "
              f"F1@oracle {w['F1_best_f1_oracle']:.4f} | "
              f"PA-F1@oracle {pa['F1_oracle']:.4f}")
    print(f"[{MODEL_NAME}-paper] 结果已保存: {out_dir}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_args():
    p = build_parser(
        "Isolation Forest 异常检测（SWaT / WADI）",
        batch_size=256, epochs=1, lr=1e-3)
    p.add_argument("--protocol", choices=["framework", "paper"],
                   default="framework",
                   help="framework=统一两阶段协议；paper=全 Attack 段评估")
    p.add_argument("--n-estimators", type=int, default=100,
                   help="树数量（sklearn 默认 100）")
    p.add_argument("--max-samples", default="auto",
                   help="每棵树的采样数：'auto' 或整数（默认 256）")
    p.add_argument("--contamination", default="auto",
                   help="'auto' 或 (0,0.5] 的比例；只影响 offset_，"
                        "本实现用 score_samples 排序，不受影响")
    args = p.parse_args()
    # max_samples 允许 "auto" 或整数
    try:
        args.max_samples = int(args.max_samples)
    except (TypeError, ValueError):
        args.max_samples = "auto"
    try:
        args.contamination = float(args.contamination)
    except (TypeError, ValueError):
        args.contamination = "auto"

    fields = set(RunConfig.__dataclass_fields__)
    cfg = RunConfig(**{k: v for k, v in vars(args).items() if k in fields})
    cfg.out_dir = Path(cfg.out_dir)
    if args.protocol == "framework":
        if not any(a == "--seq-len" or a.startswith("--seq-len=")
                   for a in sys.argv[1:]):
            cfg.seq_len = 12
        if not any(a == "--scaler" or a.startswith("--scaler=")
                   for a in sys.argv[1:]):
            cfg.scaler = "standard"
        if cfg.smoke:
            cfg.val_frac = cfg.val_frac
    return cfg, args


def main() -> None:
    cfg, args = _parse_args()
    set_seed(int(args.seed))
    if args.protocol == "paper":
        run_paper(cfg, args)
    else:
        run_framework(cfg, args)


if __name__ == "__main__":
    main()
