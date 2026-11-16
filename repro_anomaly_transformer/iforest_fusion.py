# -*- coding: utf-8 -*-
"""IsolationForest 与预测/重构模型的多策略分数融合。

动机
----
在 ``THRESHOLD_COMPARISON_ALL.md`` 中，IsolationForest（IF）在冻结测试集上
的 F1 / PA-F1 明显高于各预测/重构模型。本脚本尝试把 IF 的分数作为
**第二个专家**，与每个模型的窗口分数融合，看能否在保持高召回的同时
提升精度。

协议（与 ``benchmark_all_models.py`` / ``threshold_comparison_all.py`` 一致）
----------------------------------------------------------------------------
- 训练：Normal only；
- 验证：Attack 前 20%，只用于选阈值、融合权重、stacking 参数；
- 测试：Attack 后 80%，冻结，只报告；
- 12 步窗口聚合（单点分数 -> 12 步滑窗均值）。

阈值口径
--------
- Normal99：Normal 段窗口分数的 99% 分位（只用正常数据）；
- ValBestF1：验证集上 F1 最优阈值（用验证集标签）；
- ValFPR1：验证集**正常**窗口分数的 99% 分位（目标 FPR=1%）；
- Oracle：测试集标签搜阈值，仅作上限，不可部署。

融合方法
--------
score-level（连续分数）：
- ``mean_z`` / ``weighted_z``：Normal 段 z-score 后平均 / 加权平均；
- ``mean_robust`` / ``weighted_robust``：用 Normal 段 median 与 q99 做
  鲁棒归一化后平均 / 加权平均；
- ``max_z`` / ``min_z``：逐窗口取 max / min；
- ``stack_lr`` / ``stack_lr3``：在验证集上拟合逻辑回归（2 维 / 3 维特征）。

decision-level（报警级）：
- ``or``：任一模型报警即报警；
- ``and``：两个模型都报警才报警。

所有可学习参数（alpha、LR 权重、阈值）只在验证集上确定，测试集不参与。

用法::

    python -m repro_anomaly_transformer.iforest_fusion
    python -m repro_anomaly_transformer.iforest_fusion --models LSTM AE VAE
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)

from models.common.config import resolve_device
from models.common.data import load_swat
from models.common.protocol import select_threshold
from models.reconstruction.usad_extended_metrics import point_adjust

from .benchmark_all_models import (MODELS, _metrics, _oracle, _point_frame,
                                   _window12)

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "runs" / "recon_comparison" / "IFOREST_FUSION.csv"
OUT_MD = ROOT / "runs" / "recon_comparison" / "IFOREST_FUSION.md"

WINDOW = 12
# 兼容旧 normal99 脚本：这里改为覆盖整个 Normal 段
ALPHA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)

# artifact -> (module, family, scaler, build kwargs)
ARTIFACT_SPECS = {
    "lstm_swat": ("models.forecasting.lstm", "forecast", "standard", {}),
    "transformer_swat": ("models.forecasting.transformer",
                         "forecast", "standard", {}),
    "itransformer_swat": ("models.forecasting.itransformer",
                          "forecast", "standard", {}),
    "tcn_swat": ("models.forecasting.tcn", "forecast", "standard", {}),
    "timesnet_swat": ("models.forecasting.timesnet",
                      "forecast", "standard", {}),
    "micn_swat": ("models.forecasting.micn", "forecast", "standard", {}),
    "moderntcn_swat": ("models.forecasting.moderntcn",
                       "forecast", "standard", {}),
    "seq2seq_lstm_swat": ("models.forecasting.seq2seq_lstm",
                          "forecast", "standard", {}),
    "ae_swat": ("models.reconstruction.ae", "reconstruction",
                "standard", {}),
    "vae_swat": ("models.reconstruction.vae", "reconstruction",
                 "standard", {}),
    "anomalytransformer_swat": ("models.reconstruction.anomaly_transformer",
                                "reconstruction", "standard", {}),
    "usad_swat": ("models.reconstruction.usad", "reconstruction",
                  "minmax", {"hidden_size": 10}),
    "usad_h100": ("models.reconstruction.usad", "reconstruction",
                  "minmax", {"hidden_size": 100}),
    "lstmvae_swat": ("models.reconstruction.lstm_vae", "reconstruction",
                     "minmax11", {}),
}

THRESHOLD_SOURCES = ["Normal99", "ValBestF1", "ValFPR1", "Oracle"]
FUSION_METHODS = [
    "mean_z", "weighted_z", "mean_robust", "weighted_robust",
    "max_z", "min_z", "surprisal_sum", "surprisal_max",
    "surprisal_min", "stack_lr", "stack_lr3", "or", "and",
]
DECISION_METHODS = {"or", "and"}


# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------
def _sliding_mean(x, w: int = WINDOW) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(x, w).mean(axis=1)


def _common_index(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    names = list(frames)
    idx = frames[names[0]].index
    for name in names[1:]:
        idx = idx.intersection(frames[name].index)
    return pd.DatetimeIndex(sorted(idx))


def _normal_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    mean, std = float(np.mean(x)), float(np.std(x))
    med = float(np.median(x))
    q99 = float(np.percentile(x, 99))
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    denom = q99 - med
    if not np.isfinite(denom) or denom < 1e-12:
        denom = std
    return {"mean": mean, "std": std, "med": med, "q99": q99,
            "denom": denom}


def _transform(x, stats: dict, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if mode == "z":
        return (x - stats["mean"]) / stats["std"]
    return (x - stats["med"]) / stats["denom"]


def _f1(y, pred) -> float:
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    return float(f1)


def _binary_metrics(y, alarm) -> dict:
    """决策级融合的指标；AUC/PR_AUC 由调用方用连续分数补齐。"""
    alarm = np.asarray(alarm, dtype=np.int64)
    m, _ = _metrics(y, alarm.astype(float), 0.0)
    return m


def _auc_ap(y, s) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return (float(roc_auc_score(y, s)),
            float(average_precision_score(y, s)))


# ----------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------
def _load_frames() -> tuple[dict, dict, dict]:
    val_frames, test_frames, meta = {}, {}, {}
    for name, artifact, family in MODELS:
        art = ROOT / "runs" / artifact
        if not (art / "val.npz").exists() or not (art / "test.npz").exists():
            print(f"[skip] {name}: artifact 不存在 ({art})")
            continue
        try:
            v, _ = _window12(_point_frame(artifact, "val"))
            t, _ = _window12(_point_frame(artifact, "test"))
            val_frames[name] = v
            test_frames[name] = t
            meta[name] = {
                "artifact": artifact,
                "family": family,
                "meta": json.loads(
                    (art / "meta.json").read_text(encoding="utf-8")),
            }
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {name}: {exc}")
    if "IsolationForest" not in test_frames:
        raise RuntimeError("缺少 IsolationForest artifact，无法做融合")
    return val_frames, test_frames, meta


def _build_model(artifact: str, meta: dict, family: str,
                 device: torch.device):
    module_name, _, _, kwargs = ARTIFACT_SPECS[artifact]
    mod = importlib.import_module(module_name)
    n_channels = len(meta["channels"])
    seq_len = int(meta["seq_len"])
    pred_len = int(meta.get("pred_len", 1))
    if family == "forecast":
        model = mod.build_model(n_channels, seq_len, pred_len, None)
    else:
        model = mod.build_model(n_channels, seq_len, None, **kwargs)
    state = torch.load(ROOT / "runs" / artifact / "model.pt",
                       map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


CACHE_DIR = ROOT / "runs" / "recon_comparison" / "iforest_fusion_cache"


def _compute_normal_window_scores(artifact: str, family: str,
                                  meta: dict, bundles: dict,
                                  device: torch.device) -> np.ndarray:
    if artifact == "iforest_swat":
        bundle = bundles["standard"]
        n_est = int(meta.get("train", {}).get("n_estimators", 100))
        model = IsolationForest(
            n_estimators=n_est, max_samples="auto",
            contamination="auto", max_features=1.0, bootstrap=False,
            random_state=42, n_jobs=-1)
        model.fit(bundle.train_X)
        ps = -model.score_samples(bundle.train_X)
        return _sliding_mean(ps, WINDOW)

    _, _, scaler, _ = ARTIFACT_SPECS[artifact]
    bundle = bundles[scaler]
    model = _build_model(artifact, meta, family, device)
    npm = importlib.import_module(
        "models.classical.normal99_project_models")
    # 让 _normal_point_scores 覆盖整个 Normal 段
    npm.N_NORMAL = len(bundle.train_X)
    ps = npm._normal_point_scores(
        model, bundle.train_X, meta, family, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return _sliding_mean(ps, WINDOW)


def _normal_window_scores(name: str, artifact: str, family: str,
                          meta: dict, bundles: dict,
                          device: torch.device) -> np.ndarray:
    """整个 Normal 段的 12 步窗口分数（带磁盘缓存）。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{artifact}_normal.npy"
    meta_path = ROOT / "runs" / artifact / "meta.json"
    if cache.exists():
        try:
            if (not meta_path.exists()
                    or cache.stat().st_mtime >= meta_path.stat().st_mtime):
                return np.load(cache)
        except OSError:
            pass
    arr = _compute_normal_window_scores(
        artifact, family, meta, bundles, device)
    np.save(cache, arr)
    return arr


def _prepare_data(device: torch.device, csv: str,
                  wanted: set[str] | None = None):
    val_frames, test_frames, meta = _load_frames()
    names = [n for n in test_frames
             if wanted is None or n in wanted or n == "IsolationForest"]
    common_val = _common_index({n: val_frames[n] for n in names})
    common_test = _common_index({n: test_frames[n] for n in names})
    print(f"[fusion] 模型数={len(names)} | common val={len(common_val):,} "
          f"| common test={len(common_test):,}")

    y_val = val_frames[names[0]].loc[common_val, "label"].to_numpy(int)
    y_test = test_frames[names[0]].loc[common_test, "label"].to_numpy(int)
    for n in names[1:]:
        yv = val_frames[n].loc[common_val, "label"].to_numpy(int)
        yt = test_frames[n].loc[common_test, "label"].to_numpy(int)
        if not np.array_equal(yv, y_val):
            print(f"[warn] {n}: val 标签与基准不一致")
        if not np.array_equal(yt, y_test):
            print(f"[warn] {n}: test 标签与基准不一致")

    bundles = {}

    def get_bundle(scaler: str):
        if scaler not in bundles:
            print(f"[fusion] 加载 SWaT (scaler={scaler}) ...")
            b = load_swat(csv, val_frac=0.20, scaler=scaler,
                          verbose=False)
            # 只保留数组，释放 DataFrame
            b.train_df = b.val_df = b.test_df = None
            bundles[scaler] = b
        return bundles[scaler]

    # 先按需加载所有 scaler（只针对本次选中的模型）
    selected_artifacts = {meta[n]["artifact"] for n in names}
    for artifact, (_, _, scaler, _) in ARTIFACT_SPECS.items():
        if artifact in selected_artifacts:
            get_bundle(scaler)
    get_bundle("standard")  # IF

    normal = {}
    for name in names:
        art = meta[name]["artifact"]
        print(f"[fusion] 计算 Normal 段分数: {name} ...", flush=True)
        normal[name] = _normal_window_scores(
            name, art, meta[name]["family"], meta[name]["meta"],
            bundles, device)

    # 对齐到末尾（所有 Normal 序列都以 Normal 段最后一行结束）
    t_min = min(len(v) for v in normal.values())
    for name in normal:
        normal[name] = normal[name][-t_min:]

    val_scores = {n: val_frames[n].loc[common_val, "score"].to_numpy(float)
                  for n in names}
    test_scores = {n: test_frames[n].loc[common_test, "score"].to_numpy(float)
                   for n in names}
    return names, meta, normal, val_scores, test_scores, y_val, y_test


# ----------------------------------------------------------------------
# 融合方法
# ----------------------------------------------------------------------
def _tune_alpha(s_if, s_m, y) -> tuple[float, float]:
    best_alpha, best_f1 = 0.0, -1.0
    for alpha in ALPHA_GRID:
        fused = alpha * s_if + (1.0 - alpha) * s_m
        _, f1 = select_threshold(fused, y)
        if np.isfinite(f1) and f1 > best_f1:
            best_alpha, best_f1 = float(alpha), float(f1)
    return best_alpha, best_f1


def _fit_lr(x_if, x_m, y, cubic: bool) -> LogisticRegression:
    feats = [x_if, x_m]
    if cubic:
        feats.append(x_if * x_m)
    X = np.column_stack(feats)
    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    lr.fit(X, y)
    return lr


def _lr_score(lr: LogisticRegression, x_if, x_m, cubic: bool) -> np.ndarray:
    feats = [x_if, x_m]
    if cubic:
        feats.append(x_if * x_m)
    return lr.decision_function(np.column_stack(feats))


def _orient_scores(out: dict, y_val: np.ndarray) -> dict:
    """保证融合分数方向为“越大越异常”（方向只用验证集标签确定）。"""
    v = np.asarray(out["val"], dtype=np.float64)
    pos, neg = v[np.asarray(y_val) == 1], v[np.asarray(y_val) == 0]
    if pos.size and neg.size and float(np.mean(pos)) < float(np.mean(neg)):
        return {k: -np.asarray(val, dtype=np.float64)
                for k, val in out.items()}
    return {k: np.asarray(val, dtype=np.float64)
            for k, val in out.items()}


def _surprisal_map(normal_scores, scores) -> np.ndarray:
    """把分数映射为相对 Normal 上尾的 surprisal ``-log P(X >= s)``。

    q90 以下用经验 CDF；q90 以上用指数尾拟合外推，避免大量攻击分数
    都饱和到同一个最大值而失去区分度。
    """
    x = np.sort(np.asarray(normal_scores, dtype=np.float64))
    s = np.asarray(scores, dtype=np.float64)
    n = len(x)
    if n < 50:
        return np.zeros_like(s)
    u = (np.searchsorted(x, s, side="right") + 0.5) / (n + 1.0)
    out = -np.log(np.clip(1.0 - u, 1e-12, 1.0))
    q90 = float(np.quantile(x, 0.90))
    mask = x > q90
    if int(mask.sum()) >= 20:
        xt = x[mask]
        ranks = np.flatnonzero(mask) + 1.0
        ut = (ranks + 0.5) / (n + 1.0)
        tt = -np.log(np.clip(1.0 - ut, 1e-12, 1.0))
        slope, intercept = np.polyfit(xt, tt, 1)
        if np.isfinite(slope) and slope > 0:
            tail = s > q90
            out[tail] = intercept + slope * s[tail]
    return out


def _build_fusion(method: str, s_if: dict, s_m: dict,
                  y_val: np.ndarray) -> dict:
    """返回 continuous 融合分数（normal/val/test）及附加信息。"""
    info = {"alpha": np.nan, "val_f1": np.nan}
    if method.startswith("mean_"):
        mode = "z" if method.endswith("_z") else "robust"
        out = {k: 0.5 * (s_if[mode][k] + s_m[mode][k])
               for k in ("normal", "val", "test")}
    elif method.startswith("weighted_"):
        mode = "z" if method.endswith("_z") else "robust"
        alpha, val_f1 = _tune_alpha(
            s_if[mode]["val"], s_m[mode]["val"], y_val)
        info["alpha"] = alpha
        info["val_f1"] = val_f1
        out = {k: alpha * s_if[mode][k] + (1.0 - alpha) * s_m[mode][k]
               for k in ("normal", "val", "test")}
    elif method in ("max_z", "min_z"):
        op = np.maximum if method == "max_z" else np.minimum
        out = {k: op(s_if["z"][k], s_m["z"][k])
               for k in ("normal", "val", "test")}
    elif method.startswith("surprisal"):
        keys = ("normal", "val", "test")
        t_if = {k: _surprisal_map(s_if["z"]["normal"], s_if["z"][k])
                for k in keys}
        t_m = {k: _surprisal_map(s_m["z"]["normal"], s_m["z"][k])
               for k in keys}
        if method == "surprisal_sum":
            out = {k: t_if[k] + t_m[k] for k in keys}
        elif method == "surprisal_max":
            out = {k: np.maximum(t_if[k], t_m[k]) for k in keys}
        else:
            out = {k: np.minimum(t_if[k], t_m[k]) for k in keys}
    elif method.startswith("stack_lr"):
        cubic = method.endswith("3")
        lr = _fit_lr(s_if["z"]["val"], s_m["z"]["val"], y_val, cubic)
        out = {k: _lr_score(lr, s_if["z"][k], s_m["z"][k], cubic)
               for k in ("normal", "val", "test")}
        info["coef"] = lr.coef_.ravel().tolist()
    else:
        raise ValueError(method)
    out = _orient_scores(out, y_val)
    _, info["val_f1"] = select_threshold(out["val"], y_val)
    return {**info, "kind": "continuous", **out}


def _fuse_decision(s_if: dict, s_m: dict, y_val: np.ndarray,
                   combine: str) -> dict:
    """OR/AND 的验证集阈值与验证集 F1。"""
    # 验证集上各自选 F1 阈值（只用于方法选择，正式测试会换口径）
    t_if, _ = select_threshold(s_if["z"]["val"], y_val)
    t_m, _ = select_threshold(s_m["z"]["val"], y_val)
    a_if = s_if["z"]["val"] > t_if
    a_m = s_m["z"]["val"] > t_m
    alarm = (a_if | a_m) if combine == "or" else (a_if & a_m)
    info = {"val_f1": _f1(y_val, alarm), "kind": "decision"}
    if combine == "or":
        info["continuous"] = np.maximum(s_if["z"]["val"], s_m["z"]["val"])
    else:
        info["continuous"] = np.minimum(s_if["z"]["val"], s_m["z"]["val"])
    return info


# ----------------------------------------------------------------------
# 阈值与评估
# ----------------------------------------------------------------------
def _threshold(source: str, normal_s, val_s, test_s, y_val, y_test):
    """返回 (sign, threshold)；Oracle 允许翻转方向。"""
    if source == "Normal99":
        return 1, float(np.percentile(normal_s, 99))
    if source == "ValBestF1":
        thr, _ = select_threshold(val_s, y_val)
        return 1, float(thr)
    if source == "ValFPR1":
        normal_val = np.asarray(val_s)[np.asarray(y_val) == 0]
        if normal_val.size == 0:
            return 1, float(np.percentile(val_s, 99))
        return 1, float(np.percentile(normal_val, 99))
    if source == "Oracle":
        sign, thr = _oracle(y_test, test_s)
        return int(sign), float(thr)
    raise ValueError(source)


def _eval_continuous(y, score, sign, thr) -> dict:
    m, _ = _metrics(y, sign * np.asarray(score, dtype=np.float64),
                    float(thr))
    return m


def _val_f1_at_source(source: str, normal_s, val_s, y_val) -> float:
    """验证集上、按指定阈值口径得到的 F1（用于选融合方法）。"""
    if source == "Oracle":
        return float("nan")
    if source == "Normal99":
        thr = float(np.percentile(normal_s, 99))
    elif source == "ValFPR1":
        val_s = np.asarray(val_s, dtype=np.float64)
        normal_val = val_s[np.asarray(y_val) == 0]
        thr = (float(np.percentile(normal_val, 99))
               if normal_val.size else float(np.percentile(val_s, 99)))
    elif source == "ValBestF1":
        thr, _ = select_threshold(val_s, y_val)
        thr = float(thr)
    else:
        raise ValueError(source)
    return _f1(y_val, np.asarray(val_s) > thr)


def _decision_val_f1(source: str, s_if: dict, s_m: dict,
                     y_val, combine: str) -> float:
    if source == "Oracle":
        return float("nan")
    if source == "Normal99":
        t_if = float(np.percentile(s_if["z"]["normal"], 99))
        t_m = float(np.percentile(s_m["z"]["normal"], 99))
    elif source == "ValFPR1":
        n_if = s_if["z"]["val"][np.asarray(y_val) == 0]
        n_m = s_m["z"]["val"][np.asarray(y_val) == 0]
        t_if = float(np.percentile(n_if, 99))
        t_m = float(np.percentile(n_m, 99))
    elif source == "ValBestF1":
        t_if, _ = select_threshold(s_if["z"]["val"], y_val)
        t_m, _ = select_threshold(s_m["z"]["val"], y_val)
        t_if, t_m = float(t_if), float(t_m)
    else:
        raise ValueError(source)
    a_if = s_if["z"]["val"] > t_if
    a_m = s_m["z"]["val"] > t_m
    alarm = (a_if | a_m) if combine == "or" else (a_if & a_m)
    return _f1(y_val, alarm)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="IF 与各模型的多策略分数融合")
    p.add_argument("--csv", default=str(ROOT / "08_SWaT/SWaT_all_1s.csv"))
    p.add_argument("--models", nargs="*", default=None,
                   help="只跑指定模型（如 LSTM AE VAE）")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    device = resolve_device(args.device)
    wanted = set(args.models) | {"IsolationForest"} if args.models else None
    (names, meta, normal, val_scores, test_scores,
     y_val, y_test) = _prepare_data(device, args.csv, wanted)
    if len(names) < 2:
        raise RuntimeError("至少需要一个深度模型 + IsolationForest")

    # 每个源的归一化分数
    src = {}
    for name in names:
        st = _normal_stats(normal[name])
        src[name] = {
            "z": {k: _transform(v, st, "z") for k, v in (
                ("normal", normal[name]), ("val", val_scores[name]),
                ("test", test_scores[name]))},
            "robust": {k: _transform(v, st, "robust") for k, v in (
                ("normal", normal[name]), ("val", val_scores[name]),
                ("test", test_scores[name]))},
        }

    if "IsolationForest" not in names:
        raise RuntimeError("--models 需要包含 IsolationForest 或留空")
    if_if = src["IsolationForest"]

    rows = []
    summary = []
    for name in names:
        if name == "IsolationForest":
            continue
        sm = src[name]
        model_meta = meta[name]
        print(f"\n[fusion] {name} ({model_meta['artifact']})")

        # 基线
        for baseline_name, baseline in (
                ("Model_only", sm), ("IF_only", if_if)):
            for source in THRESHOLD_SOURCES:
                sign, thr = _threshold(
                    source, baseline["z"]["normal"],
                    baseline["z"]["val"], baseline["z"]["test"],
                    y_val, y_test)
                met = _eval_continuous(y_test, baseline["z"]["test"],
                                       sign, thr)
                rows.append({
                    "model": name, "fusion": baseline_name,
                    "threshold_source": source, "threshold": thr,
                    "val_F1": np.nan, "alpha": np.nan,
                    "val_source_F1": _val_f1_at_source(
                        source, baseline["z"]["normal"],
                        baseline["z"]["val"], y_val),
                    **met, "n": int(len(y_test)),
                    "positives": int(y_test.sum()),
                })

        # 融合方法
        fusion_info = {}
        for method in FUSION_METHODS:
            if method in DECISION_METHODS:
                info = _fuse_decision(if_if, sm, y_val, method)
                fusion_info[method] = info
                for source in THRESHOLD_SOURCES:
                    t_if, thr_if = _threshold(
                        source, if_if["z"]["normal"], if_if["z"]["val"],
                        if_if["z"]["test"], y_val, y_test)
                    t_m, thr_m = _threshold(
                        source, sm["z"]["normal"], sm["z"]["val"],
                        sm["z"]["test"], y_val, y_test)
                    a_if = (t_if * if_if["z"]["test"] > thr_if)
                    a_m = (t_m * sm["z"]["test"] > thr_m)
                    alarm = (a_if | a_m) if method == "or" else (a_if & a_m)
                    met = _binary_metrics(y_test, alarm)
                    cont = (np.maximum(t_if * if_if["z"]["test"],
                                       t_m * sm["z"]["test"])
                            if method == "or"
                            else np.minimum(t_if * if_if["z"]["test"],
                                            t_m * sm["z"]["test"]))
                    auc, ap = _auc_ap(y_test, cont)
                    met["AUC"] = auc
                    met["PR_AUC"] = ap
                    rows.append({
                        "model": name, "fusion": method,
                        "threshold_source": source,
                        "threshold": np.nan, "val_F1": info["val_f1"],
                        "alpha": np.nan,
                        "val_source_F1": _decision_val_f1(
                            source, if_if, sm, y_val, method),
                        **met, "n": int(len(y_test)),
                        "positives": int(y_test.sum()),
                    })
            else:
                info = _build_fusion(method, if_if, sm, y_val)
                fusion_info[method] = info
                for source in THRESHOLD_SOURCES:
                    sign, thr = _threshold(
                        source, info["normal"], info["val"],
                        info["test"], y_val, y_test)
                    met = _eval_continuous(y_test, info["test"],
                                           sign, thr)
                    rows.append({
                        "model": name, "fusion": method,
                        "threshold_source": source, "threshold": thr,
                        "val_F1": info["val_f1"],
                        "alpha": info.get("alpha", np.nan),
                        "val_source_F1": _val_f1_at_source(
                            source, info["normal"], info["val"], y_val),
                        **met, "n": int(len(y_test)),
                        "positives": int(y_test.sum()),
                    })

        # 按部署口径选择：验证集 ValFPR1 阈值下的 F1 最高者
        primary = [r for r in rows
                   if r["model"] == name
                   and r["threshold_source"] == "ValFPR1"
                   and r["fusion"] in FUSION_METHODS]
        candidates = {r["fusion"]: r["val_source_F1"] for r in primary
                      if np.isfinite(r["val_source_F1"])}
        if not candidates:
            candidates = {k: v["val_f1"] for k, v in fusion_info.items()
                          if np.isfinite(v["val_f1"])}
        best_method = max(candidates, key=candidates.get)
        best_val_f1 = float(candidates[best_method])
        best_rows = [r for r in rows
                     if r["model"] == name and r["fusion"] == best_method]
        base_rows = [r for r in rows
                     if r["model"] == name and r["fusion"] == "Model_only"]
        if_rows = [r for r in rows
                   if r["model"] == name and r["fusion"] == "IF_only"]
        and_rows = [r for r in rows
                    if r["model"] == name and r["fusion"] == "and"]
        best_alpha = next((r["alpha"] for r in best_rows
                           if np.isfinite(r["alpha"])), np.nan)
        summary.append({
            "model": name, "best_method": best_method,
            "best_val_F1": best_val_f1, "alpha": best_alpha,
            "best_rows": best_rows, "base_rows": base_rows,
            "if_rows": if_rows, "and_rows": and_rows,
        })
        print(f"  val-selected={best_method} "
              f"(val_F1@ValFPR1={best_val_f1:.4f}，仅参考)")

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # ---------------- Markdown ----------------
    lines = [
        "# IsolationForest 融合实验结果（SWaT）", "",
        "- 训练：Normal only；验证：Attack 前 20%；测试：Attack 后 80% 冻结",
        "- 12 步窗口；融合权重 / 阈值只在验证集上选择",
        "- `Model_only` / `IF_only` 为基线；其余为融合方法",
        "- `Normal99` 用 Normal 段窗口分数 99% 分位；"
        "`ValFPR1` 用验证集正常分数 99% 分位",
        "- `Oracle` 使用测试标签，仅作上限", "",
        "## 融合方法", "",
        "- score-level：`mean_z` / `weighted_z`（Normal z-score 平均/加权），",
        "  `mean_robust` / `weighted_robust`（median/q99 鲁棒归一化），",
        "  `max_z` / `min_z`，`surprisal_sum` / `surprisal_max` /",
        "  `surprisal_min`（Normal 上尾 surprisal），`stack_lr` / `stack_lr3`；",
        "- decision-level：`or`（任一报警）、`and`（两者都报警）。", "",
        "## 推荐融合：AND（IF 报警 且 模型也报警）", "",
        "- IF 精度高、深度模型召回高；AND 用模型给 IF 的报警做二次确认，",
        "  在不牺牲太多召回的前提下去掉一部分 IF 误报。",
        "- `Normal99` / `ValFPR1` 的阈值分别由 Normal 段和验证集正常段",
        "  的分位数确定，不使用验证集攻击标签，对分布漂移更稳健。",
        "- `ValBestF1` 列仍使用验证集标签选阈值，仅作对照。", "",
    ]
    for th in THRESHOLD_SOURCES:
        lines += [f"### {th}", "",
                  "| model | IF_only F1 | AND F1 | ΔF1 | "
                  "IF_only PA_F1 | AND PA_F1 | ΔPA_F1 | "
                  "Model_only F1 | IF_only AUC | AND AUC |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for s in summary:
            i = next(r for r in s["if_rows"]
                     if r["threshold_source"] == th)
            a = next(r for r in s["and_rows"]
                     if r["threshold_source"] == th)
            b = next(r for r in s["base_rows"]
                     if r["threshold_source"] == th)
            lines.append(
                f"| {s['model']} | {i['F1']:.4f} | {a['F1']:.4f} | "
                f"{a['F1'] - i['F1']:+.4f} | "
                f"{i['PA_F1']:.4f} | {a['PA_F1']:.4f} | "
                f"{a['PA_F1'] - i['PA_F1']:+.4f} | "
                f"{b['F1']:.4f} | {i['AUC']:.4f} | {a['AUC']:.4f} |")
        lines.append("")

    # 汇总均值（IF / AND / Model）
    lines += ["## 汇总（各模型平均）", "",
              "| threshold | IF_only F1 | AND F1 | ΔF1 | "
              "Model_only F1 | IF_only PA_F1 | AND PA_F1 | "
              "Model_only PA_F1 |",
              "|---|---|---|---|---|---|---|---|"]
    for th in THRESHOLD_SOURCES:
        vals = {k: [] for k in
                ("if_f1", "and_f1", "base_f1",
                 "if_pa", "and_pa", "base_pa")}
        for s in summary:
            i = next(r for r in s["if_rows"]
                     if r["threshold_source"] == th)
            a = next(r for r in s["and_rows"]
                     if r["threshold_source"] == th)
            b = next(r for r in s["base_rows"]
                     if r["threshold_source"] == th)
            vals["if_f1"].append(i["F1"])
            vals["and_f1"].append(a["F1"])
            vals["base_f1"].append(b["F1"])
            vals["if_pa"].append(i["PA_F1"])
            vals["and_pa"].append(a["PA_F1"])
            vals["base_pa"].append(b["PA_F1"])
        mean = {k: float(np.mean(v)) for k, v in vals.items()}
        lines.append(
            f"| {th} | {mean['if_f1']:.4f} | {mean['and_f1']:.4f} | "
            f"{mean['and_f1'] - mean['if_f1']:+.4f} | "
            f"{mean['base_f1']:.4f} | {mean['if_pa']:.4f} | "
            f"{mean['and_pa']:.4f} | {mean['base_pa']:.4f} |")
    lines.append("")
    lines += ["**AND 相对 IF 的提升统计**", ""]
    for th in THRESHOLD_SOURCES:
        df1, dpa = [], []
        for s in summary:
            i = next(r for r in s["if_rows"]
                     if r["threshold_source"] == th)
            a = next(r for r in s["and_rows"]
                     if r["threshold_source"] == th)
            df1.append(a["F1"] - i["F1"])
            dpa.append(a["PA_F1"] - i["PA_F1"])
        wins = int(sum(d > 0 for d in df1))
        lines.append(
            f"- {th}: AND 在 {wins}/{len(df1)} 个模型上 F1 更高，"
            f"平均 ΔF1={float(np.mean(df1)):+.4f}，"
            f"平均 ΔPA_F1={float(np.mean(dpa)):+.4f}")
    lines.append("")

    # 验证集自由选择（仅参考）
    lines += ["## 验证集自由选择的方法（仅参考）", "",
              "- 验证集（Attack 前 20%）与测试集存在明显分布漂移：",
              f"  验证集正例率 {float(np.mean(y_val)):.4f}，"
              f"测试集正例率 {float(np.mean(y_test)):.4f}。",
              "- 因此按验证集 F1 选出的融合方法在测试集上未必更好；",
              "  上表 AND 方法不依赖验证集攻击标签，是更稳健的选择。", "",
              "| model | best_by_val | val_F1@ValFPR1 | test F1 | "
              "test PA_F1 | IF_only F1 | IF_only PA_F1 |",
              "|---|---|---|---|---|---|---|"]
    for s in summary:
        f_val = next((r["val_source_F1"] for r in s["best_rows"]
                      if r["threshold_source"] == "ValFPR1"), np.nan)
        f = next(r for r in s["best_rows"]
                 if r["threshold_source"] == "ValFPR1")
        i = next(r for r in s["if_rows"]
                 if r["threshold_source"] == "ValFPR1")
        lines.append(
            f"| {s['model']} | {s['best_method']} | {f_val:.4f} | "
            f"{f['F1']:.4f} | {f['PA_F1']:.4f} | "
            f"{i['F1']:.4f} | {i['PA_F1']:.4f} |")
    lines.append("")

    # 与旧表对比
    old_path = (ROOT / "runs" / "recon_comparison"
                / "THRESHOLD_COMPARISON_ALL.csv")
    if old_path.exists():
        old = pd.read_csv(old_path)
        lines += ["## 与旧表 THRESHOLD_COMPARISON_ALL 对比", "",
                  "- 旧表使用旧 checkpoint；新基线为本次重训结果，数值接近即可",
                  "- 新基线 Normal99 统一用整个 Normal 段；旧表对项目模型用",
                  "  最后 5 万点，因此项目模型可能略有差异；ValFPR1 口径一致",
                  "- `ΔAND vs old IF` = 新 AND F1/PA_F1 - 旧表 IF F1/PA_F1",
                  ""]
        for th in ["Normal99", "ValFPR1", "ValBestF1"]:
            oi = old[(old.model == "IsolationForest")
                     & (old.method == th)]
            if oi.empty:
                continue
            oi = oi.iloc[0]
            lines += [f"### {th}", "",
                      "| model | old model F1 | new model F1 | "
                      "old IF F1 | new IF F1 | AND F1 | "
                      "ΔAND vs old IF | old IF PA_F1 | AND PA_F1 | "
                      "ΔPA vs old IF |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for s in summary:
                o = old[(old.model == s["model"]) & (old.method == th)]
                om_f1 = (float(o.iloc[0]["F1"]) if len(o) else np.nan)
                b = next(r for r in s["base_rows"]
                         if r["threshold_source"] == th)
                i = next(r for r in s["if_rows"]
                         if r["threshold_source"] == th)
                a = next(r for r in s["and_rows"]
                         if r["threshold_source"] == th)
                lines.append(
                    f"| {s['model']} | {om_f1:.4f} | {b['F1']:.4f} | "
                    f"{float(oi['F1']):.4f} | {i['F1']:.4f} | "
                    f"{a['F1']:.4f} | "
                    f"{a['F1'] - float(oi['F1']):+.4f} | "
                    f"{float(oi['PA_F1']):.4f} | {a['PA_F1']:.4f} | "
                    f"{a['PA_F1'] - float(oi['PA_F1']):+.4f} |")
            lines.append("")

    # 测试集事后最优融合（仅分析）
    lines += ["## 附：测试集事后最优融合（仅分析，不可部署）", "",
              "- 用测试集 F1 事后挑选融合方法，只用于看融合的上限；",
              "  正式方法选择不能使用测试标签。", "",
              "| model | method | F1 | PA_F1 | IF_only F1 | IF_only PA_F1 | "
              "ΔF1 | ΔPA_F1 |",
              "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        sub = out[(out.model == s["model"])
                  & (out.threshold_source == "ValFPR1")
                  & (out.fusion.isin(FUSION_METHODS))]
        if sub.empty:
            continue
        best = sub.loc[sub["F1"].idxmax()]
        i = next(r for r in s["if_rows"]
                 if r["threshold_source"] == "ValFPR1")
        lines.append(
            f"| {s['model']} | {best['fusion']} | {best['F1']:.4f} | "
            f"{best['PA_F1']:.4f} | {i['F1']:.4f} | {i['PA_F1']:.4f} | "
            f"{best['F1'] - i['F1']:+.4f} | "
            f"{best['PA_F1'] - i['PA_F1']:+.4f} |")
    lines.append("")

    # IF 工作点分析（Normal 分位阈值）
    lines += ["## 附：IF 工作点分析（Normal 分位阈值）", "",
              "- 以下阈值全部来自 Normal 段分位数，不使用验证/测试标签；",
              "  但“分位点选哪一个”属于部署预算，不能靠验证集选择",
              "  （验证集与测试集分布漂移严重）。",
              "- 仅用于说明：IF 的收益主要来自工作点选择，而非分数融合。", "",
              "| Normal quantile | threshold | Precision | Recall | F1 | "
              "PA_F1 |", "|---|---|---|---|---|---|"]
    for q in [0.95, 0.97, 0.98, 0.985, 0.99, 0.992, 0.995, 0.997]:
        thr = float(np.percentile(if_if["z"]["normal"], q * 100.0))
        met = _eval_continuous(y_test, if_if["z"]["test"], 1, thr)
        lines.append(
            f"| {q:.3f} | {thr:.4f} | {met['Precision']:.4f} | "
            f"{met['Recall']:.4f} | {met['F1']:.4f} | "
            f"{met['PA_F1']:.4f} |")
    lines.append("")

    # 方法选择统计
    from collections import Counter
    wins = Counter(s["best_method"] for s in summary)
    lines += ["## 验证集自由选择的方法分布", ""]
    for method, count in wins.most_common():
        lines.append(f"- `{method}`：{count} 个模型")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    # 控制台摘要
    print("\n===== IF vs AND (ValFPR1) =====")
    for s in summary:
        i = next(r for r in s["if_rows"]
                 if r["threshold_source"] == "ValFPR1")
        a = next(r for r in s["and_rows"]
                 if r["threshold_source"] == "ValFPR1")
        b = next(r for r in s["base_rows"]
                 if r["threshold_source"] == "ValFPR1")
        print(f"{s['model']:28s} "
              f"IF {i['F1']:.4f}/{i['PA_F1']:.4f}  "
              f"AND {a['F1']:.4f}/{a['PA_F1']:.4f}  "
              f"Model {b['F1']:.4f}/{b['PA_F1']:.4f}")
    print(f"\n已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")


if __name__ == "__main__":
    main()
