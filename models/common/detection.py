# -*- coding: utf-8 -*-
"""Stage 2：独立的异常检测流程。

输入是 Stage 1 保存的 artifact；输出是阈值、测试集指标、分数曲线。
不导入任何模型类，也不重新读取原始 CSV。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from .artifacts import load_artifact
from .plots import (plot_forecast_daily, plot_reconstruction_daily,
                    plot_score_daily)
from .protocol import (evaluate_fixed, select_threshold,
                       select_threshold_fpr)
from .scores import (forecast_candidates, normalize_pair,
                     recon_point_candidates, recon_window_candidates,
                     smooth_score)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class DetectionConfig:
    score_names: tuple[str, ...] | None = None
    channel_aggs: tuple[str, ...] = ("mean", "max", "top3_mean")
    threshold: str = "val_best_f1"        # val_best_f1 | val_fpr | rolling_fpr | fixed
    target_fpr: float = 0.01
    rolling_window: int = 3600
    fixed_threshold: float | None = None
    normalize: str = "none"               # none | rolling
    normalize_window: int = 3600
    smooth_window: int = 1
    select_by: str = "val_f1"             # val_f1 | val_pr_auc | val_auc
    no_plot: bool = False
    out_dir: str = ""                     # 为空时默认 <artifact>/detection
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_run_config(cls, cfg) -> "DetectionConfig":
        names = None
        raw = getattr(cfg, "scores", "") or ""
        if raw.strip():
            names = tuple(s.strip() for s in raw.split(",") if s.strip())
        return cls(
            score_names=names,
            threshold=getattr(cfg, "threshold", "val_best_f1"),
            target_fpr=float(getattr(cfg, "target_fpr", 0.01)),
            rolling_window=int(getattr(cfg, "rolling_window", 3600)),
            normalize=getattr(cfg, "normalize", "none"),
            smooth_window=int(getattr(cfg, "smooth_window", 1)),
            select_by=getattr(cfg, "select_by", "val_f1"),
            no_plot=bool(getattr(cfg, "no_plot", False)),
        )


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _safe_evaluate(name: str, score, labels, threshold: float,
                   source: str) -> dict[str, Any]:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    if len(score) != len(labels):
        raise ValueError(f"{name}: score 长度 {len(score)} != label 长度 {len(labels)}")
    if len(np.unique(labels)) < 2:
        return {
            "name": name, "AUC": float("nan"), "PR_AUC": float("nan"),
            "Precision": float("nan"), "Recall": float("nan"),
            "F1": float("nan"), "FPR": float("nan"),
            "threshold": float(threshold), "threshold_source": source,
        }
    if not np.isfinite(score).all():
        score = np.nan_to_num(score, nan=0.0,
                              posinf=np.finfo(np.float64).max,
                              neginf=-np.finfo(np.float64).max)
    return evaluate_fixed(name, score, labels, threshold,
                          threshold_source=source)


def _evaluate_vector(name: str, score, labels, threshold,
                     source: str) -> dict[str, Any]:
    """逐时刻阈值评估：threshold 可以是标量或与 score 等长的数组。"""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    thr = np.asarray(threshold, dtype=np.float64).reshape(-1)
    if thr.size == 1:
        thr = np.full_like(score, float(thr[0]))
    if len(score) != len(labels) or len(thr) != len(score):
        raise ValueError("score / labels / threshold 长度不一致")
    alarm = (score >= thr).astype(np.int64)
    normal = labels == 0
    out = {
        "name": name,
        "threshold": float(np.median(thr)),
        "threshold_min": float(np.min(thr)),
        "threshold_max": float(np.max(thr)),
        "threshold_source": source,
    }
    if len(np.unique(labels)) < 2:
        out.update({"AUC": float("nan"), "PR_AUC": float("nan"),
                    "Precision": float("nan"), "Recall": float("nan"),
                    "F1": float("nan"), "FPR": float("nan")})
        return out
    if not np.isfinite(score).all():
        score = np.nan_to_num(score, nan=0.0,
                              posinf=np.finfo(np.float64).max,
                              neginf=-np.finfo(np.float64).max)
    out.update({
        "AUC": float(roc_auc_score(labels, score)),
        "PR_AUC": float(average_precision_score(labels, score)),
        "Precision": float(precision_score(labels, alarm, zero_division=0)),
        "Recall": float(recall_score(labels, alarm, zero_division=0)),
        "F1": float(f1_score(labels, alarm, zero_division=0)),
        "FPR": float(alarm[normal].mean()) if normal.any() else float("nan"),
    })
    return out


def _rolling_threshold(score, window: int,
                       target_fpr: float) -> np.ndarray:
    """因果滚动阈值：threshold_t = quantile(score[t-W:t], 1-target_fpr)。"""
    s = pd.Series(np.asarray(score, dtype=np.float64))
    q = 1.0 - float(target_fpr)
    window = max(2, int(window))
    min_per = max(1, min(window, max(60, window // 10)))
    past = s.shift(1)
    thr = past.rolling(window, min_periods=min_per).quantile(q)
    warm = past.expanding(min_periods=1).quantile(q)
    thr = thr.fillna(warm)
    if thr.isna().any():
        fallback = (past.quantile(q) if past.notna().any()
                    else s.quantile(q))
        thr = thr.fillna(float(fallback))
    return thr.to_numpy(dtype=np.float64)


def _select_threshold(cfg: DetectionConfig, score, labels
                      ) -> tuple[float, str]:
    if cfg.threshold == "fixed":
        if cfg.fixed_threshold is None:
            raise ValueError("threshold=fixed 但 fixed_threshold 未设置")
        return float(cfg.fixed_threshold), "fixed"
    if cfg.threshold == "val_fpr":
        thr, _ = select_threshold_fpr(score, labels, cfg.target_fpr)
        return float(thr), f"val_normal_fpr_{cfg.target_fpr:g}"
    if cfg.threshold in ("val_best_f1", "val_f1"):
        thr, _ = select_threshold(score, labels)
        return float(thr), "val_best_f1"
    raise ValueError(f"未知阈值方式: {cfg.threshold}")


def _filter_scores(scores: dict[str, np.ndarray],
                   names: tuple[str, ...] | None
                   ) -> dict[str, np.ndarray]:
    if not names:
        return scores
    out = {k: v for k, v in scores.items() if k in names}
    missing = [n for n in names if n not in scores]
    if missing:
        print(f"[detect] 跳过不存在的分数: {', '.join(missing)}")
    return out


def _select_best(rows: list[dict], select_by: str) -> dict:
    key = {"val_f1": "val_F1",
           "val_pr_auc": "val_PR_AUC",
           "val_auc": "val_AUC"}.get(select_by)
    if key is None:
        raise ValueError(f"未知 select_by: {select_by}")

    def rank(row):
        v = row.get(key, float("nan"))
        return (np.isfinite(v), v if np.isfinite(v) else -np.inf)

    return max(rows, key=rank)


def _evaluate_level(level: str, val_scores: dict[str, np.ndarray],
                    test_scores: dict[str, np.ndarray],
                    val_labels: np.ndarray, test_labels: np.ndarray,
                    cfg: DetectionConfig) -> list[dict]:
    rows = []
    names = sorted(set(val_scores) & set(test_scores))
    for name in names:
        val_s = smooth_score(val_scores[name], cfg.smooth_window)
        test_s = smooth_score(test_scores[name], cfg.smooth_window)
        val_s, test_s = normalize_pair(
            val_s, test_s, method=cfg.normalize,
            window=cfg.normalize_window)

        if cfg.threshold == "rolling_fpr":
            val_thr = _rolling_threshold(
                val_s, cfg.rolling_window, cfg.target_fpr)
            test_thr = _rolling_threshold(
                test_s, cfg.rolling_window, cfg.target_fpr)
            source = (f"rolling_fpr_{cfg.rolling_window}_"
                      f"{cfg.target_fpr:g}")
            val_row = _evaluate_vector(
                name, val_s, val_labels, val_thr, source)
            test_row = _evaluate_vector(
                name, test_s, test_labels, test_thr, source)
            row_threshold = test_row["threshold"]
        else:
            thr, source = _select_threshold(cfg, val_s, val_labels)
            val_row = _safe_evaluate(name, val_s, val_labels, thr, source)
            test_row = _safe_evaluate(name, test_s, test_labels, thr, source)
            row_threshold = thr
        row = {
            "level": level,
            "name": name,
            "AUC": test_row["AUC"],
            "PR_AUC": test_row["PR_AUC"],
            "Precision": test_row["Precision"],
            "Recall": test_row["Recall"],
            "F1": test_row["F1"],
            "FPR": test_row["FPR"],
            "threshold": row_threshold,
            "threshold_min": test_row.get("threshold_min", float("nan")),
            "threshold_max": test_row.get("threshold_max", float("nan")),
            "threshold_source": source,
            "val_F1": val_row["F1"],
            "val_PR_AUC": val_row["PR_AUC"],
            "val_AUC": val_row["AUC"],
            "val_precision": val_row["Precision"],
            "val_recall": val_row["Recall"],
            "val_FPR": val_row["FPR"],
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_detection_stage(artifact_dir: str | Path,
                        cfg: DetectionConfig | None = None) -> dict[str, Any]:
    cfg = cfg or DetectionConfig()
    artifact_dir = Path(artifact_dir)
    art = load_artifact(artifact_dir)
    meta = art["meta"]
    task = meta.get("task", "forecast")
    val, test = art["val"], art["test"]

    print(f"\n[detect] model={meta.get('model')} | task={task} | "
          f"artifact={artifact_dir}")

    all_rows: list[dict] = []

    if task == "forecast":
        val_scores, test_scores, val_labels, test_labels = _level_score_data(
            art, "point", cfg)
        val_scores = _filter_scores(val_scores, cfg.score_names)
        test_scores = _filter_scores(test_scores, cfg.score_names)
        rows = _evaluate_level("point", val_scores, test_scores,
                               val_labels, test_labels, cfg)
        all_rows.extend(rows)

    else:
        # 窗口级
        if val.get("label_window") is not None:
            val_win, test_win, val_lab_w, test_lab_w = _level_score_data(
                art, "window", cfg)
            val_win = _filter_scores(val_win, cfg.score_names)
            test_win = _filter_scores(test_win, cfg.score_names)
            if val_win:
                rows = _evaluate_level(
                    "window", val_win, test_win,
                    val_lab_w, test_lab_w, cfg)
                all_rows.extend(rows)

        # 逐点级（AE/VAE/AT 有 point 分数时）
        if val.get("label_point") is not None:
            val_pt, test_pt, val_lab_p, test_lab_p = _level_score_data(
                art, "point", cfg)
            val_pt = _filter_scores(val_pt, cfg.score_names)
            test_pt = _filter_scores(test_pt, cfg.score_names)
            if val_pt:
                rows = _evaluate_level(
                    "point", val_pt, test_pt,
                    val_lab_p, test_lab_p, cfg)
                all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("没有可评估的分数配置，请检查 artifact 内容")

    out_dir = (Path(cfg.out_dir) if cfg.out_dir
               else artifact_dir / "detection")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(out_dir / "detection_metrics.csv",
                      index=False, encoding="utf-8-sig")

    # 每个 level 选一个最优配置
    selected = {}
    for level in sorted(metrics_df["level"].unique()):
        sub = [r for r in all_rows if r["level"] == level]
        best = _select_best(sub, cfg.select_by)
        selected[level] = best
        _save_selected_scores(art, level, best, cfg, out_dir)

    with open(out_dir / "selected_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "select_by": cfg.select_by,
            "threshold_method": cfg.threshold,
            "target_fpr": cfg.target_fpr,
            "rolling_window": cfg.rolling_window,
            "normalize": cfg.normalize,
            "smooth_window": cfg.smooth_window,
            "selected": selected,
        }, f, ensure_ascii=False, indent=2)

    # 额外输出：预测值/重构值 vs 真实值按天曲线
    _plot_true_vs_values(art, selected, cfg, out_dir)

    # 逐日指标：用来诊断阈值迁移/不同天的分数漂移
    _save_metrics_by_day(art, selected, cfg, out_dir)

    print("\n[detect] 测试集结果:")
    for level, row in selected.items():
        print(f"  [{level}] {row['name']} | "
              f"AUC {row['AUC']:.4f} | PR-AUC {row['PR_AUC']:.4f} | "
              f"P {row['Precision']:.4f} | R {row['Recall']:.4f} | "
              f"F1 {row['F1']:.4f} | FPR {row['FPR']:.4f}")
    print(f"[detect] 结果已保存: {out_dir}")

    return {"metrics": all_rows, "selected": selected,
            "artifact_dir": str(artifact_dir)}


def _level_score_data(art: dict, level: str, cfg: DetectionConfig):
    """返回 (val_scores, test_scores, val_labels, test_labels)。"""
    meta = art["meta"]
    val = art["val"]
    test = art["test"]

    if meta.get("task") == "forecast":
        vp = (val["pred"][:, 0, :]
              if val["pred"].ndim == 3 else val["pred"])
        vt = (val["true"][:, 0, :]
              if val["true"].ndim == 3 else val["true"])
        tp = test["pred"][:, 0, :] if test["pred"].ndim == 3 else test["pred"]
        tt = test["true"][:, 0, :] if test["true"].ndim == 3 else test["true"]
        vl = np.asarray(val["label"])
        tl = np.asarray(test["label"])
        vl = vl[:, 0] if vl.ndim == 2 else vl
        tl = tl[:, 0] if tl.ndim == 2 else tl
        vl, tl = vl.reshape(-1), tl.reshape(-1)
        val_scores = forecast_candidates(
            vp, vt, cfg.channel_aggs,
            ref_pred=vp, ref_true=vt, ref_labels=vl)
        test_scores = forecast_candidates(
            tp, tt, cfg.channel_aggs,
            ref_pred=vp, ref_true=vt, ref_labels=vl)
        for d, src in ((val_scores, val), (test_scores, test)):
            for k, v in src["scores"].items():
                d.setdefault(k, v)
        return val_scores, test_scores, vl, tl

    if level == "window":
        val_scores = recon_window_candidates(val, cfg.channel_aggs)
        test_scores = recon_window_candidates(test, cfg.channel_aggs)
        vl = np.asarray(val["label_window"]).reshape(-1)
        tl = np.asarray(test["label_window"]).reshape(-1)
    else:
        val_scores = recon_point_candidates(val, ref_artifact=val)
        test_scores = recon_point_candidates(test, ref_artifact=val)
        vl = np.asarray(val["label_point"]).reshape(-1)
        tl = np.asarray(test["label_point"]).reshape(-1)
    return val_scores, test_scores, vl, tl


def _selected_score_arrays(art: dict, level: str, row: dict,
                           cfg: DetectionConfig) -> dict:
    """按选中配置重建测试集分数、阈值和 TP/FP/FN。"""
    val_scores, test_scores, val_labels, test_labels = _level_score_data(
        art, level, cfg)
    name = row["name"]
    val_s = smooth_score(val_scores[name], cfg.smooth_window)
    test_s = smooth_score(test_scores[name], cfg.smooth_window)
    val_s, test_s = normalize_pair(val_s, test_s, method=cfg.normalize,
                                   window=cfg.normalize_window)

    if cfg.threshold == "rolling_fpr":
        val_thr = _rolling_threshold(
            val_s, cfg.rolling_window, cfg.target_fpr)
        test_thr = _rolling_threshold(
            test_s, cfg.rolling_window, cfg.target_fpr)
        threshold = float(np.median(test_thr))
        test_threshold = test_thr
    else:
        threshold = float(row["threshold"])
        val_thr = threshold
        test_threshold = threshold

    pred = (test_s >= test_threshold).astype(int)
    tp = ((pred == 1) & (test_labels == 1)).astype(int)
    fp = ((pred == 1) & (test_labels == 0)).astype(int)
    fn = ((pred == 0) & (test_labels == 1)).astype(int)
    tn = ((pred == 0) & (test_labels == 0)).astype(int)
    return {
        "val_score": val_s, "test_score": test_s,
        "val_labels": val_labels, "labels": test_labels,
        "val_threshold": val_thr, "test_threshold": test_threshold,
        "threshold": threshold, "pred": pred,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _save_selected_scores(art: dict, level: str, row: dict,
                          cfg: DetectionConfig, out_dir: Path) -> None:
    """保存选中配置的测试集逐点分数，并画异常分数按天图。"""
    meta = art["meta"]
    test = art["test"]
    name = row["name"]
    arr = _selected_score_arrays(art, level, row, cfg)

    df = pd.DataFrame({
        "timestamp": test["timestamp"],
        "label": arr["labels"].astype(int),
        "score": arr["test_score"],
        "threshold": arr["test_threshold"],
        "pred_anomaly": arr["pred"],
        "tp": arr["tp"], "fp": arr["fp"], "fn": arr["fn"], "tn": arr["tn"],
    })
    df.to_csv(out_dir / f"scores_{level}_{name}.csv",
              index=False, encoding="utf-8-sig")

    if not cfg.no_plot:
        try:
            ts = pd.to_datetime(test["timestamp"])
            plot_score_daily(
                timestamps=ts.values, score=arr["test_score"],
                labels=arr["labels"].astype(int),
                tp=arr["tp"], fp=arr["fp"], fn=arr["fn"], tn=arr["tn"],
                threshold=arr["test_threshold"],
                out_dir=str(out_dir / "plots" / f"{level}_{name}"),
                model_name=str(meta.get("model", "")),
                score_name=name,
            )
        except Exception as exc:  # 绘图失败不影响指标
            print(f"[detect] 绘图跳过: {exc}")


def _inverse_transform(values: np.ndarray, meta: dict) -> np.ndarray:
    """把标准化空间的值还原到原始物理量；没有 scaler 时原样返回。

    - standard: X = X' * scale + mean
    - minmax:   X = (X' - min) / scale（meta["scaler"]["mean"] 存 min）
    """
    scaler = meta.get("scaler") or {}
    mean = scaler.get("mean")
    scale = scaler.get("scale")
    values = np.asarray(values, dtype=np.float64)
    if mean is None or scale is None:
        return values
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if str(scaler.get("type", "")).startswith("minmax"):
        # minmax 与 minmax11 的逆变换都是 (X' - min_) / scale_
        return (values - mean) / scale
    return values * scale + mean


def _plot_true_vs_values(art: dict, selected: dict,
                         cfg: DetectionConfig, out_dir: Path) -> None:
    """额外画预测值 vs 真实值（或重构值 vs 真实值）按天曲线。"""
    if cfg.no_plot or not selected:
        return

    meta = art["meta"]
    test = art["test"]
    task = meta.get("task", "forecast")
    if task == "score":
        print("[detect] task=score，没有预测/重构值，跳过真实值对比图")
        return

    # 优先用 point 级选中配置的 TP/FP/FN 作为背景色带
    level = "point" if "point" in selected else next(iter(selected))
    row = selected[level]
    name = row["name"]
    bands = _selected_score_arrays(art, level, row, cfg)

    model_name = str(meta.get("model", ""))
    try:
        if task == "forecast":
            pred = (test["pred"][:, 0, :]
                    if test["pred"].ndim == 3 else test["pred"])
            true = (test["true"][:, 0, :]
                    if test["true"].ndim == 3 else test["true"])
            pred_raw = _inverse_transform(pred, meta)
            true_raw = _inverse_transform(true, meta)
            plot_forecast_daily(
                timestamps=test["timestamp"],
                true=true_raw, pred=pred_raw,
                residual=np.abs(pred_raw - true_raw),
                tp=bands["tp"], fp=bands["fp"], fn=bands["fn"],
                tn=bands["tn"], sensor_cols=meta["channels"],
                out_dir=str(out_dir / "plots"
                            / f"{level}_{name}_true_vs_pred"),
                model_name=model_name, score_name=name,
                threshold=bands["threshold"],
                horizon=int(meta.get("horizon", 1)),
            )
        elif task == "reconstruction":
            recon = test["channel"].get("recon_last")
            true = test["channel"].get("true_last")
            if recon is None or true is None:
                print("[detect] 没有 recon_last/true_last，跳过真实值对比图")
                return
            recon_raw = _inverse_transform(recon, meta)
            true_raw = _inverse_transform(true, meta)
            plot_reconstruction_daily(
                timestamps=test["timestamp"],
                true=true_raw, recon=recon_raw,
                sensor_cols=meta["channels"],
                out_dir=str(out_dir / "plots"
                            / f"{level}_{name}_true_vs_recon"),
                model_name=model_name, score_name=name,
                tp=bands["tp"], fp=bands["fp"], fn=bands["fn"],
                tn=bands["tn"],
            )
    except Exception as exc:  # 绘图失败不影响指标
        print(f"[detect] 真实值对比图跳过: {exc}")


def _save_metrics_by_day(art: dict, selected: dict,
                         cfg: DetectionConfig, out_dir: Path) -> None:
    """按自然日输出选中配置的测试集指标，诊断跨天分数漂移。"""
    if not selected:
        return
    ts = pd.to_datetime(art["test"]["timestamp"])
    dates = pd.Series(ts).dt.date
    rows = []
    for level, row in selected.items():
        arr = _selected_score_arrays(art, level, row, cfg)
        score, labels = arr["test_score"], arr["labels"]
        thr = arr["test_threshold"]
        for d in sorted(dates.unique()):
            mask = (dates == d).to_numpy()
            if not mask.any():
                continue
            if np.ndim(thr) == 0:
                thr_day = float(thr)
                ev = _safe_evaluate(
                    level, score[mask], labels[mask], thr_day,
                    row.get("threshold_source", ""))
            else:
                thr_day = np.asarray(thr)[mask]
                ev = _evaluate_vector(
                    level, score[mask], labels[mask], thr_day,
                    row.get("threshold_source", ""))
            pred = (score[mask] >= thr_day).astype(int)
            lab = labels[mask].astype(int)
            tp_n = int(((pred == 1) & (lab == 1)).sum())
            fp_n = int(((pred == 1) & (lab == 0)).sum())
            fn_n = int(((pred == 0) & (lab == 1)).sum())
            tn_n = int(((pred == 0) & (lab == 0)).sum())
            rows.append({
                "level": level, "date": str(d),
                "n": int(mask.sum()),
                "positives": int(labels[mask].sum()),
                "TP": tp_n, "FP": fp_n, "TN": tn_n, "FN": fn_n,
                "AUC": ev["AUC"], "PR_AUC": ev["PR_AUC"],
                "Precision": ev["Precision"], "Recall": ev["Recall"],
                "F1": ev["F1"], "FPR": ev["FPR"],
                "threshold": ev["threshold"],
                "threshold_source": row.get("threshold_source", ""),
            })
    if rows:
        pd.DataFrame(rows).to_csv(
            out_dir / "detection_metrics_by_day.csv",
            index=False, encoding="utf-8-sig")
