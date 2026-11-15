# -*- coding: utf-8 -*-
"""共享推理、评分与阈值选择工具。"""
from __future__ import annotations

import numpy as np
import torch

from .protocol import evaluate_fixed, select_best, select_threshold


@torch.no_grad()
def collect_forecast_preds(model, loader, device: torch.device):
    """返回 (pred_first_step, true_first_step, labels_first_step)。

    兼容旧接口；多步评估请用 :func:`collect_forecast_outputs`。
    """
    model.eval()
    preds, trues, labs = [], [], []
    for x, y, lab in loader:
        out = model(x.to(device, non_blocking=True))
        preds.append(out[:, 0, :].cpu().numpy())
        trues.append(y[:, 0, :].numpy())
        lab = lab.numpy()
        labs.append(lab[:, 0] if lab.ndim > 1 else lab)
    return (np.concatenate(preds, axis=0),
            np.concatenate(trues, axis=0),
            np.concatenate(labs, axis=0))


@torch.no_grad()
def collect_forecast_outputs(model, loader, device: torch.device):
    """一次推理收集：第一步预测/真值、每步标签、每步的逐通道聚合分数。

    返回::

        first_pred [N,C], first_true [N,C], labels [N,H],
        multi_scores {"mean"|"max"|"top3_mean": [N,H]}

    为避免内存爆炸，不在内存里保存完整的 [N,H,C] 预测/真值，
    而是在每个 batch 内先做通道聚合再拼接。
    """
    model.eval()
    first_pred, first_true, labs_all = [], [], []
    score_parts = {name: [] for name in ("mean", "max", "top3_mean")}

    for x, y, lab in loader:
        x = x.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        out = model(x)

        first_pred.append(out[:, 0, :].cpu().numpy())
        first_true.append(y[:, 0, :].numpy())
        lab_np = lab.numpy()
        labs_all.append(lab_np if lab_np.ndim > 1 else lab_np[:, None])

        residual = (out - y_dev).abs()                       # [B,H,C]
        score_parts["mean"].append(residual.mean(dim=2).cpu().numpy())
        score_parts["max"].append(residual.max(dim=2).values.cpu().numpy())
        k = min(3, residual.size(2))
        top3 = torch.topk(residual, k=k, dim=2).values
        score_parts["top3_mean"].append(top3.mean(dim=2).cpu().numpy())

    return (np.concatenate(first_pred, axis=0),
            np.concatenate(first_true, axis=0),
            np.concatenate(labs_all, axis=0),
            {name: np.concatenate(parts, axis=0)
             for name, parts in score_parts.items()})


def forecast_scores(residual: np.ndarray) -> dict:
    """逐点残差聚合：mean / max / top3_mean。"""
    residual = np.asarray(residual, dtype=np.float64)
    top3 = np.sort(residual, axis=1)[:, -3:]
    return {
        "mean": residual.mean(axis=1),
        "max": residual.max(axis=1),
        "top3_mean": top3.mean(axis=1),
    }


def _to_numpy(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return np.asarray(v)


@torch.no_grad()
def collect_recon_scores(model, loader, device: torch.device):
    """收集重构/对比模型的 window、point、channel 分数。

    模型需实现 ``score_batch(x)``，返回::

        {
          "window": {name: tensor [B]},
          "point":  {name: tensor [B]},          # 可选
          "channel": {"abs_last": [B,C], ...},   # 可选
        }
    """
    model.eval()
    window, point, channel, labs = {}, {}, {}, []
    for x, lab in loader:
        out = model.score_batch(x.to(device, non_blocking=True))
        for key, store in (("window", window), ("point", point),
                           ("channel", channel)):
            for name, value in out.get(key, {}).items():
                store.setdefault(name, []).append(_to_numpy(value))
        labs.append(lab.numpy())

    def concat(d):
        return {k: np.concatenate(v, axis=0) for k, v in d.items()}

    return concat(window), concat(point), concat(channel), \
        np.concatenate(labs, axis=0)


def evaluate_score_dict(val_scores: dict, val_labels: np.ndarray,
                        test_scores: dict, test_labels: np.ndarray):
    """验证集选阈值/配置，测试集按固定阈值评估。

    返回 ``(best_name, thresholds, val_f1, results, best_result)``。
    """
    best_name, thresholds, val_f1 = select_best(val_scores, val_labels)
    results = []
    for name, score in test_scores.items():
        out = evaluate_fixed(name, score, test_labels, thresholds[name])
        out["val_F1"] = val_f1[name]
        results.append(out)
    best = next(r for r in results if r["name"] == best_name)
    return best_name, thresholds, val_f1, results, best


def _lead_aggregate(scores: np.ndarray, how: str) -> np.ndarray:
    """把 [N,H] 的多步分数沿 lead 维聚合成 [N]。"""
    if how == "mean":
        return scores.mean(axis=1)
    if how == "max":
        return scores.max(axis=1)
    if how == "top3_mean":
        k = min(3, scores.shape[1])
        return np.sort(scores, axis=1)[:, -k:].mean(axis=1)
    raise ValueError(f"未知 lead 聚合方式: {how}")


def build_window_scores(scores_multi: dict) -> dict:
    """把 {name: [N,H]} 扩展成窗口级分数 {name_leadagg: [N]}。"""
    out = {}
    for name, scores in scores_multi.items():
        for how in ("mean", "max", "top3_mean"):
            out[f"{name}_{how}"] = _lead_aggregate(scores, how)
    return out


def evaluate_leadwise(val_scores: dict, val_labels: np.ndarray,
                      test_scores: dict, test_labels: np.ndarray):
    """逐 lead 点级评估（方案 A）。

    对每个预测步 k 单独在验证集上选阈值，测试集只按固定阈值报告。
    返回 ``(all_rows, best_rows)``：前者是所有配置×所有 lead，后者是
    每个 lead 按验证集 F1 选出的最优配置。
    """
    names = list(val_scores)
    n_leads = int(val_labels.shape[1])
    all_rows, best_rows = [], []
    for k in range(n_leads):
        lead_rows = []
        for name in names:
            thr, val_f1 = select_threshold(val_scores[name][:, k],
                                           val_labels[:, k])
            row = evaluate_fixed(name, test_scores[name][:, k],
                                 test_labels[:, k], thr,
                                 threshold_source="val_best_f1")
            row["lead"] = k + 1
            row["val_F1"] = float(val_f1)
            all_rows.append(row)
            lead_rows.append(row)
        finite = [r for r in lead_rows if np.isfinite(r["val_F1"])]
        best_rows.append(max(finite, key=lambda r: r["val_F1"])
                         if finite else lead_rows[0])
    return all_rows, best_rows


def evaluate_window_level(val_scores: dict, val_labels: np.ndarray,
                          test_scores: dict, test_labels: np.ndarray):
    """窗口级评估（方案 B）。

    一个预测窗口一个分数（沿 lead 聚合成 mean/max/top3_mean），
    标签为窗口内是否出现攻击（``labels.max(axis=1)``）。
    返回 ``(best_name, thresholds, val_f1, results, best,
    test_window_scores, test_window_labels)``。
    """
    val_win = build_window_scores(val_scores)
    test_win = build_window_scores(test_scores)
    val_lab = val_labels.max(axis=1).astype(np.int64)
    test_lab = test_labels.max(axis=1).astype(np.int64)
    best_name, thresholds, val_f1, results, best = evaluate_score_dict(
        val_win, val_lab, test_win, test_lab)
    return best_name, thresholds, val_f1, results, best, test_win, test_lab


def apply_fusion(model, val_scores: dict, test_scores: dict):
    """AnomalyTransformer 等模型可选：用验证集统计量构造融合分数。"""
    if not hasattr(model, "make_fused"):
        return val_scores, test_scores
    reference = dict(val_scores)
    return (model.make_fused(reference, reference),
            model.make_fused(reference, test_scores))


def channel_metrics(true: np.ndarray, pred: np.ndarray):
    """逐通道 MAE/RMSE/MAPE/有效比例（标准化空间）。"""
    mae = float(np.mean(np.abs(true - pred)))
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    mask = np.abs(true) > 1e-6
    if mask.any():
        mape = float(np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100)
    else:
        mape = float("nan")
    return mae, rmse, mape, float(mask.mean())
