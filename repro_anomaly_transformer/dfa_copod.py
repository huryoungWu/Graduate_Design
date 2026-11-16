# -*- coding: utf-8 -*-
"""DFA-COPOD：深度特征增强的 COPOD（SWaT）。

对应《COPOD优化文档.md》的 E1–E6：
- E1 COPOD（原始特征基线）
- E2 AE（深度学习检测器基线）
- E3 AE Encoder + COPOD
- E4 TCN Encoder + COPOD
- E5 iTransformer Encoder + COPOD
- E6 原始特征 COPOD + 深度特征 COPOD 融合

协议与项目其它模型一致：
- Normal only 训练；Attack 前 20% 验证；后 80% 冻结测试；
- 深度特征：seq_len=60 滑窗 -> 编码器/隐层 -> 每窗一个特征向量；
- 潜在特征上的 COPOD 分数赋给窗口末端，再做 12 步滑窗均值，
  与深度模型的 12 步评估窗口对齐；
- 阈值：Normal99 / ValFPR1 / ValBestF1；Oracle 仅上限。

用法::

    python -m repro_anomaly_transformer.dfa_copod
    python -m repro_anomaly_transformer.dfa_copod --extractors AE TCN
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.common.config import resolve_device
from models.common.data import load_swat
from models.common.protocol import select_threshold
from models.forecasting.itransformer import ITransformer
from models.forecasting.tcn import TCNForecaster
from models.reconstruction.ae import AutoEncoder
from pyod.models.copod import COPOD

from .benchmark_all_models import _metrics, _oracle
from .iforest_fusion import (_auc_ap, _binary_metrics, _eval_continuous,
                             _normal_stats, _threshold, _transform)

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "runs" / "recon_comparison"
OUT_CSV = RC / "DFA_COPOD.csv"
OUT_MD = RC / "DFA_COPOD.md"

L = 60
W = 12
FIT_N = 200000  # COPOD 拟合的 Normal 窗口数（内存/时间）
BATCH = 2048
THRESHOLD_SOURCES = ["Normal99", "ValBestF1", "ValFPR1", "Oracle"]


# ----------------------------------------------------------------------
# 特征提取
# ----------------------------------------------------------------------
def _load_extractor(kind: str, device: torch.device):
    n_ch = 45
    if kind == "AE":
        model = AutoEncoder(L, n_ch)
        path = ROOT / "runs" / "ae_swat" / "model.pt"
    elif kind == "TCN":
        model = TCNForecaster(n_ch, L, 1)
        path = ROOT / "runs" / "tcn_swat" / "model.pt"
    elif kind == "iTransformer":
        model = ITransformer(n_ch, L, 1)
        path = ROOT / "runs" / "itransformer_swat" / "model.pt"
    else:
        raise ValueError(kind)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def _extract(model, X: np.ndarray, kind: str,
             device: torch.device) -> np.ndarray:
    view = np.lib.stride_tricks.sliding_window_view(X, L, axis=0)
    n = len(view)
    parts = []
    for start in range(0, n, BATCH):
        w = view[start:start + BATCH].transpose(0, 2, 1)  # [B,L,C]
        t = torch.from_numpy(
            np.ascontiguousarray(w)).float().to(device)
        if kind == "AE":
            f = model.encoder(t.reshape(t.size(0), -1))
        elif kind == "TCN":
            h = model.blocks(model.input_proj(t.transpose(1, 2)))
            f = h[:, :, -1]
        else:
            h = (model.input_proj(t.transpose(1, 2))
                 + model.var_emb[:, :t.size(2), :])
            h = model.norm(model.encoder(h))
            f = h.mean(dim=1)
        parts.append(f.cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


# ----------------------------------------------------------------------
# 分数与窗口对齐
# ----------------------------------------------------------------------
def _window_mean(x, w=W):
    x = np.asarray(x, dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(x, w).mean(axis=1)


def _window_max(y, w=W):
    y = np.asarray(y, dtype=np.int8)
    return np.lib.stride_tricks.sliding_window_view(y, w).max(axis=1)


def _aligned_from_end_scores(scores, y):
    """60 窗分数 -> 12 步均值，并与深度模型的测试/验证窗口对齐。"""
    s12 = _window_mean(scores, W)[1:]      # 第一个对齐到 row 71
    y12 = _window_max(y, W)[L:]            # row 71 起（L=60 -> index 60）
    n = min(len(s12), len(y12))
    return s12[:n], y12[:n]


def _aligned_raw(point_scores, y):
    """逐点分数 -> 12 步均值，对齐到 row 71（丢弃前 60 个窗口）。"""
    s12 = _window_mean(point_scores, W)[60:]
    y12 = _window_max(y, W)[60:]
    n = min(len(s12), len(y12))
    return s12[:n], y12[:n]


def _copod_scores(model, X, batch=100000):
    parts = []
    for i in range(0, len(X), batch):
        parts.append(model.decision_function(X[i:i + batch]))
    return np.concatenate(parts)


# ----------------------------------------------------------------------
# 评估
# ----------------------------------------------------------------------
def _evaluate(exp, extractor, s_normal, s_val, y_val, s_test, y_test,
              fit_s, score_s) -> list[dict]:
    nw = _window_mean(s_normal, W)
    rows = []
    for source in THRESHOLD_SOURCES:
        if source == "Normal99":
            sign, thr = 1, float(np.percentile(nw, 99))
        elif source == "ValBestF1":
            thr, _ = select_threshold(s_val, y_val)
            sign, thr = 1, float(thr)
        elif source == "ValFPR1":
            nv = s_val[y_val == 0]
            sign, thr = 1, float(np.percentile(
                nv if len(nv) else s_val, 99))
        else:
            sign, thr = _oracle(y_test, s_test)
            sign, thr = int(sign), float(thr)
        met, _ = _metrics(y_test, sign * s_test, thr)
        rows.append({
            "experiment": exp, "extractor": extractor,
            "method": "COPOD-on-features",
            "threshold_source": source, "threshold": float(thr),
            **met, "n": int(len(y_test)),
            "positives": int(np.asarray(y_test).sum()),
            "fit_s": float(fit_s), "score_s": float(score_s),
        })
    return rows


def _baseline_rows() -> list[dict]:
    rows = []
    # E1: raw COPOD
    p = RC / "ISOLATION_PEERS_ALL.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d.model == "COPOD"]
        for _, r in d.iterrows():
            rows.append({
                "experiment": "E1", "extractor": "raw",
                "method": "COPOD", "threshold_source": r["threshold_source"],
                "threshold": r.get("threshold", np.nan), "Precision": r["Precision"],
                "Recall": r["Recall"], "F1": r["F1"], "FPR": r["FPR"],
                "AUC": r["AUC"], "PR_AUC": r["PR_AUC"],
                "PA_F1": r["PA_F1"], "n": int(r["n"]),
                "positives": int(r["positives"]),
                "fit_s": np.nan, "score_s": np.nan,
            })
    # E2: AE baseline
    p = RC / "IFOREST_FUSION.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[(d.model == "AE") & (d.fusion == "Model_only")]
        for _, r in d.iterrows():
            rows.append({
                "experiment": "E2", "extractor": "AE",
                "method": "AE-recon", "threshold_source": r[
                    "threshold_source"],
                "threshold": r["threshold"], "Precision": r["Precision"],
                "Recall": r["Recall"], "F1": r["F1"], "FPR": r["FPR"],
                "AUC": r["AUC"], "PR_AUC": r["PR_AUC"],
                "PA_F1": r["PA_F1"], "n": int(r["n"]),
                "positives": int(r["positives"]),
                "fit_s": np.nan, "score_s": np.nan,
            })
    return rows


def _fuse_rows(exp, name, raw, deep, yv, yt) -> list[dict]:
    """E6：原始特征 COPOD + 深度特征 COPOD 的分数融合。"""
    rows = []
    T = min(len(raw["normal"]), len(deep["normal"]))
    rn, dn = raw["normal"][-T:], deep["normal"][-T:]
    sr, sd = _normal_stats(rn), _normal_stats(dn)
    for mode in ("z", "r"):
        rz = {"normal": _transform(rn, sr, mode),
              "val": _transform(raw["val"], sr, mode),
              "test": _transform(raw["test"], sr, mode)}
        dz = {"normal": _transform(dn, sd, mode),
              "val": _transform(deep["val"], sd, mode),
              "test": _transform(deep["test"], sd, mode)}
        fusions = {
            "mean": {k: 0.5 * (rz[k] + dz[k])
                     for k in ("normal", "val", "test")},
            "max": {k: np.maximum(rz[k], dz[k])
                    for k in ("normal", "val", "test")},
            "min": {k: np.minimum(rz[k], dz[k])
                    for k in ("normal", "val", "test")},
        }
        for method, f in fusions.items():
            for source in THRESHOLD_SOURCES:
                sign, thr = _threshold(
                    source, f["normal"], f["val"], f["test"], yv, yt)
                met = _eval_continuous(yt, f["test"], sign, thr)
                rows.append({
                    "experiment": exp, "extractor": name,
                    "method": f"{method}_{mode}",
                    "threshold_source": source, "threshold": float(thr),
                    **met, "n": int(len(yt)),
                    "positives": int(np.asarray(yt).sum()),
                    "fit_s": np.nan, "score_s": np.nan,
                })
    # AND / OR
    for method in ("AND", "OR"):
        for source in THRESHOLD_SOURCES:
            sign_r, thr_r = _threshold(
                source, rz["normal"], rz["val"], rz["test"], yv, yt)
            sign_d, thr_d = _threshold(
                source, dz["normal"], dz["val"], dz["test"], yv, yt)
            a_r = sign_r * rz["test"] > thr_r
            a_d = sign_d * dz["test"] > thr_d
            alarm = (a_r & a_d) if method == "AND" else (a_r | a_d)
            met = _binary_metrics(yt, alarm)
            cont = (np.minimum(sign_r * rz["test"], sign_d * dz["test"])
                    if method == "AND"
                    else np.maximum(sign_r * rz["test"], sign_d * dz["test"]))
            met["AUC"], met["PR_AUC"] = _auc_ap(yt, cont)
            rows.append({
                "experiment": exp, "extractor": name,
                "method": method, "threshold_source": source,
                "threshold": np.nan, **met, "n": int(len(yt)),
                "positives": int(np.asarray(yt).sum()),
                "fit_s": np.nan, "score_s": np.nan,
            })
    return rows


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="DFA-COPOD 实验")
    p.add_argument("--csv", default=str(ROOT / "08_SWaT/SWaT_all_1s.csv"))
    p.add_argument("--extractors", nargs="*",
                   default=["AE", "TCN", "iTransformer"])
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-csv", default=str(OUT_CSV))
    p.add_argument("--out-md", default=str(OUT_MD))
    args = p.parse_args()

    device = resolve_device(args.device)
    print(f"[dfa-copod] device={device}")

    print("[dfa-copod] 加载 SWaT ...")
    b = load_swat(args.csv, val_frac=0.20, scaler="standard",
                  verbose=False)
    Xtr, Xv, Xt = b.train_X, b.val_X, b.test_X
    yv, yt = b.val_label.astype(int), b.test_label.astype(int)
    ts_v = pd.to_datetime(b.val_df["timestamp"]).to_numpy()
    ts_t = pd.to_datetime(b.test_df["timestamp"]).to_numpy()
    b.train_df = b.val_df = b.test_df = None
    print(f"[dfa-copod] train={len(Xtr):,} val={len(Xv):,} "
          f"test={len(Xt):,}")

    rows = _baseline_rows()
    deep_cache = {}

    for kind in args.extractors:
        print(f"\n[dfa-copod] {kind}: 提取深度特征 ...")
        model = _load_extractor(kind, device)
        t0 = time.time()
        Ftr = _extract(model, Xtr, kind, device)
        Fv = _extract(model, Xv, kind, device)
        Ft = _extract(model, Xt, kind, device)
        feat_s = time.time() - t0
        print(f"[dfa-copod] {kind}: features {Ftr.shape} "
              f"in {feat_s:.1f}s")

        # COPOD 拟合（Normal 子采样）
        idx = np.random.RandomState(42).choice(
            len(Ftr), min(FIT_N, len(Ftr)), replace=False)
        t0 = time.time()
        copod = COPOD(contamination=0.01, n_jobs=1)
        copod.fit(Ftr[idx])
        fit_s = time.time() - t0
        t0 = time.time()
        s_n = _copod_scores(copod, Ftr)
        s_v = _copod_scores(copod, Fv)
        s_t = _copod_scores(copod, Ft)
        score_s = time.time() - t0
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 60 窗分数 -> 12 步均值，对齐 row 71
        v12, vy12 = _aligned_from_end_scores(s_v, yv)
        t12, yt12 = _aligned_from_end_scores(s_t, yt)
        # 用对齐后的标签替换（应与公共 index 一致）
        exp = {"AE": "E3", "TCN": "E4",
               "iTransformer": "E5"}[kind]
        rows += _evaluate(exp, kind, s_n, v12, vy12, t12, yt12,
                          fit_s, score_s)
        deep_cache[kind] = {
            "normal": _window_mean(s_n, W),
            "val": v12, "test": t12,
        }
        print(f"[dfa-copod] {kind}: COPOD fit={fit_s:.1f}s "
              f"score={score_s:.1f}s")

    # E6：原始 COPOD + 每个深度特征 COPOD
    raw_npz = np.load(ROOT / "runs" / "isolation_peers" / "COPOD.npz")
    raw_v, _ = _aligned_raw(raw_npz["val"], yv)
    raw_t, _ = _aligned_raw(raw_npz["test"], yt)
    raw = {
        "normal": _window_mean(raw_npz["normal"], W),
        "val": raw_v, "test": raw_t,
    }
    yv12 = _window_max(yv, W)[L:]
    yt12 = _window_max(yt, W)[L:]
    for kind, deep in deep_cache.items():
        rows += _fuse_rows("E6", kind, raw, deep, yv12, yt12)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    # ---------------- Markdown ----------------
    lines = [
        "# DFA-COPOD：深度特征增强 COPOD（SWaT）", "",
        "- 训练：Normal only；验证：Attack 前 20%；测试：Attack 后 80% 冻结",
        "- 特征：seq_len=60 滑窗 -> AE/TCN/iTransformer 隐层 -> 每窗一个特征向量",
        "- 深度特征 COPOD 分数赋给窗口末端，再做 12 步均值，"
        "与深度模型窗口对齐",
        "- 阈值：Normal99 / ValFPR1 / ValBestF1；Oracle 仅上限",
        f"- COPOD 拟合使用 {FIT_N:,} 个 Normal 窗口（固定随机种子）", "",
    ]
    for source in THRESHOLD_SOURCES:
        sub = out[out.threshold_source == source]
        # E1-E5
        base = sub[sub.experiment.isin(["E1", "E2", "E3", "E4", "E5"])]
        lines += [f"## {source}", "",
                  "| exp | model | Precision | Recall | F1 | FPR | AUC | "
                  "PR_AUC | PA_F1 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in base.iterrows():
            label = {"E1": "COPOD(raw)", "E2": "AE",
                     "E3": "AE Encoder + COPOD",
                     "E4": "TCN Encoder + COPOD",
                     "E5": "iTransformer Encoder + COPOD"}[r[
                         "experiment"]]
            lines.append(
                f"| {r['experiment']} | {label} | "
                f"{r['Precision']:.4f} | {r['Recall']:.4f} | "
                f"{r['F1']:.4f} | {r['FPR']:.4f} | {r['AUC']:.4f} | "
                f"{r['PR_AUC']:.4f} | {r['PA_F1']:.4f} |")
        # E6 best by F1
        e6 = sub[sub.experiment == "E6"]
        if not e6.empty:
            best = e6.loc[e6["F1"].idxmax()]
            lines.append(
                f"| E6 | raw + {best['extractor']} COPOD "
                f"({best['method']}) | {best['Precision']:.4f} | "
                f"{best['Recall']:.4f} | {best['F1']:.4f} | "
                f"{best['FPR']:.4f} | {best['AUC']:.4f} | "
                f"{best['PR_AUC']:.4f} | {best['PA_F1']:.4f} |")
        lines.append("")

    # E6 全部方法（ValFPR1）
    e6v = out[(out.experiment == "E6")
              & (out.threshold_source == "ValFPR1")]
    if not e6v.empty:
        lines += ["## E6 融合方法明细（ValFPR1）", "",
                  "| extractor | method | Precision | Recall | F1 | "
                  "PA_F1 | AUC |", "|---|---|---|---|---|---|---|"]
        for _, r in e6v.sort_values("F1", ascending=False).iterrows():
            lines.append(
                f"| {r['extractor']} | {r['method']} | "
                f"{r['Precision']:.4f} | {r['Recall']:.4f} | "
                f"{r['F1']:.4f} | {r['PA_F1']:.4f} | {r['AUC']:.4f} |")
        lines.append("")

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已保存: {args.out_csv}")
    print(f"已保存: {args.out_md}")

    v = out[out.threshold_source == "ValFPR1"].sort_values(
        "F1", ascending=False).head(20)
    print(v[["experiment", "extractor", "method", "F1",
             "PA_F1", "AUC"]].to_string(index=False,
                                        float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
