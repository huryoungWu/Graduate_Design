# -*- coding: utf-8 -*-
"""DFA-COPOD 优化版：PCA 白化、60 步评估、特征级融合、灰区级联。

在 ``dfa_copod.py`` 的 E1–E6 基础上增加：
- E3w/E4w/E5w：PCA 白化后的隐特征 + COPOD；
- 12-step 与 60-step 两套评估；
- E7：原始窗口末点特征 + 白化隐特征拼接 + COPOD（特征级融合）；
- E8：原始 COPOD 高置信直接报警，灰区用深度 COPOD 确认（灰区级联）。

输出：
- runs/recon_comparison/DFA_COPOD_OPT.csv
- runs/recon_comparison/DFA_COPOD_OPT.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from models.common.config import resolve_device
from models.common.data import load_swat
from models.common.protocol import select_threshold
from pyod.models.copod import COPOD

from .benchmark_all_models import _metrics, _oracle, _point_frame
from .dfa_copod import (_copod_scores, _extract, _load_extractor,
                        _window_max, _window_mean, FIT_N, L, W)

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "runs" / "recon_comparison"
OUT_CSV = RC / "DFA_COPOD_OPT.csv"
OUT_MD = RC / "DFA_COPOD_OPT.md"

SOURCES = ["Normal99", "ValBestF1", "ValFPR1", "Oracle"]
PCA_DIM = 32


# ----------------------------------------------------------------------
# 分数准备
# ----------------------------------------------------------------------
def _prep_point(scores, y, protocol: str):
    if protocol == "12":
        s = _window_mean(scores, W)[60:]
        yy = _window_max(y, W)[60:]
    else:
        s = _window_mean(scores, 60)
        yy = _window_max(y, 60)
    n = min(len(s), len(yy))
    return s[:n], yy[:n]


def _prep_window(scores, y, protocol: str):
    if protocol == "12":
        s = _window_mean(scores, W)[1:]
        yy = _window_max(y, W)[60:]
    else:
        s = np.asarray(scores)
        yy = _window_max(y, 60)
    n = min(len(s), len(yy))
    return s[:n], yy[:n]


def _prep_normal(scores, protocol: str, is_point: bool):
    if is_point:
        return _window_mean(scores, W if protocol == "12" else 60)
    return _window_mean(scores, W) if protocol == "12" else np.asarray(scores)


def _evaluate(exp, extractor, method, protocol, s_n, sv, yv, st, yt,
              fit_s=np.nan, score_s=np.nan):
    rows = []
    nv = min(len(sv), len(yv))
    nt = min(len(st), len(yt))
    sv, yv = sv[:nv], yv[:nv]
    st, yt = st[:nt], yt[:nt]
    n = nt
    for source in SOURCES:
        if source == "Normal99":
            sign, thr = 1, float(np.percentile(s_n, 99))
        elif source == "ValBestF1":
            thr, _ = select_threshold(sv, yv)
            sign, thr = 1, float(thr)
        elif source == "ValFPR1":
            nv = sv[yv == 0]
            sign, thr = 1, float(np.percentile(
                nv if len(nv) else sv, 99))
        else:
            sign, thr = _oracle(yt, st)
            sign, thr = int(sign), float(thr)
        met, _ = _metrics(yt, sign * st, thr)
        rows.append({
            "protocol": protocol, "experiment": exp,
            "extractor": extractor, "method": method,
            "threshold_source": source, "threshold": float(thr),
            **met, "n": int(n), "positives": int(np.asarray(yt).sum()),
            "fit_s": float(fit_s), "score_s": float(score_s),
        })
    return rows


def _fit_copod(X, seed=42):
    idx = np.random.RandomState(seed).choice(
        len(X), min(FIT_N, len(X)), replace=False)
    t0 = time.time()
    m = COPOD(contamination=0.01, n_jobs=1)
    m.fit(X[idx])
    return m, time.time() - t0


def _fuse_protocol(raw, deep, yv, yt, protocol):
    """返回 raw/deep 在指定 protocol 下的归一化分数与标签。"""
    rn = _prep_normal(raw["normal"], protocol, True)
    dn = _prep_normal(deep["normal"], protocol, False)
    n_norm = min(len(rn), len(dn))
    rn, dn = rn[-n_norm:], dn[-n_norm:]
    rv, yv2 = _prep_point(raw["val"], yv, protocol)
    rt, yt2 = _prep_point(raw["test"], yt, protocol)
    dv, yv3 = _prep_window(deep["val"], yv, protocol)
    dt, yt3 = _prep_window(deep["test"], yt, protocol)
    nv = min(len(rv), len(dv), len(yv2), len(yv3))
    nt = min(len(rt), len(dt), len(yt2), len(yt3))
    rv, dv = rv[:nv], dv[:nv]
    rt, dt = rt[:nt], dt[:nt]
    yv2, yv3 = yv2[:nv], yv3[:nv]
    yt2, yt3 = yt2[:nt], yt3[:nt]
    # 标签应一致；以原始 COPOD 的窗口标签为准
    yv = yv2
    yt = yt2
    st_r = {"mean": rn.mean(), "std": rn.std()}
    st_d = {"mean": dn.mean(), "std": dn.std()}
    if st_r["std"] < 1e-12:
        st_r["std"] = 1.0
    if st_d["std"] < 1e-12:
        st_d["std"] = 1.0
    zr = {"normal": (rn - st_r["mean"]) / st_r["std"],
          "val": (rv - st_r["mean"]) / st_r["std"],
          "test": (rt - st_r["mean"]) / st_r["std"]}
    zd = {"normal": (dn - st_d["mean"]) / st_d["std"],
          "val": (dv - st_d["mean"]) / st_d["std"],
          "test": (dt - st_d["mean"]) / st_d["std"]}
    return zr, zd, yv, yt


def _fuse_rows(exp, extractor, raw, deep, yv, yt, protocol):
    rows = []
    zr, zd, yv, yt = _fuse_protocol(raw, deep, yv, yt, protocol)
    fusions = {
        "mean_z": {k: 0.5 * (zr[k] + zd[k])
                   for k in ("normal", "val", "test")},
        "max_z": {k: np.maximum(zr[k], zd[k])
                  for k in ("normal", "val", "test")},
        "min_z": {k: np.minimum(zr[k], zd[k])
                  for k in ("normal", "val", "test")},
    }
    for method, f in fusions.items():
        rows += _evaluate(exp, extractor, method, protocol,
                          f["normal"], f["val"], yv, f["test"], yt)
    for method in ("AND", "OR"):
        for source in SOURCES:
            if source == "Normal99":
                tr, td = (np.percentile(zr["normal"], 99),
                          np.percentile(zd["normal"], 99))
                sr = sd = 1
            elif source == "ValFPR1":
                tr = np.percentile(zr["val"][yv == 0], 99)
                td = np.percentile(zd["val"][yv == 0], 99)
                sr = sd = 1
            elif source == "ValBestF1":
                tr, _ = select_threshold(zr["val"], yv)
                td, _ = select_threshold(zd["val"], yv)
                sr = sd = 1
            else:
                sr, tr = _oracle(yt, zr["test"])
                sd, td = _oracle(yt, zd["test"])
                sr, sd = int(sr), int(sd)
            ar = sr * zr["test"] > tr
            ad = sd * zd["test"] > td
            alarm = (ar & ad) if method == "AND" else (ar | ad)
            pred = alarm.astype(int)
            from sklearn.metrics import (average_precision_score,
                                         precision_recall_fscore_support,
                                         roc_auc_score)
            p, r, f1, _ = precision_recall_fscore_support(
                yt, pred, average="binary", zero_division=0)
            neg = yt == 0
            fpr = float(((pred == 1) & neg).sum() / max(neg.sum(), 1))
            auc = float(roc_auc_score(yt, pred)) if len(
                np.unique(yt)) > 1 else np.nan
            ap = float(average_precision_score(yt, pred)) if len(
                np.unique(yt)) > 1 else np.nan
            from models.reconstruction.usad_extended_metrics import (
                point_adjust)
            pa = point_adjust(yt.astype(int), pred)
            pp, pr, pf1, _ = precision_recall_fscore_support(
                yt, pa, average="binary", zero_division=0)
            rows.append({
                "protocol": protocol, "experiment": exp,
                "extractor": extractor, "method": method,
                "threshold_source": source, "threshold": np.nan,
                "Precision": float(p), "Recall": float(r),
                "F1": float(f1), "FPR": fpr, "AUC": auc,
                "PR_AUC": ap, "PA_F1": float(pf1),
                "n": int(len(yt)), "positives": int(yt.sum()),
                "fit_s": np.nan, "score_s": np.nan,
            })
    return rows


def _cascade_rows(exp, extractor, raw, deep, yv, yt, protocol):
    rows = []
    zr, zd, yv, yt = _fuse_protocol(raw, deep, yv, yt, protocol)
    for source in ("Normal99", "ValFPR1"):
        if source == "Normal99":
            rh = np.percentile(zr["normal"], 99.5)
            rl = np.percentile(zr["normal"], 99.0)
            dm = np.percentile(zd["normal"], 99.0)
        else:
            rv = zr["val"][yv == 0]
            dv = zd["val"][yv == 0]
            rh = np.percentile(rv, 99.5)
            rl = np.percentile(rv, 99.0)
            dm = np.percentile(dv, 99.0)
        alarm = ((zr["test"] > rh)
                 | ((zr["test"] > rl) & (zd["test"] > dm)))
        pred = alarm.astype(int)
        from sklearn.metrics import (average_precision_score,
                                     precision_recall_fscore_support,
                                     roc_auc_score)
        p, r, f1, _ = precision_recall_fscore_support(
            yt, pred, average="binary", zero_division=0)
        neg = yt == 0
        fpr = float(((pred == 1) & neg).sum() / max(neg.sum(), 1))
        from models.reconstruction.usad_extended_metrics import point_adjust
        pa = point_adjust(yt.astype(int), pred)
        pp, pr, pf1, _ = precision_recall_fscore_support(
            yt, pa, average="binary", zero_division=0)
        rows.append({
            "protocol": protocol, "experiment": exp,
            "extractor": extractor, "method": "cascade",
            "threshold_source": source, "threshold": np.nan,
            "Precision": float(p), "Recall": float(r), "F1": float(f1),
            "FPR": fpr, "AUC": float(roc_auc_score(yt, pred)),
            "PR_AUC": float(average_precision_score(yt, pred)),
            "PA_F1": float(pf1), "n": int(len(yt)),
            "positives": int(yt.sum()),
            "fit_s": np.nan, "score_s": np.nan,
        })
    return rows


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="DFA-COPOD 优化版")
    p.add_argument("--csv", default=str(ROOT / "08_SWaT/SWaT_all_1s.csv"))
    p.add_argument("--extractors", nargs="*",
                   default=["AE", "TCN", "iTransformer"])
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-csv", default=str(OUT_CSV))
    p.add_argument("--out-md", default=str(OUT_MD))
    args = p.parse_args()

    device = resolve_device(args.device)
    print(f"[opt] device={device}")
    b = load_swat(args.csv, val_frac=0.20, scaler="standard",
                  verbose=False)
    Xtr, Xv, Xt = b.train_X, b.val_X, b.test_X
    yv, yt = b.val_label.astype(int), b.test_label.astype(int)
    b.train_df = b.val_df = b.test_df = None
    print(f"[opt] train={len(Xtr):,} val={len(Xv):,} test={len(Xt):,}")

    rows = []

    # ---- E1 raw COPOD ----
    z1 = np.load(ROOT / "runs" / "isolation_peers" / "COPOD.npz")
    raw = {"normal": z1["normal"], "val": z1["val"], "test": z1["test"]}
    for protocol in ("12", "60"):
        n = _prep_normal(raw["normal"], protocol, True)
        v, yv2 = _prep_point(raw["val"], yv, protocol)
        t, yt2 = _prep_point(raw["test"], yt, protocol)
        rows += _evaluate("E1", "raw", "COPOD", protocol,
                          n, v, yv2, t, yt2)

    # ---- E2 AE baseline（point scores）----
    ae_model = _load_extractor("AE", device)
    ae_meta = json.loads((ROOT / "runs" / "ae_swat" / "meta.json")
                         .read_text(encoding="utf-8"))
    npm = __import__(
        "models.classical.normal99_project_models", fromlist=["x"])
    npm.N_NORMAL = len(Xtr)
    ae_normal_ps = npm._normal_point_scores(
        ae_model, Xtr, ae_meta, "reconstruction", device)
    ae_v = _point_frame("ae_swat", "val")
    ae_t = _point_frame("ae_swat", "test")
    for protocol in ("12", "60"):
        n = _prep_normal(ae_normal_ps, protocol, True)
        v, yv2 = _prep_point(ae_v["score"].to_numpy(), yv, protocol)
        t, yt2 = _prep_point(ae_t["score"].to_numpy(), yt, protocol)
        rows += _evaluate("E2", "AE", "AE-recon", protocol,
                          n, v, yv2, t, yt2)
    del ae_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- 深度特征 ----
    deep_scores = {}
    for kind in args.extractors:
        print(f"\n[opt] {kind}: 特征提取 ...")
        model = _load_extractor(kind, device)
        t0 = time.time()
        Ftr = _extract(model, Xtr, kind, device)
        Fv = _extract(model, Xv, kind, device)
        Ft = _extract(model, Xt, kind, device)
        feat_s = time.time() - t0
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[opt] {kind}: features {Ftr.shape} in {feat_s:.1f}s")

        # 原始隐特征 COPOD
        copod, fit_s = _fit_copod(Ftr)
        t0 = time.time()
        sn = _copod_scores(copod, Ftr)
        sv = _copod_scores(copod, Fv)
        st = _copod_scores(copod, Ft)
        score_s = time.time() - t0
        exp = {"AE": "E3", "TCN": "E4",
               "iTransformer": "E5"}[kind]
        for protocol in ("12", "60"):
            n = _prep_normal(sn, protocol, False)
            v, yv2 = _prep_window(sv, yv, protocol)
            t, yt2 = _prep_window(st, yt, protocol)
            rows += _evaluate(exp, kind, "COPOD-raw-latent", protocol,
                              n, v, yv2, t, yt2, fit_s, score_s)
        deep_scores[(kind, "raw")] = {
            "normal": sn, "val": sv, "test": st}

        # PCA 白化
        pca_idx = np.random.RandomState(42).choice(
            len(Ftr), min(FIT_N, len(Ftr)), replace=False)
        pca = PCA(n_components=min(PCA_DIM, Ftr.shape[1]),
                  whiten=True, random_state=42).fit(Ftr[pca_idx])
        Ftw = pca.transform(Ftr)
        Fvw = pca.transform(Fv)
        Ftw2 = pca.transform(Ft)
        copod_w, fit_ws = _fit_copod(Ftw)
        t0 = time.time()
        snw = _copod_scores(copod_w, Ftw)
        svw = _copod_scores(copod_w, Fvw)
        stw = _copod_scores(copod_w, Ftw2)
        score_ws = time.time() - t0
        expw = {"AE": "E3w", "TCN": "E4w",
                "iTransformer": "E5w"}[kind]
        for protocol in ("12", "60"):
            n = _prep_normal(snw, protocol, False)
            v, yv2 = _prep_window(svw, yv, protocol)
            t, yt2 = _prep_window(stw, yt, protocol)
            rows += _evaluate(expw, kind, "COPOD-PCA-whiten", protocol,
                              n, v, yv2, t, yt2, fit_ws, score_ws)
        deep_scores[(kind, "whiten")] = {
            "normal": snw, "val": svw, "test": stw}

        # E7 特征级融合：原始末点 + 白化隐特征
        Xlast = {"normal": Xtr[L - 1:], "val": Xv[L - 1:],
                 "test": Xt[L - 1:]}
        Fcat = {
            "normal": np.concatenate(
                [Xlast["normal"], Ftw], axis=1),
            "val": np.concatenate([Xlast["val"], Fvw], axis=1),
            "test": np.concatenate([Xlast["test"], Ftw2], axis=1),
        }
        copod_c, fit_cs = _fit_copod(Fcat["normal"])
        t0 = time.time()
        snc = _copod_scores(copod_c, Fcat["normal"])
        svc = _copod_scores(copod_c, Fcat["val"])
        stc = _copod_scores(copod_c, Fcat["test"])
        score_cs = time.time() - t0
        for protocol in ("12", "60"):
            n = _prep_normal(snc, protocol, False)
            v, yv2 = _prep_window(svc, yv, protocol)
            t, yt2 = _prep_window(stc, yt, protocol)
            rows += _evaluate("E7", kind, "COPOD-feature-concat",
                              protocol, n, v, yv2, t, yt2,
                              fit_cs, score_cs)

        # E6 / E8（原始 + 深度 COPOD）
        for variant in ("raw", "whiten"):
            deep = deep_scores[(kind, variant)]
            for protocol in ("12", "60"):
                rows += _fuse_rows("E6", f"{kind}-{variant}",
                                   raw, deep, yv, yt, protocol)
                rows += _cascade_rows("E8", f"{kind}-{variant}",
                                      raw, deep, yv, yt, protocol)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    # ---------------- Markdown ----------------
    lines = [
        "# DFA-COPOD 优化版（SWaT）", "",
        "- E1/E2：原始 COPOD / AE 基线",
        "- E3–E5：AE/TCN/iTransformer 隐特征 + COPOD",
        "- E3w–E5w：PCA 白化后的隐特征 + COPOD",
        "- E6：原始 COPOD + 深度 COPOD 分数融合",
        "- E7：原始末点特征 + 白化隐特征拼接 + COPOD",
        "- E8：灰区级联（COPOD 高置信直接报警，灰区深度 COPOD 确认）",
        "- 12-step 与 60-step 两套评估；Normal99/ValBestF1/ValFPR1/Oracle",
        f"- PCA 维度={PCA_DIM}，COPOD 拟合子样本={FIT_N:,}", "",
    ]
    for protocol in ("12", "60"):
        lines += [f"## {protocol}-step 主结果", "",
                  "| exp | model | Normal99 F1/PA | ValFPR1 F1/PA | "
                  "ValBestF1 F1/PA | AUC |",
                  "|---|---|---|---|---|---|"]
        for exp in ["E1", "E2", "E3", "E3w", "E4", "E4w",
                    "E5", "E5w", "E7"]:
            sub = out[(out.protocol == protocol)
                      & (out.experiment == exp)]
            if sub.empty:
                continue
            label = {
                "E1": "COPOD(raw)", "E2": "AE",
                "E3": "AE+COPOD", "E3w": "AE+PCA-COPOD",
                "E4": "TCN+COPOD", "E4w": "TCN+PCA-COPOD",
                "E5": "iTransformer+COPOD",
                "E5w": "iTransformer+PCA-COPOD",
                "E7": "raw-last + PCA latent",
            }[exp]
            vals = {}
            for s in ("Normal99", "ValFPR1", "ValBestF1"):
                r = sub[sub.threshold_source == s]
                vals[s] = f"{r['F1'].iloc[0]:.4f}/{r['PA_F1'].iloc[0]:.4f}"
            auc = sub["AUC"].mean()
            lines.append(
                f"| {exp} | {label} | {vals['Normal99']} | "
                f"{vals['ValFPR1']} | {vals['ValBestF1']} | "
                f"{auc:.4f} |")
        lines.append("")

        # E6 best per extractor/source
        lines += [f"### {protocol}-step E6 最优融合", "",
                  "| source | extractor-variant | method | P | R | F1 | "
                  "PA-F1 | AUC |", "|---|---|---|---|---|---|---|---|"]
        e6 = out[(out.protocol == protocol) & (out.experiment == "E6")]
        for s in ("Normal99", "ValFPR1", "ValBestF1"):
            for ev in e6.extractor.unique():
                x = e6[(e6.threshold_source == s)
                       & (e6.extractor == ev)]
                if x.empty:
                    continue
                b = x.loc[x.F1.idxmax()]
                lines.append(
                    f"| {s} | {ev} | {b['method']} | "
                    f"{b['Precision']:.4f} | {b['Recall']:.4f} | "
                    f"{b['F1']:.4f} | {b['PA_F1']:.4f} | "
                    f"{b['AUC']:.4f} |")
        lines.append("")

        # E8
        lines += [f"### {protocol}-step E8 灰区级联", "",
                  "| extractor-variant | source | P | R | F1 | PA-F1 | AUC |",
                  "|---|---|---|---|---|---|---|"]
        e8 = out[(out.protocol == protocol) & (out.experiment == "E8")]
        for _, r in e8.iterrows():
            lines.append(
                f"| {r['extractor']} | {r['threshold_source']} | "
                f"{r['Precision']:.4f} | {r['Recall']:.4f} | "
                f"{r['F1']:.4f} | {r['PA_F1']:.4f} | {r['AUC']:.4f} |")
        lines.append("")

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已保存: {args.out_csv}")
    print(f"已保存: {args.out_md}")

    # console summary
    for protocol in ("12", "60"):
        print(f"\n===== {protocol}-step ValFPR1 top =====")
        v = out[(out.protocol == protocol)
                & (out.threshold_source == "ValFPR1")].sort_values(
            "F1", ascending=False).head(15)
        print(v[["experiment", "extractor", "method", "F1",
                 "PA_F1", "AUC"]].to_string(
                     index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
