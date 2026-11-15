# -*- coding: utf-8 -*-
"""所有模型的四种阈值来源对比。

- Normal99：Normal 分数的 99% 分位
- ValBestF1：Attack 前 20% 验证集按 F1 选阈值
- ValFPR1：验证集正常点分数的 99% 分位（目标 FPR=1%）
- Oracle：测试集标签选阈值（仅上限）

测试集：Attack 后 80%，冻结。12 步窗口（SBT 为 60 步）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.common.protocol import select_threshold

from .benchmark_all_models import (MODELS, _metrics, _point_frame,
                                   _window12)

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "runs" / "recon_comparison" / "THRESHOLD_COMPARISON_ALL.csv"
OUT_MD = ROOT / "runs" / "recon_comparison" / "THRESHOLD_COMPARISON_ALL.md"


def main():
    val_data, test_frames = {}, {}
    for name, artifact, family in MODELS:
        try:
            v, _ = _window12(_point_frame(artifact, "val"))
            t, _ = _window12(_point_frame(artifact, "test"))
            val_data[name] = (v["score"].to_numpy(float),
                              v["label"].to_numpy(int))
            test_frames[name] = t
        except Exception as exc:
            print(f"[skip] {name}: {exc}")

    # AT paper 12-step
    at_dir = ROOT / "repro_anomaly_transformer/runs/swat_20_80"
    te = np.load(at_dir / "test_energy.npy")
    va = np.load(at_dir / "val_energy.npy")
    ts = pd.read_csv(at_dir / "test_scores.csv")
    yp = ts["label"].to_numpy(np.int8)[:len(te)]
    data_y = np.load(
        ROOT / "repro_anomaly_transformer/data/swat/test_y.npy")
    n_val = int(len(data_y) * 0.2)
    yv = data_y[:len(va)].astype(int)
    w = 12
    va_w = np.lib.stride_tricks.sliding_window_view(va, w).mean(axis=1)
    yv_w = np.lib.stride_tricks.sliding_window_view(yv, w).max(axis=1)
    te_w = np.lib.stride_tricks.sliding_window_view(te, w).mean(axis=1)
    yt_w = np.lib.stride_tricks.sliding_window_view(yp, w).max(axis=1)
    tsw = pd.to_datetime(ts["timestamp"])[:len(te)][w - 1:]
    nv = min(len(va_w), len(yv_w))
    nt = min(len(te_w), len(yt_w), len(tsw))
    val_data["AnomalyTransformer(paper)"] = (va_w[:nv], yv_w[:nv])
    test_frames["AnomalyTransformer(paper)"] = pd.DataFrame(
        {"score": te_w[:nt], "label": yt_w[:nt]}, index=tsw[:nt])
    tr = np.load(ROOT / "repro_anomaly_transformer/runs/swat/train_energy.npy")
    tr_w = np.lib.stride_tricks.sliding_window_view(tr, w).mean(axis=1)
    n99_at = float(np.percentile(tr_w, 99))

    # SBT 60-step
    sbt_dir = ROOT / "repro_anomaly_transformer/runs/sbt_20_80"
    sbt = pd.read_csv(sbt_dir / "test_scores.csv")
    test_frames["SBT"] = pd.DataFrame(
        {"score": sbt["score"].to_numpy(float),
         "label": sbt["label_window"].to_numpy(int)},
        index=pd.to_datetime(sbt["timestamp"]))
    n99_sbt = 0.00120511

    # common 12-step test index
    names12 = [n for n in test_frames if n != "SBT"]
    common = test_frames[names12[0]].index
    for n in names12[1:]:
        common = common.intersection(test_frames[n].index)
    common = pd.DatetimeIndex(sorted(common))
    y_common = test_frames[names12[0]].loc[common, "label"].to_numpy(int)

    # Normal-99 thresholds
    proj = pd.read_csv(ROOT / "runs/normal99_project_models/comparison.csv")
    proj = proj[proj.method == "q99_all"].set_index("model")[
        "threshold"].to_dict()
    full = pd.read_csv(ROOT / "runs/normal99_full_attack/comparison.csv")
    full = full[(full.dataset == "SWaT") & (full.method == "q99_all")]
    full = full.set_index("model")["threshold"].to_dict()
    n99 = {}
    for name, _, _ in MODELS:
        if name == "AnomalyTransformer(project)":
            n99[name] = float(proj["AnomalyTransformer"])
        elif name in proj:
            n99[name] = float(proj[name])
        elif name in full:
            n99[name] = float(full[name])
    n99["AnomalyTransformer(paper)"] = n99_at
    n99["SBT"] = n99_sbt

    # benchmark CSV: ValBestF1 / Oracle
    bench = pd.read_csv(
        ROOT / "runs/recon_comparison/ALL_MODELS_BENCHMARK.csv")
    bench = bench.set_index("model")

    rows = []
    for name in test_frames:
        if name == "SBT":
            te = test_frames[name]
            yte = te["label"].to_numpy(int)
            s_te = te["score"].to_numpy(float)
            n = len(yte)
        else:
            te = test_frames[name].loc[common]
            yte = te["label"].to_numpy(int)
            s_te = te["score"].to_numpy(float)
            n = len(yte)

        # Normal99
        if name in n99:
            m = _metrics(yte, s_te, n99[name])[0]
            rows.append({"model": name, "method": "Normal99",
                         "threshold": n99[name], **m,
                         "n": n, "positives": int(yte.sum())})

        # ValBestF1 / Oracle from benchmark
        if name in bench.index:
            b = bench.loc[name]
            rows.append({
                "model": name, "method": "ValBestF1",
                "threshold": b["frozen_threshold"],
                "Precision": b["frozen_Precision"],
                "Recall": b["frozen_Recall"], "F1": b["frozen_F1"],
                "FPR": b["frozen_FPR"], "AUC": b["frozen_AUC"],
                "PR_AUC": b["frozen_PR_AUC"],
                "PA_F1": b["frozen_PA_F1"],
                "n": int(b["n"]), "positives": int(b["positives"]),
            })
            rows.append({
                "model": name, "method": "Oracle",
                "threshold": b["oracle_threshold"],
                "Precision": b["oracle_Precision"],
                "Recall": b["oracle_Recall"], "F1": b["oracle_F1"],
                "FPR": b["oracle_FPR"], "AUC": b["frozen_AUC"],
                "PR_AUC": b["frozen_PR_AUC"],
                "PA_F1": b["oracle_PA_F1"],
                "n": int(b["n"]), "positives": int(b["positives"]),
            })

        # ValFPR1: 验证集正常分数的 99% 分位
        if name in val_data:
            vs, vy = val_data[name]
            vn = vs[vy == 0]
            if len(vn):
                thr = float(np.percentile(vn, 99))
                m = _metrics(yte, s_te, thr)[0]
                rows.append({"model": name, "method": "ValFPR1",
                             "threshold": thr, **m,
                             "n": n, "positives": int(yte.sum())})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    cols = ["model", "method", "threshold", "Precision", "Recall",
            "F1", "FPR", "AUC", "PR_AUC", "PA_F1", "n", "positives"]
    lines = ["# 所有模型的四种阈值来源对比（SWaT）", "",
             "- 训练：Normal only；测试：Attack 后 80% 冻结",
             "- Normal99：Normal 分数 99% 分位",
             "- ValBestF1：Attack 前 20% 验证集选 F1 最优阈值",
             "- ValFPR1：验证集正常分数 99% 分位（目标 FPR=1%）",
             "- Oracle：测试集选阈值，仅作上限", ""]
    for method in ["Normal99", "ValBestF1", "ValFPR1", "Oracle"]:
        sub = out[out.method == method]
        if not len(sub):
            continue
        lines += [f"## {method}", ""]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, x in sub.iterrows():
            vals = []
            for c in cols:
                v = x[c]
                if isinstance(v, (float, np.floating)):
                    vals.append("nan" if np.isnan(v) else f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(out[cols].to_string(index=False,
                              float_format=lambda x: f"{x:.4f}"))
    print(f"已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")


if __name__ == "__main__":
    main()
