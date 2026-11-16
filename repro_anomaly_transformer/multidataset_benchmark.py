# -*- coding: utf-8 -*-
"""跨数据集异常检测基准：经典检测器 + 统一协议。

数据集缓存由 ``models.common.multidataset`` 生成。
模型：COPOD / ECOD / IF / HBOS / LOF / OCSVM / HST / RRCF。
协议：Normal only 训练；val = 测试段的前 20%（按正例段切分）；
      12 步窗口；Normal99 / ValBestF1 / ValFPR1。

额外支持：
- ``--seeds``：稳定性（同一模型不同随机种子）；
- ``--noise``：测试集高斯噪声鲁棒性；
- ``--missing``：测试集随机缺失鲁棒性；
- 输出 fit_s / score_s / train_n，用于轻量性对比。

输出：runs/recon_comparison/MULTIDATASET.csv / .md
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from models.common.config import resolve_device
from models.common.protocol import select_threshold
from models.classical.isolation_peers import _BatchCutForest, _window, \
    _window_labels

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "multidataset"
RC = ROOT / "runs" / "recon_comparison"
OUT_CSV = RC / "MULTIDATASET.csv"
OUT_MD = RC / "MULTIDATASET.md"

WINDOW = 12
SOURCES = ["Normal99", "ValBestF1", "ValFPR1"]

DATASETS = [
    "psm", "sml_machine-1-1", "sml_machine-1-2", "sml_machine-2-1",
    "batadal", "battledim_2018", "battledim_2019",
    "smap_A-1", "smap_A-2", "smap_P-1", "smap_S-1",
    "leakdb_hanoi500", "leakdb_hanoi1000", "leakdb_net1",
    "swat", "wadi",
]

MODELS = ["COPOD", "ECOD", "IF", "HBOS", "LOF", "OCSVM", "HST", "RRCF"]


def _load(name: str):
    d = CACHE / name
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return (np.load(d / "train_X.npy"), np.load(d / "val_X.npy"),
            np.load(d / "val_y.npy"), np.load(d / "test_X.npy"),
            np.load(d / "test_y.npy"), meta)


def _subsample(X, n, seed=42):
    if n is None or n >= len(X):
        return X, len(X)
    idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
    return X[idx], n


def _fit_score(name: str, Xtr, Xv, Xt, seed: int, device):
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    from pyod.models.hbos import HBOS
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from sklearn.ensemble import IsolationForest

    if name == "COPOD":
        m, nfit = COPOD(contamination=0.01, n_jobs=1), len(Xtr)
    elif name == "ECOD":
        m, nfit = ECOD(contamination=0.01, n_jobs=1), len(Xtr)
    elif name == "HBOS":
        m, nfit = HBOS(n_bins=10, contamination=0.01), len(Xtr)
    elif name == "IF":
        m = IsolationForest(n_estimators=100, max_samples="auto",
                            contamination="auto", random_state=seed,
                            n_jobs=-1)
        nfit = len(Xtr)
    elif name == "LOF":
        m = LOF(n_neighbors=20, novelty=True, contamination=0.01,
                n_jobs=-1)
        Xtr, nfit = _subsample(Xtr, 20000, seed)
    elif name == "OCSVM":
        m = OCSVM(kernel="rbf", nu=0.01, gamma="scale",
                  cache_size=1000, contamination=0.01)
        Xtr, nfit = _subsample(Xtr, 10000, seed)
    elif name == "HST":
        m = _BatchCutForest(n_trees=100, sample_size=256, seed=seed,
                            score_kind="exp_depth")
        nfit = len(Xtr)
    elif name == "RRCF":
        m = _BatchCutForest(n_trees=100, sample_size=256, seed=seed,
                            score_kind="depth")
        nfit = len(Xtr)
    elif name == "DIF":
        from pyod.models.dif import DIF
        m = DIF(n_ensemble=5, n_estimators=6, max_samples=256,
                representation_dim=20, hidden_neurons=[64, 32],
                contamination=0.01, random_state=seed,
                device=str(device))
        Xtr, nfit = _subsample(Xtr, 20000, seed)
    elif name == "DeepSVDD":
        from pyod.models.deep_svdd import DeepSVDD
        n_features = Xtr.shape[1]
        m = DeepSVDD(n_features=n_features, use_ae=False,
                     hidden_neurons=[64, 32], epochs=50,
                     batch_size=128, preprocessing=False,
                     contamination=0.01, verbose=0,
                     random_state=seed)
        Xtr, nfit = _subsample(Xtr, 20000, seed)
    else:
        raise ValueError(name)

    # Normal99 必须用连续的 Normal 尾段，不能随机打乱点序
    n_norm = min(100000, len(Xtr))
    X_norm = Xtr[-n_norm:]

    t0 = time.time()
    if name in ("HST", "RRCF"):
        m.fit(Xtr)
        fit_s = time.time() - t0
        t0 = time.time()
        sn = m.score_samples(X_norm)
        sv = m.score_samples(Xv)
        st = m.score_samples(Xt)
        score_s = time.time() - t0
    elif name == "IF":
        m.fit(Xtr)
        fit_s = time.time() - t0
        t0 = time.time()
        sn = -m.score_samples(X_norm)
        sv = -m.score_samples(Xv)
        st = -m.score_samples(Xt)
        score_s = time.time() - t0
    else:
        m.fit(Xtr)
        fit_s = time.time() - t0
        t0 = time.time()
        sn = m.decision_function(X_norm)
        sv = m.decision_function(Xv)
        st = m.decision_function(Xt)
        score_s = time.time() - t0
    return (np.asarray(sn).ravel(), np.asarray(sv).ravel(),
            np.asarray(st).ravel(), fit_s, score_s, nfit)


def _metrics(y, s, thr):
    from repro_anomaly_transformer.benchmark_all_models import _metrics
    return _metrics(y, s, thr)[0]


def _eval(dataset, model, seed, sv, st, yv, yt, fit_s, score_s, train_n,
          perturb=""):
    vw, vy = _window(sv), _window_labels(yv)
    tw, ty = _window(st), _window_labels(yt)
    nv = min(len(vw), len(vy))
    nt = min(len(tw), len(ty))
    vw, vy = vw[:nv], vy[:nv]
    tw, ty = tw[:nt], ty[:nt]
    n = nt
    rows = []
    for source in SOURCES:
        if source == "Normal99":
            # 无 Normal 分数时用验证集正常点近似；这里由调用方补
            continue
        if source == "ValBestF1":
            thr, _ = select_threshold(vw, vy)
            thr = float(thr)
        else:
            nv = vw[vy == 0]
            thr = float(np.percentile(nv if len(nv) else vw, 99))
        met = _metrics(ty, tw, thr)
        rows.append({
            "dataset": dataset, "model": model, "seed": seed,
            "perturb": perturb, "threshold_source": source,
            "threshold": thr, "Precision": met["Precision"],
            "Recall": met["Recall"], "F1": met["F1"],
            "FPR": met["FPR"], "AUC": met["AUC"],
            "PR_AUC": met["PR_AUC"], "PA_F1": met["PA_F1"],
            "n": int(n), "positives": int(ty.sum()),
            "fit_s": float(fit_s), "score_s": float(score_s),
            "train_n": int(train_n),
        })
    return rows


def _normal99_row(dataset, model, seed, sn, st, yt, fit_s, score_s,
                  train_n, perturb=""):
    nw = _window(sn)
    tw, ty = _window(st), _window_labels(yt)
    n = min(len(tw), len(ty))
    tw, ty = tw[:n], ty[:n]
    thr = float(np.percentile(nw, 99))
    met = _metrics(ty, tw, thr)
    return {
        "dataset": dataset, "model": model, "seed": seed,
        "perturb": perturb, "threshold_source": "Normal99",
        "threshold": thr, "Precision": met["Precision"],
        "Recall": met["Recall"], "F1": met["F1"], "FPR": met["FPR"],
        "AUC": met["AUC"], "PR_AUC": met["PR_AUC"],
        "PA_F1": met["PA_F1"], "n": int(n),
        "positives": int(ty.sum()), "fit_s": float(fit_s),
        "score_s": float(score_s), "train_n": int(train_n),
    }


def _perturb(X, kind, level, seed=42):
    rng = np.random.RandomState(seed)
    if kind == "noise":
        return X + rng.normal(0, level, size=X.shape).astype(np.float32)
    if kind == "missing":
        mask = rng.rand(*X.shape) < level
        out = X.copy()
        out[mask] = 0.0
        return out
    return X


def main() -> None:
    p = argparse.ArgumentParser(description="跨数据集经典检测器基准")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=[42])
    p.add_argument("--noise", nargs="*", type=float, default=[])
    p.add_argument("--missing", nargs="*", type=float, default=[])
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-csv", default=str(OUT_CSV))
    p.add_argument("--out-md", default=str(OUT_MD))
    args = p.parse_args()

    device = resolve_device(args.device)
    datasets = args.datasets or DATASETS
    models = args.models or MODELS
    rows = []
    print(f"[multi] device={device} datasets={len(datasets)} "
          f"models={len(models)} seeds={args.seeds}")

    for ds in datasets:
        if not (CACHE / ds).exists():
            print(f"[skip] {ds}: cache missing")
            continue
        Xtr, Xv, yv, Xt, yt, meta = _load(ds)
        print(f"\n[multi] {ds}: train={len(Xtr):,} val={len(Xv):,} "
              f"test={len(Xt):,} features={Xtr.shape[1]}", flush=True)
        for seed in args.seeds:
            for model in models:
                try:
                    sn, sv, st, fit_s, score_s, nfit = _fit_score(
                        model, Xtr, Xv, Xt, seed, device)
                    rows += _eval(ds, model, seed, sv, st, yv, yt,
                                  fit_s, score_s, nfit)
                    rows.append(_normal99_row(
                        ds, model, seed, sn, st, yt,
                        fit_s, score_s, nfit))
                    print(f"  {model:6s} seed={seed} "
                          f"fit={fit_s:.2f}s score={score_s:.2f}s",
                          flush=True)
                    # 鲁棒性：测试集扰动
                    for kind, levels in (("noise", args.noise),
                                         ("missing", args.missing)):
                        for lv in levels:
                            Xp = _perturb(Xt, kind, lv, seed)
                            _, svp, stp, f2, s2, n2 = _fit_score(
                                model, Xtr, Xv, Xp, seed, device)
                            rows += _eval(
                                ds, model, seed, svp, stp, yv, yt,
                                f2, s2, n2, perturb=f"{kind}={lv}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[fail] {ds} {model} seed={seed}: {exc}")

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    # ---------------- Markdown ----------------
    lines = [
        "# 跨数据集经典检测器基准", "",
        "- 数据集：PSM / SML×3 / BATADAL / BattLeDIM×2 / SMAP-MS L×4 / "
        "LeakDB×3 / SWaT / WADI",
        "- 协议：Normal only；验证=测试段前 20%（按正例段切分）；"
        "12 步窗口；Normal99 / ValBestF1 / ValFPR1",
        "- 指标：P/R/F1/FPR/AUC/PR-AUC/PA-F1；fit_s/score_s 为轻量性指标", "",
    ]
    for source in SOURCES:
        sub = out[out.threshold_source == source]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="dataset", columns="model",
                              values="F1", aggfunc="mean")
        lines += [f"## {source} F1", "",
                  "| dataset | " + " | ".join(piv.columns) + " |",
                  "|" + "|".join(["---"] * (len(piv.columns) + 1)) + "|"]
        for ds, r in piv.iterrows():
            vals = [("nan" if pd.isna(v) else f"{v:.4f}")
                    for v in r.values]
            lines.append(f"| {ds} | " + " | ".join(vals) + " |")
        lines.append("")

    # 模型平均/稳定性
    v = out[out.threshold_source == "ValFPR1"]
    agg = v.groupby("model").agg(
        mean_F1=("F1", "mean"), std_F1=("F1", "std"),
        mean_PA=("PA_F1", "mean"), mean_AUC=("AUC", "mean"),
        mean_fit=("fit_s", "mean"), mean_score=("score_s", "mean"),
        datasets=("dataset", "nunique")).sort_values("mean_F1",
                                                     ascending=False)
    lines += ["## 模型平均（ValFPR1）", "",
              "| model | mean F1 | std F1 | mean PA-F1 | mean AUC | "
              "mean fit_s | mean score_s | datasets |",
              "|---|---|---|---|---|---|---|---|"]
    for m, r in agg.iterrows():
        lines.append(
            f"| {m} | {r.mean_F1:.4f} | {r.std_F1:.4f} | "
            f"{r.mean_PA:.4f} | {r.mean_AUC:.4f} | "
            f"{r.mean_fit:.2f} | {r.mean_score:.2f} | "
            f"{int(r.datasets)} |")
    lines.append("")

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已保存: {args.out_csv}")
    print(f"已保存: {args.out_md}")
    print(agg.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
