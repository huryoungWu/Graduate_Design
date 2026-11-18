# -*- coding: utf-8 -*-
"""Fit weighted four/five-branch fusion on validation; evaluate on test."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from repro_anomaly_transformer.robust_dfa_fusion import (  # noqa: E402
    _metrics_for, _tail_z,
)

SRC = ROOT / "runs" / "four_branch_scores"
RC = ROOT / "runs" / "recon_comparison"
OUT_CSV = RC / "FOUR_BRANCH_FUSION.csv"
OUT_MD = RC / "FOUR_BRANCH_FUSION.md"
ORDER = ["batadal", "leakdb_hanoi500", "leakdb_hanoi1000",
         "leakdb_net1", "psm", "sml_machine-1-1",
         "sml_machine-1-2", "sml_machine-2-1", "swat", "wadi"]
BRANCHES = ["raw", "tcn_whiten", "tcn_err", "ae_err", "usad_err"]
SETS = {
    "3B-err": ["raw", "tcn_whiten", "tcn_err"],
    "3B-err-raw": ["raw", "tcn_raw", "tcn_err"],
    "3B-usad": ["raw", "tcn_whiten", "usad_err"],
    "3B-ae": ["raw", "tcn_whiten", "ae_err"],
    "4B-usad": ["raw", "tcn_whiten", "tcn_err", "usad_err"],
    "4B-usad-raw": ["raw", "tcn_raw", "tcn_err", "usad_err"],
    "4B-ae": ["raw", "tcn_whiten", "tcn_err", "ae_err"],
    "5B": BRANCHES,
}


def load(ds):
    z = np.load(SRC / f"{ds}.npz")
    out = {}
    keys = BRANCHES + ["tcn_raw"]
    T = min(len(z[f"{b}_n"]) for b in keys)
    for b in keys:
        n = z[f"{b}_n"].astype(np.float64)
        out[b] = (_tail_z(n[-T:], n[-T:]),
                  _tail_z(z[f"{b}_v"], n[-T:]),
                  _tail_z(z[f"{b}_t"], n[-T:]))
    out["yv"] = z["y_val"].astype(int)
    out["yt"] = z["y_test"].astype(int)
    return out


def comps(k, step=0.1):
    n = int(round(1 / step))
    for c in itertools.product(range(n + 1), repeat=k):
        if sum(c) == n:
            yield tuple(x / n for x in c)


def fuse(data, keys, w):
    return sum(wi * data[k][i] for k, wi in zip(keys, w) for i in (0,))  # unused


def vec(data, keys, w, part):
    idx = {"n": 0, "v": 1, "t": 2}[part]
    return sum(wi * data[k][idx] for k, wi in zip(keys, w))


def val_auc(data, keys, w):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(data["yv"], vec(data, keys, w, "v")))


def evaluate(data, keys, w):
    fn = vec(data, keys, w, "n")
    fv = vec(data, keys, w, "v")
    ft = vec(data, keys, w, "t")
    return _metrics_for(data["yv"], fv, data["yt"], ft, fn)


def fixed_baselines(data):
    out = {}
    out["BASE-raw"] = evaluate(data, ["raw", "tcn_raw"], (0.5, 0.5))
    out["BASE-whiten"] = evaluate(data, ["raw", "tcn_whiten"], (0.5, 0.5))  # max below
    # max fusion for BASE-whiten
    fn = np.maximum(data["raw"][0], data["tcn_whiten"][0])
    fv = np.maximum(data["raw"][1], data["tcn_whiten"][1])
    ft = np.maximum(data["raw"][2], data["tcn_whiten"][2])
    out["BASE-whiten"] = _metrics_for(data["yv"], fv, data["yt"], ft, fn)
    return out


def cascade(data, use_recon=True):
    rz_n, rz_v, rz_t = data["raw"]
    ez_n, ez_v, ez_t = data["tcn_err"]
    cz_n, cz_v, cz_t = (data["usad_err"] if use_recon else data["ae_err"])
    hi = float(np.quantile(rz_n, 0.995))
    lo = float(np.quantile(rz_n, 0.99))
    e_thr = float(np.quantile(ez_n, 0.99))
    c_thr = float(np.quantile(cz_n, 0.99))
    def rule(r, e, c):
        return (r > hi) | ((r > lo) & ((e > e_thr) | (c > c_thr)))
    pred_v = rule(rz_v, ez_v, cz_v)
    pred_t = rule(rz_t, ez_t, cz_t)
    # metrics from predictions
    from sklearn.metrics import (average_precision_score,
                                 precision_recall_fscore_support,
                                 roc_auc_score)
    from models.reconstruction.usad_extended_metrics import point_adjust
    def met(y, p):
        pr, rc, f1, _ = precision_recall_fscore_support(
            y, p.astype(int), average="binary", zero_division=0)
        neg = y == 0
        fpr = float((p[neg].sum()) / max(neg.sum(), 1))
        pa = point_adjust(y.astype(int), p.astype(int))
        _, _, paf1, _ = precision_recall_fscore_support(
            y, pa, average="binary", zero_division=0)
        return {"F1": float(f1), "P": float(pr), "R": float(rc),
                "PA_F1": float(paf1), "FPR": fpr}
    vm, tm = met(data["yv"], pred_v), met(data["yt"], pred_t)
    return {
        "Normal99_Precision": tm["P"], "Normal99_Recall": tm["R"],
        "Normal99_F1": tm["F1"], "Normal99_PA_F1": tm["PA_F1"],
        "Normal99_FPR": tm["FPR"], "Normal99_AUC": float("nan"),
        "ValBestF1_F1": vm["F1"], "ValBestF1_PA_F1": vm["PA_F1"],
    }


def main():
    datasets = [ds for ds in ORDER if (SRC / f"{ds}.npz").exists()]
    data = {ds: load(ds) for ds in datasets}
    print("datasets:", datasets)

    rows = []
    # fixed baselines + equal-weight sets
    for ds in datasets:
        d = data[ds]
        for name, met in fixed_baselines(d).items():
            rows.append({"dataset": ds, "method": name, **met})
        for name, keys in SETS.items():
            w = tuple(1 / len(keys) for _ in keys)
            rows.append({"dataset": ds, "method": f"{name}-equal",
                         **evaluate(d, keys, w)})
        rows.append({"dataset": ds, "method": "cascade",
                     **cascade(d, use_recon=True)})

    # per-dataset weights selected on validation AUC (coarse grid, fast)
    for ds in datasets:
        d = data[ds]
        for name, keys in SETS.items():
            w = max(comps(len(keys), 0.2),
                    key=lambda ww: val_auc(d, keys, ww))
            met = evaluate(d, keys, w)
            rows.append({"dataset": ds, "method": f"{name}-valW",
                         "weights": str(w), **met})
        print(f"  valW {ds} done", flush=True)

    # global weights selected on mean validation AUC
    for name, keys in SETS.items():
        w = max(comps(len(keys), 0.2),
                key=lambda ww: float(np.mean(
                    [val_auc(data[ds], keys, ww) for ds in datasets])))
        for ds in datasets:
            met = evaluate(data[ds], keys, w)
            rows.append({"dataset": ds, "method": f"{name}-globalW",
                         "weights": str(w), **met})
        print(f"  globalW {name} {w}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(out[["dataset", "method", "Normal99_Precision",
               "Normal99_Recall", "Normal99_F1", "Normal99_PA_F1",
               "Normal99_AUC"]].to_string(index=False))

    # markdown
    lines = ["# Four/five-branch robust fusion", "",
             "- Branches: raw COPOD, TCN-whiten latent COPOD, "
             "TCN forecast error, AE/USAD reconstruction error.",
             "- Scores: train-normal ECDF -> -log(1-p) -> z -> ±8.",
             "- Weights selected on VALIDATION AUC; test labels not used.",
             "- `-valW` = per-dataset weights; `-globalW` = one global "
             "weight vector selected on mean validation AUC.",
             "- `cascade` = raw > 99.5% OR gray zone (99-99.5%) confirmed "
             "by forecast/reconstruction > 99%.", "",
             "| dataset | method | P | R | F1 | PA-F1 | AUC | weights |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        lines.append(
            f"| {r.dataset} | {r.method} | {r.Normal99_Precision:.4f} | "
            f"{r.Normal99_Recall:.4f} | {r.Normal99_F1:.4f} | "
            f"{r.Normal99_PA_F1:.4f} | {r.Normal99_AUC:.4f} | "
            f"{r.get('weights', '')} |")
    lines += ["", "## Mean over datasets", "",
              "| method | mean P | mean R | mean F1 | mean PA-F1 | "
              "mean AUC |", "|---|---|---|---|---|---|"]
    for m, sub in out.groupby("method", sort=False):
        lines.append(
            f"| {m} | {sub.Normal99_Precision.mean():.4f} | "
            f"{sub.Normal99_Recall.mean():.4f} | "
            f"{sub.Normal99_F1.mean():.4f} | "
            f"{sub.Normal99_PA_F1.mean():.4f} | "
            f"{sub.Normal99_AUC.mean():.4f} |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT_CSV)
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
