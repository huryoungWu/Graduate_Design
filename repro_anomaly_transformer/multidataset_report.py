# -*- coding: utf-8 -*-
"""汇总跨数据集基准：性能、泛化、稳定性、鲁棒性、轻量性。

主指标：
- AUC（阈值无关）用于泛化/鲁棒性主比较；
- ValBestF1（验证集选阈值）用于 F1/PA 比较；
- ValFPR1 作为部署阈值参考（部分数据集会因阈值过严出现 0 报警）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "runs" / "recon_comparison"

CORE = RC / "MULTIDATASET.csv"
STAB = RC / "MULTIDATASET_STABILITY.csv"
ROB = RC / "MULTIDATASET_ROBUST.csv"
OUT_MD = RC / "MULTIDATASET_REPORT.md"
OUT_CSV = RC / "MULTIDATASET_SUMMARY.csv"

SIZE_NOTE = {
    "COPOD": "O(d) ECDF 表",
    "ECOD": "O(d) ECDF 表",
    "IF": "100 棵孤立树",
    "HBOS": "d×10 bins 直方图",
    "LOF": "20k 参考点 + KDTree",
    "OCSVM": "≤10k 训练点 + RBF 支持向量",
    "HST": "100 棵半空间树（256 样本/树）",
    "RRCF": "100 棵随机切分树（256 样本/树）",
}
FAMILY = {
    "COPOD": "distribution", "ECOD": "distribution",
    "IF": "isolation", "HST": "isolation", "RRCF": "isolation",
    "HBOS": "density", "LOF": "density",
    "OCSVM": "boundary",
}


def _load(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _perf(core: pd.DataFrame) -> pd.DataFrame:
    v = core[core.threshold_source == "ValBestF1"].copy()
    rows = []
    for model, g in v.groupby("model"):
        auc = g.groupby("dataset")["AUC"].mean()
        f1 = g.groupby("dataset")["F1"].mean()
        rank_auc = v.groupby("dataset")["AUC"].rank(
            ascending=False, method="min")
        rank_f1 = v.groupby("dataset")["F1"].rank(
            ascending=False, method="min")
        rr_auc = g.assign(rank=rank_auc.loc[g.index]).groupby(
            "dataset")["rank"].mean()
        rr_f1 = g.assign(rank=rank_f1.loc[g.index]).groupby(
            "dataset")["rank"].mean()
        rows.append({
            "model": model, "family": FAMILY.get(model, ""),
            "mean_AUC": auc.mean(), "std_AUC": auc.std(),
            "min_AUC": auc.min(), "mean_rank_AUC": rr_auc.mean(),
            "wins_AUC": int((rank_auc.loc[g.index] == 1).sum()),
            "mean_F1": f1.mean(), "std_F1": f1.std(),
            "min_F1": f1.min(), "mean_rank_F1": rr_f1.mean(),
            "wins_F1": int((rank_f1.loc[g.index] == 1).sum()),
            "mean_PA": g["PA_F1"].mean(),
            "mean_fit_s": g["fit_s"].mean(),
            "mean_score_s": g["score_s"].mean(),
            "mean_train_n": g["train_n"].mean(),
            "datasets": auc.size,
        })
    out = pd.DataFrame(rows).sort_values("mean_AUC", ascending=False)
    out["size_note"] = out["model"].map(SIZE_NOTE).fillna("")
    return out


def _stab(stab: pd.DataFrame | None) -> pd.DataFrame | None:
    if stab is None:
        return None
    v = stab[stab.threshold_source == "ValBestF1"].copy()
    rows = []
    for model, g in v.groupby("model"):
        f1 = g.groupby(["dataset", "seed"])["F1"].mean().reset_index()
        auc = g.groupby(["dataset", "seed"])["AUC"].mean().reset_index()
        rows.append({
            "model": model,
            "mean_seed_std_F1": f1.groupby("dataset")["F1"].std().mean(),
            "max_seed_std_F1": f1.groupby("dataset")["F1"].std().max(),
            "mean_seed_std_AUC": auc.groupby("dataset")["AUC"].std().mean(),
            "datasets": f1["dataset"].nunique(),
        })
    return pd.DataFrame(rows).sort_values("mean_seed_std_AUC")


def _rob(rob: pd.DataFrame | None) -> pd.DataFrame | None:
    if rob is None:
        return None
    v = rob[rob.threshold_source == "ValBestF1"].copy()
    v["perturb"] = v["perturb"].fillna("")
    clean = v[v.perturb == ""]
    base = clean.groupby(["dataset", "model"])[
        ["F1", "PA_F1", "AUC"]].mean()
    rows = []
    for (ds, model, pert), g in v.groupby(
            ["dataset", "model", "perturb"]):
        if pert == "":
            continue
        b = base.loc[(ds, model)]
        rows.append({
            "dataset": ds, "model": model, "perturb": pert,
            "dF1": g["F1"].mean() - b["F1"],
            "dPA": g["PA_F1"].mean() - b["PA_F1"],
            "dAUC": g["AUC"].mean() - b["AUC"],
            "retention_F1": (g["F1"].mean() / b["F1"]
                             if b["F1"] > 1e-9 else np.nan),
            "retention_AUC": (g["AUC"].mean() / b["AUC"]
                              if b["AUC"] > 1e-9 else np.nan),
        })
    out = pd.DataFrame(rows)
    return out if not out.empty else None


def main() -> None:
    core = _load(CORE)
    if core is None:
        raise SystemExit(f"missing {CORE}")
    perf = _perf(core)
    perf.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    stab = _stab(_load(STAB))
    rob = _rob(_load(ROB))

    lines = [
        "# 跨数据集异常检测对比报告", "",
        "- 数据集：PSM / SML×3 / BATADAL / BattLeDIM×2 / SMAP-MS L×4 / "
        "LeakDB×3 / SWaT / WADI",
        "- 协议：Normal only；验证=测试段前 20%（按正例段切分）；"
        "12 步窗口",
        "- **AUC 为阈值无关主指标**；`ValBestF1` 用于 F1/PA；"
        "`ValFPR1` 仅作部署参考",
        "- BattLeDIM 的 `leak_onset` 是稀疏事件，逐点 F1/AUC 仅作参考，"
        "不适合与其它数据集直接平均",
        "- SMAP/MSL 为多通道、SML 为多机器、LeakDB 为多场景；"
        "报告同时给出跨实体方差", "",
        "## 1. 性能与跨数据集泛化", "",
        "| model | family | mean AUC | std AUC | min AUC | "
        "mean rank AUC | wins AUC | mean ValBestF1 | std F1 | min F1 | "
        "mean PA-F1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in perf.iterrows():
        lines.append(
            f"| {r['model']} | {r['family']} | {r['mean_AUC']:.4f} | "
            f"{r['std_AUC']:.4f} | {r['min_AUC']:.4f} | "
            f"{r['mean_rank_AUC']:.2f} | {int(r['wins_AUC'])} | "
            f"{r['mean_F1']:.4f} | {r['std_F1']:.4f} | "
            f"{r['min_F1']:.4f} | {r['mean_PA']:.4f} |")
    lines.append("")

    # 每数据集最佳
    v = core[core.threshold_source == "ValBestF1"]
    auc = v.pivot_table(index="dataset", columns="model", values="AUC",
                        aggfunc="mean")
    f1 = v.pivot_table(index="dataset", columns="model", values="F1",
                       aggfunc="mean")
    lines += ["## 2. 各数据集最佳模型", "",
              "| dataset | best AUC | AUC | best ValBestF1 | F1 |",
              "|---|---|---|---|---|"]
    for ds in auc.index:
        a = auc.loc[ds].dropna().sort_values(ascending=False)
        f = f1.loc[ds].dropna().sort_values(ascending=False)
        if len(a) and len(f):
            lines.append(
                f"| {ds} | {a.index[0]} | {a.iloc[0]:.4f} | "
                f"{f.index[0]} | {f.iloc[0]:.4f} |")
    lines.append("")

    lines += ["## 3. 数据集 × 模型 AUC", "",
              "| dataset | " + " | ".join(auc.columns) + " |",
              "|" + "|".join(["---"] * (len(auc.columns) + 1)) + "|"]
    for ds, row in auc.iterrows():
        vals = ["nan" if pd.isna(x) else f"{x:.4f}" for x in row.values]
        lines.append(f"| {ds} | " + " | ".join(vals) + " |")
    lines.append("")

    lines += ["## 4. 数据集 × 模型 ValBestF1", "",
              "| dataset | " + " | ".join(f1.columns) + " |",
              "|" + "|".join(["---"] * (len(f1.columns) + 1)) + "|"]
    for ds, row in f1.iterrows():
        vals = ["nan" if pd.isna(x) else f"{x:.4f}" for x in row.values]
        lines.append(f"| {ds} | " + " | ".join(vals) + " |")
    lines.append("")

    # 家族/跨实体
    lines += ["## 5. 家族与跨实体一致性", "",
              "| family | mean AUC | mean ValBestF1 | mean cross-entity std AUC |",
              "|---|---|---|---|"]
    for fam, g in perf.groupby("family"):
        lines.append(
            f"| {fam} | {g['mean_AUC'].mean():.4f} | "
            f"{g['mean_F1'].mean():.4f} | {g['std_AUC'].mean():.4f} |")
    lines.append("")
    groups = {
        "SML 3 machines": [d for d in auc.index if d.startswith("sml_")],
        "SMAP/MSL 4 channels": [d for d in auc.index if d.startswith("smap_")],
        "LeakDB 3 networks": [d for d in auc.index if d.startswith("leakdb_")],
    }
    lines += ["| group | best model (mean AUC) | mean AUC | worst entity AUC |",
              "|---|---|---|---|"]
    for gname, ds_list in groups.items():
        sub = auc.loc[[d for d in ds_list if d in auc.index]]
        best = sub.mean().idxmax()
        lines.append(
            f"| {gname} | {best} | {sub[best].mean():.4f} | "
            f"{sub[best].min():.4f} |")
    lines.append("")

    # 稳定性
    lines += ["## 6. 稳定性（3 seeds，ValBestF1/AUC）", ""]
    if stab is not None:
        lines += ["| model | mean seed std AUC | mean seed std F1 | "
                  "max seed std F1 | datasets |",
                  "|---|---|---|---|---|"]
        for _, r in stab.iterrows():
            lines.append(
                f"| {r['model']} | {r['mean_seed_std_AUC']:.4f} | "
                f"{r['mean_seed_std_F1']:.4f} | "
                f"{r['max_seed_std_F1']:.4f} | {int(r['datasets'])} |")
    else:
        lines.append("（未运行）")
    lines.append("")

    # 鲁棒性
    lines += ["## 7. 鲁棒性（噪声/缺失，ValBestF1）", ""]
    if rob is not None:
        for metric, label in (("dAUC", "ΔAUC"), ("dF1", "ΔF1")):
            piv = rob.pivot_table(index="model", columns="perturb",
                                  values=metric, aggfunc="mean")
            lines += [f"### {label}", "",
                      "| model | " + " | ".join(piv.columns) + " |",
                      "|" + "|".join(["---"] * (len(piv.columns) + 1)) + "|"]
            for m, row in piv.iterrows():
                vals = ["nan" if pd.isna(x) else f"{x:+.4f}"
                        for x in row.values]
                lines.append(f"| {m} | " + " | ".join(vals) + " |")
            lines.append("")
        piv = rob.pivot_table(index="model", columns="perturb",
                              values="retention_AUC", aggfunc="mean")
        lines += ["### AUC 保留率", "",
                  "| model | " + " | ".join(piv.columns) + " |",
                  "|" + "|".join(["---"] * (len(piv.columns) + 1)) + "|"]
        for m, row in piv.iterrows():
            vals = ["nan" if pd.isna(x) else f"{x:.3f}" for x in row.values]
            lines.append(f"| {m} | " + " | ".join(vals) + " |")
        lines.append("")
    else:
        lines.append("（未运行）")
    lines.append("")

    # 轻量性
    lines += ["## 8. 轻量性", "",
              "| model | mean fit_s | mean score_s | mean train_n | "
              "model size |", "|---|---|---|---|---|"]
    for _, r in perf.iterrows():
        lines.append(
            f"| {r['model']} | {r['mean_fit_s']:.3f} | "
            f"{r['mean_score_s']:.3f} | {r['mean_train_n']:.0f} | "
            f"{r['size_note']} |")
    lines.append("")

    # ValFPR1 参考
    vf = core[core.threshold_source == "ValFPR1"]
    agg = vf.groupby("model").agg(
        mean_F1=("F1", "mean"), mean_PA=("PA_F1", "mean"),
        mean_AUC=("AUC", "mean")).sort_values("mean_AUC", ascending=False)
    lines += ["## 9. ValFPR1 部署阈值参考", "",
              "- 部分数据集（BattLeDIM、SMAP、LeakDB 小测试段）在 1% FPR "
              "阈值下出现 0 报警，F1 为 0；主比较请看 AUC/ValBestF1", "",
              "| model | mean F1 | mean PA-F1 | mean AUC |",
              "|---|---|---|---|"]
    for m, r in agg.iterrows():
        lines.append(f"| {m} | {r.mean_F1:.4f} | {r.mean_PA:.4f} | "
                     f"{r.mean_AUC:.4f} |")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")


if __name__ == "__main__":
    main()
