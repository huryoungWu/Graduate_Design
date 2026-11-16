# -*- coding: utf-8 -*-
"""合并 Isolation 家族/相邻流派与深度模型的对比结果。

输入：
- runs/recon_comparison/ISOLATION_PEERS_FAST.csv
- runs/recon_comparison/ISOLATION_PEERS_INNE.csv
- runs/recon_comparison/ISOLATION_PEERS_DIF.csv
- runs/recon_comparison/IFOREST_FUSION.csv（深度模型 Model_only 行 + IF）

输出：
- runs/recon_comparison/ISOLATION_PEERS_ALL.csv
- runs/recon_comparison/ISOLATION_PEERS_ALL.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RC = ROOT / "runs" / "recon_comparison"
OUT_CSV = RC / "ISOLATION_PEERS_ALL.csv"
OUT_MD = RC / "ISOLATION_PEERS_ALL.md"

PEER_FILES = [
    RC / "ISOLATION_PEERS_FAST.csv",
    RC / "ISOLATION_PEERS_INNE.csv",
    RC / "ISOLATION_PEERS_DIF.csv",
]

COLS = ["family", "model", "threshold_source", "Precision", "Recall",
        "F1", "FPR", "AUC", "PR_AUC", "PA_F1", "n", "positives",
        "fit_s", "score_s", "train_n", "note", "source"]

SOURCES = ["ValFPR1", "Normal99", "ValBestF1", "Oracle"]


def _load_peers() -> pd.DataFrame:
    frames = []
    for path in PEER_FILES:
        if path.exists():
            d = pd.read_csv(path)
            d["source"] = "peer"
            frames.append(d)
        else:
            print(f"[merge] 跳过不存在: {path}")
    if not frames:
        raise SystemExit("没有 peer 结果文件")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["model", "threshold_source"],
                              keep="first")
    return out


def _load_deep() -> pd.DataFrame:
    path = RC / "IFOREST_FUSION.csv"
    d = pd.read_csv(path)
    d = d[d["fusion"] == "Model_only"].copy()
    d["family"] = "deep model"
    d["source"] = "deep"
    d["fit_s"] = np.nan
    d["score_s"] = np.nan
    d["train_n"] = np.nan
    d["note"] = "retrained deep model"
    return d[COLS]


def _load_if_full() -> pd.DataFrame:
    path = RC / "IFOREST_FUSION.csv"
    d = pd.read_csv(path)
    d = d[d["fusion"] == "IF_only"].copy()
    d = d.drop_duplicates(subset=["threshold_source"], keep="first")
    d["family"] = "isolation"
    d["model"] = "IsolationForest(full99)"
    d["source"] = "IF full99"
    d["fit_s"] = np.nan
    d["score_s"] = np.nan
    d["train_n"] = np.nan
    d["note"] = "full-Normal Normal99; ValFPR1/ValBestF1 same as peer IF"
    return d[COLS]


def _table(df: pd.DataFrame, source: str) -> list[str]:
    sub = df[df.threshold_source == source].sort_values(
        "F1", ascending=False)
    lines = [
        f"## {source}", "",
        "| family | model | Precision | Recall | F1 | FPR | AUC | "
        "PR_AUC | PA_F1 | n | source |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in sub.iterrows():
        lines.append(
            f"| {r['family']} | {r['model']} | {r['Precision']:.4f} | "
            f"{r['Recall']:.4f} | {r['F1']:.4f} | {r['FPR']:.4f} | "
            f"{r['AUC']:.4f} | {r['PR_AUC']:.4f} | {r['PA_F1']:.4f} | "
            f"{int(r['n'])} | {r['source']} |")
    lines.append("")
    return lines


def main() -> None:
    peers = _load_peers()
    deep = _load_deep()
    df = pd.concat([peers, deep], ignore_index=True)
    # 只保留需要的列
    df = df[COLS]
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    lines = [
        "# Isolation 家族 / 相邻流派 vs 深度模型（SWaT）", "",
        "- 训练：Normal only；验证：Attack 前 20%；测试：Attack 后 80% 冻结",
        "- 12 步窗口；窗口已对齐到深度模型的 359,865 个测试窗口",
        "- detector 的 Normal99 除 INNE 外均用整个 Normal 段；"
        "INNE 用最后 5 万点（见 note），其 Normal99 不直接可比",
        "- `source=deep` 为上一轮重训的 14 个深度模型；`source=peer` "
        "为本次新实现的 detector",
        "- 近似/子采样说明见各 detector 的 note 列", "",
    ]
    for src in SOURCES:
        lines += _table(df, src)

    # 家族平均（ValFPR1）
    lines += ["## ValFPR1 家族平均", "",
              "| family | mean F1 | mean PA-F1 | mean AUC | count |",
              "|---|---|---|---|---|"]
    v = df[df.threshold_source == "ValFPR1"]
    for fam, g in v.groupby("family"):
        lines.append(
            f"| {fam} | {g['F1'].mean():.4f} | {g['PA_F1'].mean():.4f} | "
            f"{g['AUC'].mean():.4f} | {len(g)} |")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")
    print(v.sort_values("F1", ascending=False)[
        ["family", "model", "F1", "PA_F1", "AUC"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
