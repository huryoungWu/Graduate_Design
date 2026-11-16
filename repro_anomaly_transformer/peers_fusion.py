# -*- coding: utf-8 -*-
"""经典检测器（ECOD/COPOD/IF/HST/RRCF/DIF/HBOS）与深度模型的多策略融合。

协议与 ``iforest_fusion.py`` 一致：
- 训练：Normal only；验证：Attack 前 20%；测试：Attack 后 80% 冻结；
- 12 步窗口；窗口对齐到深度模型的 359,865 个测试窗口；
- 阈值：Normal99 / ValFPR1 / ValBestF1；Oracle 仅上限。

融合：
- decision-level：AND、OR；
- score-level：mean_z、max_z、min_z（Normal 段 z-score 后组合）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.common.config import resolve_device
from models.common.data import load_swat
from models.classical.isolation_peers import (_window, _window_labels,
                                               START_OFFSET, WINDOW)

from .benchmark_all_models import MODELS  # noqa: F401
from .iforest_fusion import (_binary_metrics, _auc_ap, _eval_continuous,
                             _load_frames, _normal_stats, _threshold,
                             _transform, CACHE_DIR)

ROOT = Path(__file__).resolve().parents[1]
RC = ROOT / "runs" / "recon_comparison"
PEER_DIR = ROOT / "runs" / "isolation_peers"
OUT_CSV = RC / "PEERS_FUSION.csv"
OUT_MD = RC / "PEERS_FUSION.md"

# 显示名 -> 分数文件名
BASE_FILES = {
    "ECOD": "ECOD",
    "COPOD": "COPOD",
    "IsolationForest": "IsolationForest",
    "HST(batch)": "HST",
    "RRCF(batch RCT)": "RRCF",
    "DIF": "DIF",
    "HBOS": "HBOS",
}
METHODS = ["AND", "OR", "mean_z", "max_z", "min_z"]
SOURCES = ["Normal99", "ValFPR1", "ValBestF1", "Oracle"]


def _load_peer(file: str) -> dict:
    z = np.load(PEER_DIR / f"{file}.npz")
    return {
        "normal": _window(z["normal"]),
        "val": _window(z["val"])[START_OFFSET:],
        "test": _window(z["test"])[START_OFFSET:],
        "y_val": _window_labels(z["y_val"])[START_OFFSET:],
        "y_test": _window_labels(z["y_test"])[START_OFFSET:],
    }


def _load_deep():
    val_frames, test_frames, meta = _load_frames()
    deep = {}
    for name, info in meta.items():
        if name == "IsolationForest":
            continue
        cache = CACHE_DIR / f"{info['artifact']}_normal.npy"
        if not cache.exists():
            print(f"[skip] {name}: 缺少 Normal cache")
            continue
        deep[name] = {
            "normal": np.load(cache),
            "val_frame": val_frames[name],
            "test_frame": test_frames[name],
        }
    return deep


def _peer_index(csv: str) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex,
                                   dict]:
    b = load_swat(csv, val_frac=0.20, scaler="standard", verbose=False)
    val_ts = pd.to_datetime(b.val_df["timestamp"]).to_numpy()
    test_ts = pd.to_datetime(b.test_df["timestamp"]).to_numpy()
    b.train_df = b.val_df = b.test_df = None
    off = START_OFFSET + WINDOW - 1  # 71
    return (pd.DatetimeIndex(val_ts[off:]),
            pd.DatetimeIndex(test_ts[off:]), {})


def _align(partner: str, split: str, deep: dict,
           peer: dict, index: pd.DatetimeIndex):
    if partner == "IsolationForest":
        s = peer["IsolationForest"][split]
        y = peer["IsolationForest"][f"y_{split}"]
        return s, y, np.ones(len(s), dtype=bool)
    df = (deep[partner]["test_frame"] if split == "test"
          else deep[partner]["val_frame"]).reindex(index)
    mask = df["score"].notna().to_numpy()
    return (df.loc[mask, "score"].to_numpy(float),
            df.loc[mask, "label"].to_numpy(int), mask)


def _z_pair(base_n, part_n, base_v, part_v, base_t, part_t):
    T = min(len(base_n), len(part_n))
    bn, pn = np.asarray(base_n)[-T:], np.asarray(part_n)[-T:]
    sb, sp = _normal_stats(bn), _normal_stats(pn)
    out = {}
    for key in ("normal", "val", "test"):
        b = {"normal": bn, "val": base_v, "test": base_t}[key]
        p = {"normal": pn, "val": part_v, "test": part_t}[key]
        out[key] = (_transform(b, sb, "z"), _transform(p, sp, "z"))
    return out


def _eval_method(base, partner, method, source, b, p, yv, yt,
                 base_only, partner_only):
    if method in ("AND", "OR"):
        sign_b, thr_b = _threshold(
            source, base["normal"], b["val"], b["test"], yv, yt)
        sign_p, thr_p = _threshold(
            source, p["normal_pred"], p["val"], p["test"], yv, yt)
        a_b = sign_b * b["test"] > thr_b
        a_p = sign_p * p["test"] > thr_p
        alarm = (a_b & a_p) if method == "AND" else (a_b | a_p)
        met = _binary_metrics(yt, alarm)
        cont = (np.maximum(sign_b * b["test"], sign_p * p["test"])
                if method == "OR"
                else np.minimum(sign_b * b["test"], sign_p * p["test"]))
        met["AUC"], met["PR_AUC"] = _auc_ap(yt, cont)
        return met, float(thr_b), float(thr_p)

    # score-level
    z = _z_pair(base["normal"], p["normal_pred"],
                b["val"], p["val"], b["test"], p["test"])
    if method == "mean_z":
        fused = {k: 0.5 * (z[k][0] + z[k][1])
                 for k in ("normal", "val", "test")}
    elif method == "max_z":
        fused = {k: np.maximum(z[k][0], z[k][1])
                 for k in ("normal", "val", "test")}
    else:
        fused = {k: np.minimum(z[k][0], z[k][1])
                 for k in ("normal", "val", "test")}
    sign, thr = _threshold(source, fused["normal"], fused["val"],
                           fused["test"], yv, yt)
    met = _eval_continuous(yt, fused["test"], sign, thr)
    return met, float(thr), np.nan


def main() -> None:
    p = argparse.ArgumentParser(description="经典检测器 × 深度模型融合")
    p.add_argument("--csv", default=str(ROOT / "08_SWaT/SWaT_all_1s.csv"))
    p.add_argument("--bases", nargs="*", default=None)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"])
    args = p.parse_args()

    print("[fusion] 加载经典检测器分数 ...")
    peers = {name: _load_peer(f) for name, f in BASE_FILES.items()}
    print("[fusion] 加载深度模型 val/test 窗口 ...")
    deep = _load_deep()

    val_index, test_index, _ = _peer_index(args.csv)
    print(f"[fusion] val windows={len(val_index):,} "
          f"test windows={len(test_index):,}")

    bases = args.bases or list(BASE_FILES)
    rows = []

    for base_name in bases:
        base = peers[base_name]
        partners = [k for k in deep if k != base_name]
        if base_name != "IsolationForest":
            partners.append("IsolationForest")
        print(f"\n[fusion] base={base_name} partners={len(partners)}")

        # 基座单独
        base_only = {}
        for source in SOURCES:
            sign, thr = _threshold(
                source, base["normal"], base["val"], base["test"],
                base["y_val"], base["y_test"])
            met = _eval_continuous(
                base["y_test"], base["test"], sign, thr)
            base_only[source] = met
            rows.append({
                "base": base_name, "partner": "-", "method": "Base_only",
                "threshold_source": source,
                "threshold_base": float(thr), "threshold_partner": np.nan,
                **met, "n": int(len(base["test"])),
                "positives": int(base["y_test"].sum()),
            })

        for partner_name in partners:
            if partner_name == "IsolationForest":
                pdat = peers["IsolationForest"]
                p_normal = pdat["normal"]
            else:
                pdat = None
                p_normal = deep[partner_name]["normal"]

            # 对齐
            st, yt, mt = _align(partner_name, "test", deep, peers,
                                test_index)
            sv, yv, mv = _align(partner_name, "val", deep, peers,
                                val_index)
            if partner_name == "IsolationForest":
                sv, yv = pdat["val"], pdat["y_val"]
                st, yt = pdat["test"], pdat["y_test"]
                mv = np.ones(len(sv), dtype=bool)
                mt = np.ones(len(st), dtype=bool)

            b_test, b_val = base["test"][mt], base["val"][mv]
            b_yt, b_yv = base["y_test"][mt], base["y_val"][mv]
            if not np.array_equal(b_yt, yt):
                print(f"[warn] {base_name}+{partner_name}: test 标签不一致")
            if not np.array_equal(b_yv, yv):
                print(f"[warn] {base_name}+{partner_name}: val 标签不一致")

            b = {"normal": base["normal"], "val": b_val, "test": b_test}
            q = {"normal_pred": p_normal, "val": sv, "test": st}

            partner_only = {}
            for source in SOURCES:
                sign, thr = _threshold(
                    source, p_normal, sv, st, yv, yt)
                met = _eval_continuous(yt, st, sign, thr)
                partner_only[source] = met
                rows.append({
                    "base": base_name, "partner": partner_name,
                    "method": "Partner_only",
                    "threshold_source": source,
                    "threshold_base": np.nan,
                    "threshold_partner": float(thr), **met,
                    "n": int(len(st)), "positives": int(yt.sum()),
                })

            for method in METHODS:
                for source in SOURCES:
                    met, thr_b, thr_p = _eval_method(
                        base, partner_name, method, source, b, q,
                        yv, yt, base_only[source], partner_only[source])
                    rows.append({
                        "base": base_name, "partner": partner_name,
                        "method": method,
                        "threshold_source": source,
                        "threshold_base": thr_b,
                        "threshold_partner": thr_p, **met,
                        "n": int(len(st)), "positives": int(yt.sum()),
                    })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # ---------------- Summary ----------------
    lines = [
        "# 经典检测器 × 深度模型融合（SWaT）", "",
        "- 基座：ECOD / COPOD / IF / HST / RRCF / DIF / HBOS",
        "- 伙伴：14 个重训深度模型 + IF（基座为 IF 时不重复）",
        "- 训练 Normal only；验证 Attack 前 20%；测试 Attack 后 80% 冻结",
        "- 融合：AND / OR / mean_z / max_z / min_z",
        "- 阈值：Normal99 / ValFPR1 / ValBestF1；Oracle 仅上限", "",
    ]
    for source in SOURCES:
        sub = out[out.threshold_source == source]
        base_rows = sub[sub.method == "Base_only"].set_index("base")
        lines += [f"## {source}", "",
                  "| base | base F1 | base PA-F1 | AND mean F1 | "
                  "AND ΔF1 | AND mean PA-F1 | AND ΔPA | AND best partner | "
                  "best F1 | best PA-F1 | mean_z ΔF1 | min_z ΔF1 | "
                  "max_z ΔF1 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for base_name in bases:
            bf = base_rows.loc[base_name]
            ands = sub[(sub.base == base_name) & (sub.method == "AND")]
            means = sub[(sub.base == base_name)
                        & (sub.method == "mean_z")]
            mins = sub[(sub.base == base_name) & (sub.method == "min_z")]
            maxs = sub[(sub.base == base_name) & (sub.method == "max_z")]
            if ands.empty:
                continue
            best = ands.loc[ands["F1"].idxmax()]
            lines.append(
                f"| {base_name} | {bf['F1']:.4f} | {bf['PA_F1']:.4f} | "
                f"{ands['F1'].mean():.4f} | "
                f"{ands['F1'].mean() - bf['F1']:+.4f} | "
                f"{ands['PA_F1'].mean():.4f} | "
                f"{ands['PA_F1'].mean() - bf['PA_F1']:+.4f} | "
                f"{best['partner']} | {best['F1']:.4f} | "
                f"{best['PA_F1']:.4f} | "
                f"{means['F1'].mean() - bf['F1']:+.4f} | "
                f"{mins['F1'].mean() - bf['F1']:+.4f} | "
                f"{maxs['F1'].mean() - bf['F1']:+.4f} |")
        lines.append("")

    # 最佳组合 top30（ValFPR1）
    v = out[(out.threshold_source == "ValFPR1")
            & (out.method.isin(METHODS))].copy()
    top = v.sort_values("F1", ascending=False).head(30)
    lines += ["## ValFPR1 最佳组合 Top 30", "",
              "| base | partner | method | Precision | Recall | F1 | "
              "PA-F1 | AUC |", "|---|---|---|---|---|---|---|---|"]
    for _, r in top.iterrows():
        lines.append(
            f"| {r['base']} | {r['partner']} | {r['method']} | "
            f"{r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | "
            f"{r['PA_F1']:.4f} | {r['AUC']:.4f} |")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n已保存: {OUT_CSV}")
    print(f"已保存: {OUT_MD}")

    # console: ValFPR1 base summary
    print("\n===== ValFPR1 base summary =====")
    sub = out[out.threshold_source == "ValFPR1"]
    for base_name in bases:
        bf = sub[(sub.base == base_name) & (sub.method == "Base_only")]
        ands = sub[(sub.base == base_name) & (sub.method == "AND")]
        if bf.empty or ands.empty:
            continue
        b = bf.iloc[0]
        print(f"{base_name:18s} base {b['F1']:.4f}/{b['PA_F1']:.4f}  "
              f"AND mean {ands['F1'].mean():.4f}/{ands['PA_F1'].mean():.4f}  "
              f"ΔF1 {ands['F1'].mean()-b['F1']:+.4f}  "
              f"ΔPA {ands['PA_F1'].mean()-b['PA_F1']:+.4f}")


if __name__ == "__main__":
    main()
