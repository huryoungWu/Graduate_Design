# -*- coding: utf-8 -*-
"""多数据集统一缓存：把异构数据集转成统一 train/val/test 数组。

缓存目录：``runs/multidataset/<name>/``
- ``train_X.npy`` [Ntr, C] float32（只含正常段）
- ``val_X.npy`` / ``val_y.npy``
- ``test_X.npy`` / ``test_y.npy``
- ``meta.json``：features / scaler / 原始时间范围 / 备注

统一协议：Normal 训练；测试段前 20% 作为验证，后 80% 作为冻结测试。
对 PSM/SML/SMAP 等自带 test 的数据，测试段指其 test 集；
对 BATADAL/LeakDB/BattLeDIM，测试段指有标签的泄漏/攻击段。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "multidataset"


def _save(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
          feature_cols: list[str], label_col: str,
          split_frac: float = 0.2, note: str = "",
          train_ts=None, test_ts=None) -> None:
    """train_df 只含正常；test_df 含 label；前 20% test 作为验证。"""
    feature_cols = [c for c in feature_cols
                    if c in train_df.columns and c in test_df.columns]
    train_raw = train_df[feature_cols].to_numpy(np.float64)
    test_raw = test_df[feature_cols].to_numpy(np.float64)
    y = test_df[label_col].to_numpy(np.int8)
    # 按正例段切分：保证 val 和 test 都有正例
    segs, in_seg, start = [], False, 0
    for i, v in enumerate(y == 1):
        if v and not in_seg:
            start, in_seg = i, True
        elif not v and in_seg:
            segs.append((start, i - 1))
            in_seg = False
    if in_seg:
        segs.append((start, len(y) - 1))
    if not segs:
        n_val = int(len(test_raw) * split_frac)
    elif len(segs) == 1:
        a, b = segs[0]
        n_val = a + max(1, (b - a + 1) // 2)
    else:
        k = max(1, int(round(len(segs) * split_frac)))
        k = min(k, len(segs) - 1)
        n_val = segs[k - 1][1] + 1
    n_val = min(max(n_val, 1), len(test_raw) - 1)
    # 删除训练集恒定列
    keep = train_raw.std(axis=0) > 1e-12
    feature_cols = [c for c, k in zip(feature_cols, keep) if k]
    train_raw = train_raw[:, keep]
    test_raw = test_raw[:, keep]
    tr, va, te = train_raw, test_raw[:n_val], test_raw[n_val:]
    yv, yt = y[:n_val], y[n_val:]
    scaler = StandardScaler()
    tr = scaler.fit_transform(tr)
    va = scaler.transform(va)
    te = scaler.transform(te)
    feats = feature_cols
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_X.npy", tr.astype(np.float32))
    np.save(out / "val_X.npy", va.astype(np.float32))
    np.save(out / "val_y.npy", yv.astype(np.int8))
    np.save(out / "test_X.npy", te.astype(np.float32))
    np.save(out / "test_y.npy", yt.astype(np.int8))
    meta = {
        "name": name, "features": feats,
        "n_features": len(feats),
        "train_rows": int(len(tr)), "val_rows": int(len(va)),
        "test_rows": int(len(te)),
        "val_positives": int(yv.sum()), "test_positives": int(yt.sum()),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "note": note,
    }
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[multidataset] {name:28s} train={len(tr):>8,} "
          f"val={len(va):>8,} test={len(te):>8,} "
          f"pos(val/test)={int(yv.sum())}/{int(yt.sum())}")


def _num_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


# ----------------------------------------------------------------------
# 具体数据集
# ----------------------------------------------------------------------
def build_psm() -> None:
    base = ROOT / "01_PSM/RANSynCoders-main/data"
    tr = pd.read_csv(base / "train.csv")
    te = pd.read_csv(base / "test.csv")
    lab = pd.read_csv(base / "test_label.csv")
    lab = lab.rename(columns={lab.columns[0]: "timestamp_(min)",
                              lab.columns[1]: "label"})
    te = te.merge(lab, on="timestamp_(min)", how="left")
    te["label"] = te["label"].fillna(0).astype(int)
    feats = _num_cols(tr, {"timestamp_(min)"})
    tr = tr[feats].dropna()
    te = te.dropna(subset=feats)
    _save("psm", tr, te, feats, "label",
          note="PSM train normal / test labeled; first 20% test as val")


def build_sml(machines: list[str]) -> None:
    base = ROOT / "02_SML/OmniAnomaly-master/ServerMachineDataset"
    for m in machines:
        tr = pd.read_csv(base / f"train/{m}.txt", header=None)
        te = pd.read_csv(base / f"test/{m}.txt", header=None)
        lab = pd.read_csv(base / f"test_label/{m}.txt", header=None)
        feats = [f"f{i}" for i in range(tr.shape[1])]
        tr.columns = feats
        te.columns = feats
        te["label"] = lab.iloc[:, 0].to_numpy(np.int8)
        if te["label"].sum() == 0:
            print(f"[skip] SML {m}: no positive in test")
            continue
        _save(f"sml_{m}", tr, te, feats, "label",
              note=f"SML {m}; test_label 0/1")


def build_batadal() -> None:
    df = pd.read_csv(ROOT / "03_BATADAL/BATADAL_all.csv")
    df["label"] = df["label"].fillna(0).astype(int)
    tr = df[df["subset"] == "train_normal_2014_2015"].copy()
    te = df[df["subset"] == "test_labeled_2016"].copy()
    feats = _num_cols(df, {"timestamp", "label", "subset"})
    _save("batadal", tr, te, feats, "label",
          note="BATADAL 2014-2015 normal -> 2016 labeled")


def build_battledim(year: str) -> None:
    df = pd.read_csv(ROOT / f"04_BattLeDIM/BattLeDIM_{year}_5min.csv")
    df["label"] = df["leak_onset"].astype(int)
    feats = _num_cols(
        df, {"timestamp", "any_leak", "leak_onset",
             "leak_onset_count", "total_leak_m3h"})
    feats = [c for c in feats if not c.startswith("leak_")]
    if year == "2018":
        onset = df.index[df["leak_onset"] == 1]
        first = int(onset[0]) if len(onset) else int(len(df) * 0.7)
        tr = df.iloc[:first].copy()
        te = df.iloc[first:].copy()
        _save("battledim_2018", tr, te, feats, "label",
              note="2018: first 8 days normal -> leak onsets (event-level)")
    else:
        tr = pd.read_csv(
            ROOT / "04_BattLeDIM/BattLeDIM_2018_5min.csv")
        onset = tr.index[tr["leak_onset"] == 1]
        first = int(onset[0]) if len(onset) else int(len(tr) * 0.7)
        tr = tr.iloc[:first].copy()
        te = df.copy()
        _save("battledim_2019", tr, te, feats, "label",
              note="2018 pre-leak normal -> 2019 leak onsets (cross-year)")


def build_smap(channels: list[str]) -> None:
    lab = pd.read_parquet(
        ROOT / "05_SMAP_MSL/telemanom/data/labeled_anomalies.parquet")
    lab = lab.set_index("chan_id")
    for ch in channels:
        p = ROOT / f"05_SMAP_MSL/telemanom/data/test/{ch}.parquet"
        if not p.exists():
            print(f"[skip] SMAP {ch}: file missing")
            continue
        df = pd.read_parquet(p)
        n = len(df)
        y = np.zeros(n, dtype=np.int8)
        seqs = lab.loc[ch, "anomaly_sequences"]
        if isinstance(seqs, str):
            seqs = ast.literal_eval(seqs)
        seqs = list(seqs) if not isinstance(seqs, np.ndarray) else seqs
        # 兼容 [[a,b], ...] 与扁平 [a,b]
        if len(seqs) == 2 and np.isscalar(seqs[0]):
            seqs = [seqs]
        for item in seqs:
            a, b = item[0], item[1]
            y[int(a):min(int(b) + 1, n)] = 1
        feats = [c for c in df.columns if c != "timestep"]
        df["label"] = y
        first = int(np.flatnonzero(y == 1)[0]) if y.sum() else int(n * 0.7)
        # 从正常段尾部留出一部分给测试段，保证 val 同时有正常/异常
        holdout = min(first, max(1000, int(first * 0.2)))
        tr = df.iloc[:first - holdout].copy()
        te = df.iloc[first - holdout:].copy()
        if len(tr) < 500 or te["label"].sum() == 0:
            print(f"[skip] SMAP {ch}: train/test too small")
            continue
        _save(f"smap_{ch}", tr, te, feats, "label",
              note=f"SMAP {ch}; train=pre-first-anomaly normal")


def build_leakdb() -> None:
    base = ROOT / "07_LeakDB"
    nets = {
        "hanoi500": ("Hanoi500_Scenario-001.csv", "Hanoi500_Scenario-007.csv"),
        "hanoi1000": ("Hanoi1000_Scenario-001.csv", "Hanoi1000_Scenario-004.csv"),
        "net1": ("Net1_1000_Scenario-004.csv", "Net1_1000_Scenario-002.csv"),
    }
    for net, (no_leak, leak) in nets.items():
        tr = pd.read_csv(base / no_leak)
        te = pd.read_csv(base / leak)
        feats = _num_cols(
            te, {"timestamp", "leak_label"} |
            {c for c in te.columns if c.startswith("leak_")})
        if tr["leak_label"].sum() != 0:
            print(f"[warn] LeakDB {net}: no-leak scenario has positives")
        _save(f"leakdb_{net}", tr, te, feats, "leak_label",
              note=f"LeakDB {net}: no-leak -> {leak}")


def build_swat_wadi() -> None:
    from models.common.data import load_swat
    for name, csv in [("swat", "08_SWaT/SWaT_all_1s.csv"),
                      ("wadi", "09_WADI/WADI_all_1s.csv")]:
        b = load_swat(ROOT / csv, val_frac=0.20, scaler="standard",
                      verbose=False)
        # 复用 load_swat 的 scaler 结果
        out = OUT / name
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "train_X.npy", b.train_X.astype(np.float32))
        np.save(out / "val_X.npy", b.val_X.astype(np.float32))
        np.save(out / "val_y.npy", b.val_label.astype(np.int8))
        np.save(out / "test_X.npy", b.test_X.astype(np.float32))
        np.save(out / "test_y.npy", b.test_label.astype(np.int8))
        meta = {
            "name": name, "features": b.sensor_cols,
            "n_features": len(b.sensor_cols),
            "train_rows": len(b.train_X), "val_rows": len(b.val_X),
            "test_rows": len(b.test_X),
            "val_positives": int(b.val_label.sum()),
            "test_positives": int(b.test_label.sum()),
            "note": "existing SWaT/WADI protocol",
        }
        (out / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"[multidataset] {name:28s} train={len(b.train_X):>8,} "
              f"val={len(b.val_X):>8,} test={len(b.test_X):>8,}")


def load(name: str):
    d = OUT / name
    return (np.load(d / "train_X.npy"),
            np.load(d / "val_X.npy"),
            np.load(d / "val_y.npy"),
            np.load(d / "test_X.npy"),
            np.load(d / "test_y.npy"),
            json.loads((d / "meta.json").read_text(encoding="utf-8")))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_psm()
    build_sml(["machine-1-1", "machine-1-2", "machine-2-1"])
    build_batadal()
    build_battledim("2018")
    build_battledim("2019")
    build_smap(["A-1", "A-2", "P-1", "S-1"])
    build_leakdb()
    build_swat_wadi()


if __name__ == "__main__":
    main()
