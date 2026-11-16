# -*- coding: utf-8 -*-
"""Isolation 家族与相邻流派异常检测器的统一 SWaT 基准。

覆盖：
- isolation-based：IsolationForest、Robust Random Cut Forest (batch RCT)、
  INNE、DIF、Half-Space Trees (batch HST)
- density-based：LOF、HBOS
- boundary-based：One-Class SVM、DeepSVDD
- distribution-based：ECOD、COPOD

协议与项目其它模型一致：
- 只用 Normal 段拟合；
- 验证：Attack 前 20%；测试：Attack 后 80% 冻结；
- 逐点分数 -> 12 步滑窗均值；
- 阈值口径：Normal99（Normal 尾段 99% 分位）、ValBestF1、ValFPR1、
  Oracle（测试标签，仅上限）。

说明
----
- RRCF / HST 使用批量随机切分树近似实现（RCT depth / 2^-depth），
  不是官方流式实现；RRCF 未实现 CoDisp。
- 受内存/时间限制，INNE/DIF/DeepSVDD 用 50k Normal 子样本，
  LOF 用 20k，OCSVM 用 10k；HBOS/ECOD/COPOD/RRCF/HST 用全部 Normal。
- 所有方法使用 StandardScaler（与 IsolationForest 一致）；INNE 额外提供
  PCA-10 变体以缓解高维饱和。

用法::

    python -m models.classical.isolation_peers --smoke
    python -m models.classical.isolation_peers --methods INNE DIF LOF HBOS
    python -m models.classical.isolation_peers
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from models.common.config import resolve_device
from models.common.data import load_swat
from models.common.protocol import select_threshold
from models.reconstruction.usad_extended_metrics import point_adjust

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "runs" / "recon_comparison" / "ISOLATION_PEERS.csv"
OUT_MD = ROOT / "runs" / "recon_comparison" / "ISOLATION_PEERS.md"
SCORE_DIR = ROOT / "runs" / "isolation_peers"

WINDOW = 12
NORMAL_TAIL = 50000
# 深度模型 seq_len=60：其 12 步窗口从 t=71 开始；本脚本点分数从 t=0 开始，
# 窗口从 t=11 开始。丢弃前 60 个窗口后，测试集窗口数与融合实验一致。
START_OFFSET = 60

THRESHOLD_SOURCES = ["Normal99", "ValBestF1", "ValFPR1", "Oracle"]


# ----------------------------------------------------------------------
# 指标与窗口
# ----------------------------------------------------------------------
def _window(x, w: int = WINDOW) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(x, w).mean(axis=1)


def _window_labels(y, w: int = WINDOW) -> np.ndarray:
    y = np.asarray(y, dtype=np.int8)
    return np.lib.stride_tricks.sliding_window_view(y, w).max(axis=1)


def _metrics(y, s, thr) -> dict:
    from repro_anomaly_transformer.benchmark_all_models import _metrics
    return _metrics(y, s, thr)[0]


def _oracle(y, s):
    from repro_anomaly_transformer.benchmark_all_models import _oracle
    return _oracle(y, s)


# ----------------------------------------------------------------------
# 批量随机切分树（RRCF / HST 近似）
# ----------------------------------------------------------------------
class _RandomCutTree:
    def __init__(self, X: np.ndarray, depth_limit: int,
                 rng: np.random.RandomState):
        self.X = X
        self.depth_limit = int(depth_limit)
        self.rng = rng
        self.root = self._build(np.arange(len(X)), 0)

    def _build(self, idx: np.ndarray, depth: int) -> dict:
        node = {"depth": depth, "size": int(len(idx))}
        if depth >= self.depth_limit or len(idx) <= 1:
            node.update(leaf=True, mass=int(len(idx)))
            return node
        Xn = self.X[idx]
        ranges = Xn.max(axis=0) - Xn.min(axis=0)
        valid = np.flatnonzero(ranges > 0)
        if valid.size == 0:
            node.update(leaf=True, mass=int(len(idx)))
            return node
        q = int(self.rng.choice(valid))
        lo = float(Xn[:, q].min())
        hi = float(Xn[:, q].max())
        cut = float(self.rng.uniform(lo, hi))
        mask = Xn[:, q] <= cut
        left, right = idx[mask], idx[~mask]
        if left.size == 0 or right.size == 0:
            node.update(leaf=True, mass=int(len(idx)))
            return node
        node.update(
            leaf=False, q=q, cut=cut,
            left=self._build(left, depth + 1),
            right=self._build(right, depth + 1))
        return node

    def depths(self, Xq: np.ndarray) -> np.ndarray:
        depth = np.zeros(len(Xq), dtype=np.int32)
        stack = [(self.root, np.arange(len(Xq)))]
        while stack:
            node, ids = stack.pop()
            if ids.size == 0:
                continue
            if node.get("leaf"):
                depth[ids] = node["depth"]
                continue
            q, cut = node["q"], node["cut"]
            go = Xq[ids, q] <= cut
            stack.append((node["left"], ids[go]))
            stack.append((node["right"], ids[~go]))
        return depth


class _BatchCutForest:
    """随机切分树集成；score_kind=depth 或 exp_depth。"""

    def __init__(self, n_trees: int = 100, sample_size: int = 256,
                 seed: int = 42, score_kind: str = "depth"):
        self.n_trees = int(n_trees)
        self.sample_size = int(sample_size)
        self.rng = np.random.RandomState(seed)
        self.score_kind = score_kind
        self.trees: list[_RandomCutTree] = []
        self.depth_limit = max(1, int(math.ceil(math.log2(self.sample_size))))

    def fit(self, X: np.ndarray) -> "_BatchCutForest":
        n = len(X)
        for _ in range(self.n_trees):
            size = min(self.sample_size, n)
            idx = self.rng.choice(n, size=size, replace=False)
            self.trees.append(
                _RandomCutTree(X[idx], self.depth_limit, self.rng))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(X), dtype=np.float64)
        for tree in self.trees:
            d = tree.depths(X)
            if self.score_kind == "depth":
                scores -= d.astype(np.float64)
            else:  # HST: mean(2^-depth)
                scores += np.exp2(-d.astype(np.float64))
        return scores / len(self.trees)


def _rrcf_scores(X_train, X_normal, X_val, X_test, smoke):
    n_trees = 30 if smoke else 100
    sample = 128 if smoke else 256
    model = _BatchCutForest(
        n_trees=n_trees, sample_size=sample, seed=42,
        score_kind="depth").fit(X_train)
    return (model.score_samples(X_normal),
            model.score_samples(X_val),
            model.score_samples(X_test))


def _hst_scores(X_train, X_normal, X_val, X_test, smoke):
    n_trees = 30 if smoke else 100
    sample = 128 if smoke else 256
    model = _BatchCutForest(
        n_trees=n_trees, sample_size=sample, seed=7,
        score_kind="exp_depth").fit(X_train)
    return (model.score_samples(X_normal),
            model.score_samples(X_val),
            model.score_samples(X_test))


# ----------------------------------------------------------------------
# PyOD 包装
# ----------------------------------------------------------------------
def _subsample(X, n, seed=42):
    if n is None or n >= len(X):
        return X, len(X)
    idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
    return X[idx], n


def _fit_pyod(name, ctor, X_train, X_normal_full, X_val, X_test,
              fit_n, smoke):
    if smoke:
        fit_n = 3000 if fit_n is None else min(fit_n, 3000)
    Xf, n = _subsample(X_train, fit_n)
    t0 = time.time()
    model = ctor()
    model.fit(Xf)
    fit_s = time.time() - t0
    t0 = time.time()
    sn = np.asarray(model.decision_function(X_normal_full)).ravel()
    sv = np.asarray(model.decision_function(X_val)).ravel()
    st = np.asarray(model.decision_function(X_test)).ravel()
    score_s = time.time() - t0
    return sn, sv, st, fit_s, score_s, n, model


def _make_pyod(name, device, smoke):
    from pyod.models.inne import INNE
    from pyod.models.dif import DIF
    from pyod.models.lof import LOF
    from pyod.models.hbos import HBOS
    from pyod.models.ocsvm import OCSVM
    from pyod.models.deep_svdd import DeepSVDD
    from pyod.models.ecod import ECOD
    from pyod.models.copod import COPOD

    if name == "INNE":
        return (lambda: INNE(n_estimators=50, max_samples=256,
                             contamination=0.01, random_state=42),
                50000, "INNE 50 trees, 45-D raw; Normal99 on tail")
    if name == "INNE-PCA10":
        return (None, 50000, "INNE 50 trees on PCA-10; Normal99 on tail")
    if name == "DIF":
        return (lambda: DIF(
            n_ensemble=5, n_estimators=6, max_samples=256,
            representation_dim=20, hidden_neurons=[64, 32],
            contamination=0.01, random_state=42,
            device=str(device)), 50000,
            "DIF, 5 ensembles (reduced for runtime)")
    if name == "LOF":
        return (lambda: LOF(n_neighbors=20, novelty=True,
                            contamination=0.01, n_jobs=-1),
                20000, "LOF k=20")
    if name == "HBOS":
        return (lambda: HBOS(n_bins=10, contamination=0.01),
                None, "HBOS 10 bins, full Normal")
    if name == "OCSVM":
        return (lambda: OCSVM(kernel="rbf", nu=0.01, gamma="scale",
                              cache_size=1000, contamination=0.01),
                10000, "RBF OCSVM nu=0.01")
    if name == "DeepSVDD":
        return (lambda: DeepSVDD(
            n_features=45, use_ae=False, hidden_neurons=[64, 32],
            epochs=50, batch_size=128, preprocessing=False,
            contamination=0.01, verbose=0, random_state=42),
            50000, "DeepSVDD 50 epochs")
    if name == "ECOD":
        return (lambda: ECOD(contamination=0.01, n_jobs=1),
                None, "ECOD, full Normal")
    if name == "COPOD":
        return (lambda: COPOD(contamination=0.01, n_jobs=1),
                None, "COPOD, full Normal")
    raise ValueError(name)


# ----------------------------------------------------------------------
# 评估
# ----------------------------------------------------------------------
def _evaluate(family, name, s_normal, s_val, y_val, s_test, y_test,
              fit_s, score_s, train_n, note) -> list[dict]:
    nw = _window(s_normal)
    vw, vy = _window(s_val), _window_labels(y_val)
    tw, ty = _window(s_test), _window_labels(y_test)
    vw, vy = vw[START_OFFSET:], vy[START_OFFSET:]
    tw, ty = tw[START_OFFSET:], ty[START_OFFSET:]
    n = len(tw)

    rows = []
    for source in THRESHOLD_SOURCES:
        if source == "Normal99":
            thr, sign = float(np.percentile(nw, 99)), 1
        elif source == "ValBestF1":
            thr, _ = select_threshold(vw, vy)
            thr, sign = float(thr), 1
        elif source == "ValFPR1":
            normal_val = vw[vy == 0]
            thr = (float(np.percentile(normal_val, 99))
                   if normal_val.size else float(np.percentile(vw, 99)))
            sign = 1
        else:
            sign, thr = _oracle(ty, tw)
            sign, thr = int(sign), float(thr)
        met = _metrics(ty, sign * tw, thr)
        rows.append({
            "family": family, "model": name,
            "threshold_source": source, "threshold": thr,
            "Precision": met["Precision"], "Recall": met["Recall"],
            "F1": met["F1"], "FPR": met["FPR"], "AUC": met["AUC"],
            "PR_AUC": met["PR_AUC"], "PA_F1": met["PA_F1"],
            "n": int(n), "positives": int(ty.sum()),
            "fit_s": float(fit_s), "score_s": float(score_s),
            "train_n": int(train_n), "note": note,
        })
    return rows


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Isolation 家族及相邻流派基准")
    p.add_argument("--csv", default=str(ROOT / "08_SWaT/SWaT_all_1s.csv"))
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out-csv", default=str(OUT_CSV))
    p.add_argument("--out-md", default=str(OUT_MD))
    args = p.parse_args()

    device = resolve_device(args.device)
    print(f"[peers] device={device}")

    print("[peers] 加载 SWaT ...")
    b = load_swat(args.csv, val_frac=0.20, scaler="standard", verbose=False)
    X_train = b.train_X
    X_val = b.val_X
    X_test = b.test_X
    y_val = b.val_label.astype(int)
    y_test = b.test_label.astype(int)
    if args.smoke:
        X_val = X_val[:20000]
        X_test = X_test[:20000]
        y_val = y_val[:20000]
        y_test = y_test[:20000]
    X_normal_full = X_train
    X_normal_tail = X_train[-NORMAL_TAIL:]
    b.train_df = b.val_df = b.test_df = None
    print(f"[peers] train={len(X_train):,} val={len(X_val):,} "
          f"test={len(X_test):,} normal_tail={len(X_normal_tail):,}")

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    wanted = set(args.methods) if args.methods else None

    def should(name):
        return wanted is None or name in wanted

    # ---- 纯 NumPy / sklearn 方法 ----
    if should("IsolationForest"):
        from sklearn.ensemble import IsolationForest
        t0 = time.time()
        m = IsolationForest(n_estimators=100, max_samples="auto",
                            contamination="auto", max_features=1.0,
                            bootstrap=False, random_state=42, n_jobs=-1)
        m.fit(X_train)
        fit_s = time.time() - t0
        t0 = time.time()
        sn = -m.score_samples(X_normal_full)
        sv = -m.score_samples(X_val)
        st = -m.score_samples(X_test)
        score_s = time.time() - t0
        rows += _evaluate("isolation", "IsolationForest",
                          sn, sv, y_val, st, y_test,
                          fit_s, score_s, len(X_train),
                          "sklearn IF, full Normal")
        np.savez_compressed(
            SCORE_DIR / "IsolationForest.npz",
            normal=sn, val=sv, test=st,
            y_val=y_val, y_test=y_test)
        print(f"[peers] IsolationForest done fit={fit_s:.1f}s "
              f"score={score_s:.1f}s")

    if should("RRCF"):
        t0 = time.time()
        sn, sv, st = _rrcf_scores(X_train, X_normal_full, X_val, X_test,
                                  args.smoke)
        fit_s = time.time() - t0
        rows += _evaluate("isolation", "RRCF(batch RCT)",
                          sn, sv, y_val, st, y_test,
                          fit_s, fit_s, len(X_train),
                          "batch random-cut depth approximation")
        np.savez_compressed(SCORE_DIR / "RRCF.npz",
                            normal=sn, val=sv, test=st,
                            y_val=y_val, y_test=y_test)
        print(f"[peers] RRCF done {fit_s:.1f}s")

    if should("HST"):
        t0 = time.time()
        sn, sv, st = _hst_scores(X_train, X_normal_full, X_val, X_test,
                                 args.smoke)
        fit_s = time.time() - t0
        rows += _evaluate("isolation", "HST(batch)",
                          sn, sv, y_val, st, y_test,
                          fit_s, fit_s, len(X_train),
                          "batch half-space trees approximation")
        np.savez_compressed(SCORE_DIR / "HST.npz",
                            normal=sn, val=sv, test=st,
                            y_val=y_val, y_test=y_test)
        print(f"[peers] HST done {fit_s:.1f}s")

    # ---- INNE / INNE-PCA10 ----
    from pyod.models.inne import INNE
    for name in ["INNE", "INNE-PCA10"]:
        if not should(name):
            continue
        t0 = time.time()
        fit_n = 5000 if args.smoke else 50000
        Xf, nfit = _subsample(X_train, fit_n)
        if name == "INNE-PCA10":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=10, random_state=42).fit(Xf)
            Xf_p = pca.transform(Xf)
            Xn_p = pca.transform(X_normal_tail)
            Xv_p = pca.transform(X_val)
            Xt_p = pca.transform(X_test)
            model = INNE(n_estimators=50, max_samples=256,
                         contamination=0.01, random_state=42)
            model.fit(Xf_p)
        else:
            Xn_p, Xv_p, Xt_p = X_normal_tail, X_val, X_test
            model = INNE(n_estimators=50, max_samples=256,
                         contamination=0.01, random_state=42)
            model.fit(Xf)
        fit_s = time.time() - t0
        t0 = time.time()
        sn = model.decision_function(Xn_p)
        sv = model.decision_function(Xv_p)
        st = model.decision_function(Xt_p)
        score_s = time.time() - t0
        note = ("INNE 45-D raw" if name == "INNE"
                else "INNE on PCA-10")
        rows += _evaluate("isolation", name, sn, sv, y_val, st, y_test,
                          fit_s, score_s, nfit, note)
        np.savez_compressed(SCORE_DIR / f"{name}.npz",
                            normal=sn, val=sv, test=st,
                            y_val=y_val, y_test=y_test)
        print(f"[peers] {name} done fit={fit_s:.1f}s "
              f"score={score_s:.1f}s")

    # ---- 其它 PyOD 方法 ----
    pyod_names = ["DIF", "LOF", "HBOS", "OCSVM", "DeepSVDD",
                  "ECOD", "COPOD"]
    for name in pyod_names:
        if not should(name):
            continue
        ctor, fit_n, note = _make_pyod(name, device, args.smoke)
        t0 = time.time()
        sn, sv, st, fit_s, score_s, nfit, model = _fit_pyod(
            name, ctor, X_train, X_normal_full, X_val, X_test,
            fit_n, args.smoke)
        total_s = time.time() - t0
        family = {"LOF": "density", "HBOS": "density",
                  "OCSVM": "boundary", "DeepSVDD": "boundary",
                  "ECOD": "distribution", "COPOD": "distribution",
                  "DIF": "isolation"}[name]
        rows += _evaluate(family, name, sn, sv, y_val, st, y_test,
                          fit_s, score_s, nfit, note)
        np.savez_compressed(SCORE_DIR / f"{name}.npz",
                            normal=sn, val=sv, test=st,
                            y_val=y_val, y_test=y_test)
        print(f"[peers] {name} done fit={fit_s:.1f}s "
              f"score={score_s:.1f}s total={total_s:.1f}s")

    if not rows:
        raise SystemExit("没有运行任何方法")

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    # ---------------- Markdown ----------------
    lines = [
        "# Isolation 家族与相邻流派基准（SWaT）", "",
        "- 训练：Normal only；验证：Attack 前 20%；测试：Attack 后 80% 冻结",
        "- 12 步窗口；StandardScaler；指标含 Precision/Recall/F1/FPR/AUC/PA-F1",
        "- 为与深度模型（seq_len=60）对齐，验证/测试窗口丢弃前 60 个",
        "- `Normal99` 默认用整个 Normal 段 99% 分位；INNE 用最后 5 万点（见 note）；"
        "`ValFPR1` 用验证集正常分数 99% 分位",
        "- `Oracle` 使用测试标签，仅作上限",
        "- RRCF/HST 为批量随机切分树近似；INNE/DIF/DeepSVDD 用 50k、"
        "LOF 20k、OCSVM 10k Normal 子样本，见 note 列", "",
    ]
    families = ["isolation", "density", "boundary", "distribution"]
    for fam in families:
        sub = out[out.family == fam]
        if sub.empty:
            continue
        lines += [f"## {fam}", "",
                  "| model | method | Precision | Recall | F1 | FPR | AUC | "
                  "PR_AUC | PA_F1 | note |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.sort_values(["F1"], ascending=False).iterrows():
            lines.append(
                f"| {r['model']} | {r['threshold_source']} | "
                f"{r['Precision']:.4f} | {r['Recall']:.4f} | "
                f"{r['F1']:.4f} | {r['FPR']:.4f} | {r['AUC']:.4f} | "
                f"{r['PR_AUC']:.4f} | {r['PA_F1']:.4f} | {r['note']} |")
        lines.append("")

    # 各方法 ValFPR1 对比
    lines += ["## ValFPR1 汇总（按 F1 排序）", "",
              "| model | Precision | Recall | F1 | PA_F1 | AUC |",
              "|---|---|---|---|---|---|"]
    v = out[out.threshold_source == "ValFPR1"].sort_values(
        "F1", ascending=False)
    for _, r in v.iterrows():
        lines.append(
            f"| {r['model']} | {r['Precision']:.4f} | {r['Recall']:.4f} | "
            f"{r['F1']:.4f} | {r['PA_F1']:.4f} | {r['AUC']:.4f} |")
    lines.append("")

    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")

    print("\n===== ValFPR1 summary =====")
    print(v[["family", "model", "Precision", "Recall", "F1",
             "PA_F1", "AUC"]].to_string(index=False,
                                        float_format=lambda x: f"{x:.4f}"))
    print(f"\n已保存: {args.out_csv}")
    print(f"已保存: {args.out_md}")


if __name__ == "__main__":
    main()
