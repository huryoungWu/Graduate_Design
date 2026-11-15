# -*- coding: utf-8 -*-
"""把 Normal 99% 阈值 + 全 Attack 协议扩展到目录内其它预测/重构模型。

现有项目模型（seq_len=60）的 checkpoint 只覆盖 SWaT；本脚本：
1. 加载 `runs/<model>_swat/model.pt`；
2. 在 Normal 末尾 50k 个样本上打分，取 12 步聚合分数的 99% 分位；
3. 从已保存的 val/test artifact 拼接出整个 Attack 段的逐点分数；
4. 用同一个 Normal 阈值在全 Attack 段上评估 12 步窗口级
   P/R/F1/FPR、AUC/PR-AUC 和逐点 PA-F1。

- 预测模型：逐点分数 = (pred - true)^2 的通道均值；
- 重构模型：逐点分数 = 窗口最后时刻的 ts_sq_ts；
- 两者的逐点分数再做 12 步滑动平均，得到与前面五个模型
  （window=12）可比的窗口级分数。

注意：
- Normal 阈值是在训练用 Normal 上算的（项目模型训练使用了全部
  Normal），存在 in-sample 偏差；
- 每个模型输入窗口长度是 60，但评估窗口是 12；
- Attack 分数来自 val+test artifact 拼接，跨界处用 ffill/bfill。

用法::

    python -m models.classical.normal99_project_models
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import gc
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from models.common.config import resolve_device
from models.common.data import (ForecastDataset, ReconstructionDataset,
                                load_swat)
from models.reconstruction.usad_extended_metrics import point_adjust, prf

ROOT = Path(__file__).resolve().parents[2]
N_NORMAL = 50000
EVAL_WIN = 12
OUT_DIR = ROOT / "runs" / "normal99_project_models"

FORECAST = [
    ("LSTM", "lstm_swat", "models.forecasting.lstm"),
    ("Transformer", "transformer_swat", "models.forecasting.transformer"),
    ("iTransformer", "itransformer_swat", "models.forecasting.itransformer"),
    ("TCN", "tcn_swat", "models.forecasting.tcn"),
    ("TimesNet", "timesnet_swat", "models.forecasting.timesnet"),
    ("MICN", "micn_swat", "models.forecasting.micn"),
    ("ModernTCN", "moderntcn_swat", "models.forecasting.moderntcn"),
    ("Seq2Seq-LSTM", "seq2seq_lstm_swat",
     "models.forecasting.seq2seq_lstm"),
]
RECON = [
    ("AE", "ae_swat", "models.reconstruction.ae"),
    ("VAE", "vae_swat", "models.reconstruction.vae"),
    ("AnomalyTransformer", "anomalytransformer_swat",
     "models.reconstruction.anomaly_transformer"),
]


def _first_step(a):
    return a[:, 0, :] if a.ndim == 3 else a


def _sliding_mean(x, w):
    x = np.asarray(x, dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(x, w).mean(axis=1)


def _sliding_max(x, w):
    x = np.asarray(x, dtype=np.int8)
    return np.lib.stride_tricks.sliding_window_view(x, w).max(axis=1)


def _attack_frame(csv_path: str):
    df = pd.read_csv(csv_path, usecols=["timestamp", "label", "period"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    attack = df[df["period"].astype(str).str.lower() == "attack"].copy()
    attack = attack.reset_index(drop=True)
    attack["row"] = np.arange(len(attack))
    return attack


def _artifact_point_scores(art_dir: Path, kind: str,
                           row_of_ts: dict, n_rows: int):
    """拼接 val+test artifact 的逐点分数，返回长度 n_rows 的序列。"""
    ps = np.full(n_rows, np.nan, dtype=np.float64)
    for split in ("val", "test"):
        path = art_dir / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as z:
            ts = pd.to_datetime(z["timestamp"])
            if kind == "forecast":
                pred = _first_step(z["pred"])
                true = _first_step(z["true"])
                score = ((pred - true) ** 2).mean(axis=1)
            else:
                score = z["ts_sq_ts"][:, -1]
        for t, v in zip(ts, score):
            r = row_of_ts.get(pd.Timestamp(t))
            if r is not None:
                ps[r] = float(v)
    s = pd.Series(ps)
    s = s.ffill().bfill()
    return s.to_numpy()


def _normal_point_scores(model, X, meta, kind, device, batch=256):
    L, P, H = meta["seq_len"], meta["pred_len"], meta["horizon"]
    if kind == "forecast":
        # 取末尾 N_NORMAL 个预测样本
        start = max(0, len(X) - N_NORMAL - L - H)
        ds = ForecastDataset(X[start:], np.zeros(len(X) - start, np.int8),
                             L, P, H)
        loader = DataLoader(ds, batch_size=batch, shuffle=False,
                            num_workers=0)
        parts = []
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                err = _first_step(pred) - _first_step(y)
                parts.append((err ** 2).mean(dim=1).cpu().numpy())
        return np.concatenate(parts).astype(np.float64)

    start = max(0, len(X) - N_NORMAL - L + 1)
    ds = ReconstructionDataset(X[start:], np.zeros(len(X) - start, np.int8),
                               L)
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=0)
    parts = []
    with torch.no_grad():
        for x, _ in loader:
            out = model.score_batch(x.to(device, non_blocking=True))
            parts.append(out["ts"]["sq_ts"][:, -1].cpu().numpy())
    return np.concatenate(parts).astype(np.float64)


def _evaluate_methods(name, kind, dataset, normal_ps, attack_ps, attack_y):
    nw = _sliding_mean(normal_ps, EVAL_WIN)
    ws = _sliding_mean(attack_ps, EVAL_WIN)
    wy = _sliding_max(attack_y, EVAL_WIN)
    n = min(len(ws), len(wy))
    ws, wy = ws[:n], wy[:n]
    auc = (float(roc_auc_score(wy, ws))
           if len(np.unique(wy)) > 1 else float("nan"))
    ap = (float(average_precision_score(wy, ws))
          if len(np.unique(wy)) > 1 else float("nan"))

    methods = [
        ("q90_all", float(np.quantile(nw, 0.90))),
        ("q95_all", float(np.quantile(nw, 0.95))),
        ("q99_all", float(np.quantile(nw, 0.99))),
        ("q99.9_all", float(np.quantile(nw, 0.999))),
        # 部署流报警预算：不使用标签
        ("alarm5", float(np.quantile(ws, 0.95))),
        ("alarm5.86", float(np.quantile(ws, 1.0 - 0.0586))),
        ("alarm9.28", float(np.quantile(ws, 1.0 - 0.0928))),
        ("alarm10", float(np.quantile(ws, 0.90))),
        ("alarm12", float(np.quantile(ws, 0.88))),
        ("alarm15", float(np.quantile(ws, 0.85))),
    ]
    rows = []
    for method, thr in methods:
        pred = (ws >= thr).astype(np.int8)
        p, r, f1 = prf(wy, pred)
        tp = int(((pred == 1) & (wy == 1)).sum())
        fp = int(((pred == 1) & (wy == 0)).sum())
        fn = int(((pred == 0) & (wy == 1)).sum())
        tn = int(((pred == 0) & (wy == 0)).sum())
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        point_pred = (attack_ps >= thr).astype(np.int8)
        pa = point_adjust(attack_y, point_pred)
        p_pa, r_pa, f1_pa = prf(attack_y, pa)
        rows.append({
            "model": name, "type": kind, "dataset": dataset,
            "protocol": "calibrated threshold + full Attack (12-step eval)",
            "method": method,
            "seq_len": None,  # filled by caller
            "threshold": thr,
            "normal_samples": int(len(nw)),
            "attack_windows": int(n),
            "AUC": auc, "PR_AUC": ap,
            "Precision": p, "Recall": r, "F1": f1, "FPR": fpr,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "PA_F1_fixed_threshold": f1_pa,
            "PA_Precision": p_pa, "PA_Recall": r_pa,
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="项目内预测/重构模型的 Normal99 + 全 Attack 扩展")
    p.add_argument("--csv", default=str(ROOT / "08_SWaT/SWaT_all_1s.csv"))
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--models", nargs="*", default=None,
                   help="只运行指定模型名，例如 LSTM AE")
    args = p.parse_args()

    device = resolve_device(args.device)
    csv_path = args.csv
    bundle = load_swat(csv_path, val_frac=0.2, scaler="standard")
    attack = _attack_frame(csv_path)
    attack_y = attack["label"].to_numpy(np.int8)
    row_of_ts = {t: int(r) for t, r in
                 zip(attack["timestamp"], attack["row"])}
    print(f"[project-N99] device={device} | Normal windows="
          f"{len(bundle.train_X):,} | Attack rows={len(attack):,}")

    rows = []
    specs = [(name, prefix, mod, "forecast") for name, prefix, mod in FORECAST]
    specs += [(name, prefix, mod, "reconstruction")
              for name, prefix, mod in RECON]
    if args.models:
        wanted = set(args.models)
        specs = [s for s in specs if s[0] in wanted]

    for name, prefix, mod_name, kind in specs:
        art = ROOT / "runs" / prefix
        ckpt = art / "model.pt"
        meta_path = art / "meta.json"
        if not (ckpt.exists() and meta_path.exists()):
            print(f"[skip] {name}: missing checkpoint/meta")
            continue
        meta = json.load(open(meta_path, encoding="utf-8"))
        L = int(meta["seq_len"])
        P = int(meta.get("pred_len", 1))
        H = int(meta.get("horizon", 1))
        n_channels = len(meta["channels"])
        meta["pred_len"], meta["horizon"] = P, H

        print(f"\n[{name}] {kind} seq={L} pred={P} h={H} "
              f"channels={n_channels} ...", flush=True)
        try:
            mod = importlib.import_module(mod_name)
            if kind == "forecast":
                model = mod.build_model(n_channels, L, P, None)
            else:
                model = mod.build_model(n_channels, L, None)
            state = torch.load(ckpt, map_location=device,
                               weights_only=True)
            model.load_state_dict(state)
            model.to(device).eval()

            normal_ps = _normal_point_scores(
                model, bundle.train_X, meta, kind, device,
                batch=args.batch_size)
            attack_ps = _artifact_point_scores(
                art, kind, row_of_ts, len(attack_y))
            model_rows = _evaluate_methods(
                name, kind, "SWaT", normal_ps, attack_ps, attack_y)
            for row in model_rows:
                row["seq_len"] = L
            rows.extend(model_rows)
            f1s = " | ".join(
                f"{r['method']}={r['F1']:.3f}" for r in model_rows)
            print(f"  AUC={model_rows[0]['AUC']:.4f} | {f1s}",
                  flush=True)
        except Exception as exc:
            print(f"  [ERROR] {name}: {exc}", flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "comparison_methods.csv", index=False,
              encoding="utf-8-sig")
    df[df["method"] == "q99_all"].to_csv(
        OUT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    if rows:
        show = ["model", "type", "seq_len", "method", "threshold",
                "AUC", "PR_AUC", "Precision", "Recall", "F1", "FPR",
                "PA_F1_fixed_threshold"]
        print("\n=== project models: calibrated threshold + full Attack (SWaT) ===")
        print(df[show].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n已保存: {OUT_DIR / 'comparison_methods.csv'}")


if __name__ == "__main__":
    main()
