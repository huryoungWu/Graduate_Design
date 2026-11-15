# -*- coding: utf-8 -*-
"""Anomaly-Transformer 复现：SWaT / WADI 训练与测试。

模型、损失、阈值和 point-adjust 评价均按 thuml/Anomaly-Transformer
（以及 Manami-GIWA fork）的实现逻辑移植；数据部分使用本地 CSV 预处理
后的 .npy（见 prepare_data.py）。

用法::

    python -m repro_anomaly_transformer.run --dataset swat
    python -m repro_anomaly_transformer.run --dataset wadi
    python -m repro_anomaly_transformer.run --dataset swat --smoke
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_fscore_support,
                             roc_auc_score)
from torch.utils.data import DataLoader

from .dataset import WindowDataset, load_prepared
from .vendor.AnomalyTransformer import AnomalyTransformer
from models.common.protocol import select_threshold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_DIR = {
    "swat": ROOT / "repro_anomaly_transformer" / "data" / "swat",
    "wadi": ROOT / "repro_anomaly_transformer" / "data" / "wadi",
}


# ---------------------------------------------------------------------------
# 原仓库损失与工具
# ---------------------------------------------------------------------------
def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def norm_prior(prior):
    return prior / prior.sum(dim=-1, keepdim=True)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Updating learning rate to {lr}")


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name="", delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif (score < self.best_score + self.delta
              or score2 < self.best_score2 + self.delta):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> "
                  f"{val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(),
                   Path(path) / f"{self.dataset}_checkpoint.pth")
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


# ---------------------------------------------------------------------------
# 复现 Solver
# ---------------------------------------------------------------------------
class Solver:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if args.device == "auto" and torch.cuda.is_available()
            else ("cpu" if args.device == "auto" else args.device))
        print(f"device={self.device}", flush=True)

        data = load_prepared(args.data_dir)
        train_x = data["train_X"]
        test_x = data["test_X"]
        test_y = data["test_y"]
        self.test_timestamp = data["test_timestamp"]

        if args.smoke:
            n_tr = min(len(train_x), args.smoke_rows)
            n_te = min(len(test_x), args.smoke_rows)
            train_x, test_x, test_y = (train_x[:n_tr], test_x[:n_te],
                                       test_y[:n_te])
            print(f"[smoke] train={n_tr:,} test={n_te:,}", flush=True)

        self.win_size = args.win_size
        self.input_c = int(train_x.shape[1])
        self.output_c = self.input_c
        self.batch_size = args.batch_size
        self.clip_value = (args.clip_scaled if args.clip_scaled > 0 else None)
        self.criterion = nn.MSELoss()
        if self.clip_value:
            print(f"[data] clip scaled windows to "
                  f"[-{self.clip_value}, {self.clip_value}]", flush=True)

        # 官方模式：用 test_loader 做 early stopping（无标签，仅输入）。
        # normal：用 Normal 最后 20% 做验证，训练只用前 80% Normal。
        # attack_val：Attack 前 20% 验证 + 选阈值，后 80% 冻结测试。
        self.val_thre_loader = None
        if args.val_mode == "attack_val":
            n_val = int(len(test_x) * args.val_frac)
            val_x, val_y = test_x[:n_val], test_y[:n_val]
            test_x, test_y = test_x[n_val:], test_y[n_val:]
            self.test_timestamp = self.test_timestamp[n_val:]
            self.train_loader = self._loader(
                train_x, mode="train", shuffle=True)
            self.val_loader = self._loader(
                val_x, val_y, mode="val", shuffle=False)
            self.val_thre_loader = self._loader(
                val_x, val_y, mode="thre", shuffle=False)
            self.energy_train_X = train_x
            print(f"[split] Attack val={len(val_x):,} | "
                  f"frozen test={len(test_x):,} | val_frac={args.val_frac}",
                  flush=True)
        elif args.val_mode == "normal":
            n_val = int(len(train_x) * 0.2)
            self.train_loader = self._loader(
                train_x[:-n_val], mode="train", shuffle=True)
            self.val_loader = self._loader(
                train_x[-n_val:], mode="val", shuffle=False)
            self.energy_train_X = train_x[:-n_val]
        else:
            self.train_loader = self._loader(
                train_x, mode="train", shuffle=True)
            self.val_loader = self._loader(
                test_x, test_y, mode="test", shuffle=False)
            self.energy_train_X = train_x

        self.test_loader = self._loader(
            test_x, test_y, mode="test", shuffle=False)
        self.thre_loader = self._loader(
            test_x, test_y, mode="thre", shuffle=False)
        self.test_y = test_y

        self.build_model()
        print(f"train windows={len(self.train_loader.dataset):,} | "
              f"test windows={len(self.test_loader.dataset):,} | "
              f"thre windows={len(self.thre_loader.dataset):,} | "
              f"input_c={self.input_c}", flush=True)

    def _loader(self, X, y=None, mode="train", shuffle=False):
        step = 1 if mode in ("train", "val", "test") else self.args.win_size
        ds = WindowDataset(X, y, win_size=self.args.win_size,
                           step=step, mode=mode,
                           clip_value=self.clip_value)
        return DataLoader(ds, batch_size=self.args.batch_size,
                          shuffle=shuffle, num_workers=0,
                          pin_memory=True, drop_last=False)

    def build_model(self):
        self.model = AnomalyTransformer(
            win_size=self.args.win_size, enc_in=self.input_c,
            c_out=self.output_c, d_model=self.args.d_model,
            n_heads=self.args.n_heads, e_layers=self.args.e_layers,
            d_ff=self.args.d_ff, dropout=self.args.dropout,
            activation="gelu", output_attention=True)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.model.to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"model params={n_params:,}", flush=True)

    def _vali(self, loader):
        self.model.eval()
        loss_1, loss_2 = [], []
        with torch.no_grad():
            for input_data, _ in loader:
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    p = norm_prior(prior[u])
                    series_loss += (
                        torch.mean(my_kl_loss(series[u], p.detach()))
                        + torch.mean(my_kl_loss(p.detach(), series[u])))
                    prior_loss += (
                        torch.mean(my_kl_loss(p, series[u].detach()))
                        + torch.mean(my_kl_loss(series[u].detach(), p)))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                rec_loss = self.criterion(output, input)
                loss_1.append((rec_loss - self.args.k * series_loss).item())
                loss_2.append((rec_loss + self.args.k * prior_loss).item())
        return float(np.average(loss_1)), float(np.average(loss_2))

    def train(self):
        print("======================TRAIN MODE======================",
              flush=True)
        path = Path(self.args.out_dir)
        path.mkdir(parents=True, exist_ok=True)
        early = EarlyStopping(patience=self.args.patience, verbose=True,
                              dataset_name=self.args.dataset)
        train_steps = len(self.train_loader)
        t_start = time.time()
        for epoch in range(self.args.epochs):
            epoch_time = time.time()
            self.model.train()
            loss1_list = []
            iter_count = 0
            time_now = time.time()
            for i, (input_data, _) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    p = norm_prior(prior[u])
                    series_loss += (
                        torch.mean(my_kl_loss(series[u], p.detach()))
                        + torch.mean(my_kl_loss(p.detach(), series[u])))
                    prior_loss += (
                        torch.mean(my_kl_loss(p, series[u].detach()))
                        + torch.mean(my_kl_loss(series[u].detach(), p)))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                rec_loss = self.criterion(output, input)

                loss1_list.append(
                    (rec_loss - self.args.k * series_loss).item())
                loss1 = rec_loss - self.args.k * series_loss
                loss2 = rec_loss + self.args.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left = speed * ((self.args.epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: "
                          f"{left:.1f}s", flush=True)
                    iter_count = 0
                    time_now = time.time()

                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print(f"Epoch: {epoch + 1} cost time: "
                  f"{time.time() - epoch_time:.1f}s | "
                  f"Train Loss: {np.average(loss1_list):.7f}", flush=True)
            val1, val2 = self._vali(self.val_loader)
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                  f"Train Loss: {np.average(loss1_list):.7f} | "
                  f"Vali Loss: {val1:.7f}", flush=True)
            early(val1, val2, self.model, path)
            if early.early_stop:
                print("Early stopping", flush=True)
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.args.lr)
        print(f"训练总耗时: {time.time() - t_start:.1f}s", flush=True)

    @torch.no_grad()
    def _energy(self, loader):
        self.model.eval()
        criterion = nn.MSELoss(reduction="none")
        energy = []
        labels = []
        temperature = self.args.temperature
        for input_data, label in loader:
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                p = norm_prior(prior[u])
                series_loss += my_kl_loss(
                    series[u], p.detach()) * temperature
                prior_loss += my_kl_loss(
                    p, series[u].detach()) * temperature
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            energy.append(cri.detach().cpu().numpy())
            labels.append(label.numpy())
        energy = np.concatenate(energy, axis=0).reshape(-1)
        labels = np.concatenate(labels, axis=0).reshape(-1)
        return energy, labels

    @staticmethod
    def _point_adjust(gt, pred):
        """原仓库 test() 里的 detection adjustment，逐段扩展。"""
        gt = np.asarray(gt).astype(int)
        pred = np.asarray(pred).astype(int).copy()
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1
        return pred

    def test(self):
        print("======================TEST MODE======================",
              flush=True)
        ckpt = Path(self.args.out_dir) / f"{self.args.dataset}_checkpoint.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"找不到 checkpoint: {ckpt}")
        self.model.load_state_dict(
            torch.load(ckpt, map_location=self.device, weights_only=True))
        self.model.eval()

        test_energy, test_labels = self._energy(self.thre_loader)
        n = min(len(test_energy), len(test_labels))
        test_energy, test_labels = test_energy[:n], test_labels[:n]

        val_energy = None
        train_energy = None
        val_f1 = float("nan")
        if self.val_thre_loader is not None:
            # 20/80 协议：阈值只在 Attack 前 20% 验证集上选。
            val_energy, val_labels = self._energy(self.val_thre_loader)
            nv = min(len(val_energy), len(val_labels))
            val_energy, val_labels = val_energy[:nv], val_labels[:nv]
            thresh, val_f1 = select_threshold(val_energy, val_labels)
            threshold_mode = "val_best_f1"
            print(f"[threshold] val_best_f1={thresh} | "
                  f"val_F1={val_f1:.4f}", flush=True)
        else:
            # 官方原始协议：train+test 拼接后取分位数。
            train_energy, _ = self._energy(self.train_loader)
            combined = np.concatenate([train_energy, test_energy], axis=0)
            thresh = float(np.percentile(
                combined, 100 - self.args.anomaly_ratio))
            threshold_mode = "official_combined"
            print(f"Threshold : {thresh}", flush=True)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)
        pred_adj = self._point_adjust(gt, pred)

        def _prf(y, p):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, p, average="binary", zero_division=0)
            return float(precision), float(recall), float(f1)

        p0, r0, f0 = _prf(gt, pred)
        p1, r1, f1 = _prf(gt, pred_adj)
        acc0 = float(accuracy_score(gt, pred))
        acc1 = float(accuracy_score(gt, pred_adj))
        auc = (float(roc_auc_score(gt, test_energy))
               if len(np.unique(gt)) > 1 else float("nan"))
        ap = (float(average_precision_score(gt, test_energy))
              if len(np.unique(gt)) > 1 else float("nan"))

        tp = int(((pred_adj == 1) & (gt == 1)).sum())
        fp = int(((pred_adj == 1) & (gt == 0)).sum())
        tn = int(((pred_adj == 0) & (gt == 0)).sum())
        fn = int(((pred_adj == 0) & (gt == 1)).sum())

        print(f"Accuracy : {acc1:0.4f}, Precision : {p1:0.4f}, "
              f"Recall : {r1:0.4f}, F-score : {f1:0.4f}", flush=True)
        print(f"(unadjusted) P={p0:.4f} R={r0:.4f} F1={f0:.4f} | "
              f"AUC={auc:.4f} PR-AUC={ap:.4f}", flush=True)

        out_dir = Path(self.args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if train_energy is not None:
            np.save(out_dir / "train_energy.npy", train_energy)
        if val_energy is not None:
            np.save(out_dir / "val_energy.npy", val_energy)
        np.save(out_dir / "test_energy.npy", test_energy)

        ts = self.test_timestamp[:n]
        import pandas as pd
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(ts).astype(str),
            "label": gt,
            "score": test_energy,
            "pred": pred,
            "pred_adjusted": pred_adj,
            "tp": ((pred_adj == 1) & (gt == 1)).astype(int),
            "fp": ((pred_adj == 1) & (gt == 0)).astype(int),
            "tn": ((pred_adj == 0) & (gt == 0)).astype(int),
            "fn": ((pred_adj == 0) & (gt == 1)).astype(int),
        })
        df.to_csv(out_dir / "test_scores.csv", index=False,
                  encoding="utf-8-sig")

        protocol = ("20/80 frozen test (val_best_f1)"
                    if self.args.val_mode == "attack_val"
                    else "Anomaly-Transformer official (PA evaluation)")
        metrics = {
            "dataset": self.args.dataset,
            "data_dir": str(self.args.data_dir),
            "protocol": protocol,
            "train_data": "Normal only",
            "test_used_in_training": False,
            "model": "AnomalyTransformer",
            "input_c": self.input_c,
            "win_size": self.args.win_size,
            "step": 1,
            "clip_scaled": self.args.clip_scaled,
            "batch_size": self.args.batch_size,
            "epochs": self.args.epochs,
            "lr": self.args.lr,
            "k": self.args.k,
            "d_model": self.args.d_model,
            "n_heads": self.args.n_heads,
            "e_layers": self.args.e_layers,
            "d_ff": self.args.d_ff,
            "dropout": self.args.dropout,
            "anomaly_ratio": self.args.anomaly_ratio,
            "temperature": self.args.temperature,
            "val_mode": self.args.val_mode,
            "val_frac": self.args.val_frac,
            "threshold_mode": threshold_mode,
            "val_F1": val_f1,
            "seed": self.args.seed,
            "threshold": thresh,
            "n_train_energy": (0 if train_energy is None
                               else int(len(train_energy))),
            "n_test_points": int(n),
            "positives": int(gt.sum()),
            "accuracy": acc1,
            "precision": p1,
            "recall": r1,
            "f1": f1,
            "accuracy_unadjusted": acc0,
            "precision_unadjusted": p0,
            "recall_unadjusted": r0,
            "f1_unadjusted": f0,
            "AUC": auc,
            "PR_AUC": ap,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        }
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        self._plots(df, thresh, out_dir, f1)
        print(f"结果已保存: {out_dir}", flush=True)
        return metrics

    def _plots(self, df, thresh, out_dir, f1):
        if self.args.no_plot:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            # 全测试集概览
            n = len(df)
            stride = max(1, n // 20000)
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(np.arange(0, n, stride), df["score"].to_numpy()[::stride],
                    color="purple", linewidth=0.8, label="score")
            ax.axhline(thresh, color="gray", linestyle="--",
                       linewidth=0.8, label=f"thr={thresh:.6f}")
            ax.set_title(f"[AnomalyTransformer] {self.args.dataset.upper()} "
                         f"test score (PA F1={f1:.4f})")
            ax.set_xlabel("point index")
            ax.set_ylabel("score")
            ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / "test_score_overview.png", dpi=120)
            plt.close(fig)

            # 按天四色背景图（复用项目已有绘图）
            from models.common.plots import plot_score_daily
            plot_score_daily(
                timestamps=df["timestamp"].to_numpy(),
                score=df["score"].to_numpy(),
                labels=df["label"].to_numpy(),
                tp=df["tp"].to_numpy(), fp=df["fp"].to_numpy(),
                fn=df["fn"].to_numpy(), tn=df["tn"].to_numpy(),
                threshold=thresh,
                out_dir=str(out_dir / "plots" / "daily"),
                model_name="AnomalyTransformer",
                score_name="association-weighted energy",
            )
        except Exception as exc:
            print(f"[plot] 跳过: {exc}", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Anomaly-Transformer 复现（SWaT/WADI）")
    p.add_argument("--dataset", choices=["swat", "wadi"], required=True)
    p.add_argument("--data-dir", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--win-size", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--anomaly-ratio", type=float, default=0.1,
                   help="阈值分位：percentile(100-anomaly_ratio)")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--e-layers", type=int, default=3)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--val-mode",
                   choices=["test", "normal", "attack_val"], default="test")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="attack_val 模式下 Attack 前 val_frac 做验证，"
                        "其余做冻结测试")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--clip-scaled", type=float, default=0.0,
                   help="WADI 建议 10：把标准化后的窗口裁剪到 [-v,v]，"
                        "抑制 2B_AIT_002_PV 这类极端值")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-rows", type=int, default=10000)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    if not args.data_dir:
        args.data_dir = str(DATA_DIR[args.dataset])
    if not args.out_dir:
        name = args.dataset + ("_smoke" if args.smoke else "")
        args.out_dir = str(
            ROOT / "repro_anomaly_transformer" / "runs" / name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    print("------------ Options -------------")
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")
    print("-------------- End ----------------")

    solver = Solver(args)
    solver.train()
    solver.test()


if __name__ == "__main__":
    main()
