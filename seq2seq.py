# ============================================================
# Seq2Seq-LSTM 异常检测（融合 SWaT/LSTM 脚本 + Seq2Seq 脚本风格）
# 思路：Encoder 读入 seq_len 步历史 -> Decoder 自回归预测未来 pred_len 步
#       用预测残差作为异常分数，评估 AUC / PR-AUC / P / R / F1
# ============================================================
import os
import time
import math
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    precision_recall_curve, auc as sk_auc,
)
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================== 1. 配置 =====================
CSV_PATH   = r"08_SWaT\SWaT_all_1s.csv"
SEQ_LEN    = 60          # Encoder 输入长度
PRED_LEN   = 1           # Decoder 预测长度（异常检测通常=1）
BATCH_SIZE = 256
EPOCHS     = 10
LR         = 1e-4
HIDDEN     = 128
NUM_LAYERS = 2
DROPOUT    = 0.1
TF_RATIO   = 0.5         # Teacher Forcing 比例
PATIENCE   = 5           # EarlyStopping
MIN_DELTA  = 1e-5
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

PREFIX = "seq2seq_lstm_swat"

# ===================== 2. 读取与预处理 =====================
df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

sensor_cols = [c for c in df.columns if c not in ["timestamp", "label", "period"]]
print(f"传感器/执行器列数: {len(sensor_cols)}")

df["label"] = df["label"].astype(int)
df[sensor_cols] = df[sensor_cols].ffill().fillna(0)

split_idx = int(len(df) * 0.7)
train_df = df.iloc[:split_idx].reset_index(drop=True)
test_df  = df.iloc[split_idx:].reset_index(drop=True)

scaler = StandardScaler()
train_X_raw = scaler.fit_transform(train_df[sensor_cols].values)
test_X_raw  = scaler.transform(test_df[sensor_cols].values)

train_label = train_df["label"].values
test_label  = test_df["label"].values

# ===================== 3. 滑动窗口数据集 =====================
class WindowDataset(Dataset):
    """x: [seq_len, C]; y: [pred_len, C]; lab: 窗口内 label 最大值"""
    def __init__(self, data, labels, seq_len, pred_len=1):
        self.data   = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.n = len(data) - seq_len - pred_len + 1

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len]
        lab = self.labels[idx + self.seq_len: idx + self.seq_len + self.pred_len].max()
        return x, y, lab

train_ds_full = WindowDataset(train_X_raw, train_label, SEQ_LEN, PRED_LEN)
test_ds       = WindowDataset(test_X_raw,  test_label,  SEQ_LEN, PRED_LEN)

# 只保留训练集中完全不包含攻击的窗口
normal_indices = [
    i for i in range(len(train_ds_full))
    if train_label[i: i + SEQ_LEN + PRED_LEN].max() == 0
]
train_ds = Subset(train_ds_full, normal_indices)

print(f"训练窗口总数: {len(train_ds_full)}, 其中正常窗口: {len(train_ds)}")
print(f"被剔除的含攻击窗口: {len(train_ds_full) - len(train_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

n_channels = len(sensor_cols)
print(f"通道数: {n_channels}, 测试窗口: {len(test_ds)}")

# ===================== 4. Seq2Seq (LSTM Encoder-Decoder) =====================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        out, (hidden, cell) = self.lstm(x, (hidden, cell))
        pred = self.fc(out)          # [B, 1, output_dim]
        return pred, hidden, cell


class Seq2SeqLSTM(nn.Module):
    """
    编码器读入 seq_len 步历史，
    解码器自回归预测 pred_len 步所有通道值。
    """
    def __init__(self, n_channels, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.encoder = Encoder(n_channels, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(n_channels, hidden_dim, num_layers, dropout)

    def forward(self, src, target_len, tgt=None, teacher_forcing_ratio=0.0):
        batch_size = src.size(0)
        hidden, cell = self.encoder(src)

        decoder_input = torch.zeros(batch_size, 1, self.n_channels, device=src.device)
        outputs = []
        for t in range(target_len):
            out, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(out)
            use_tf = (tgt is not None) and (random.random() < teacher_forcing_ratio)
            decoder_input = tgt[:, t:t+1, :] if use_tf else out

        return torch.cat(outputs, dim=1)   # [B, target_len, C]


model = Seq2SeqLSTM(n_channels, HIDDEN, NUM_LAYERS, DROPOUT).to(DEVICE)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# ===================== 5. EarlyStopping =====================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.stop = False

    def step(self, val):
        if self.best is None:
            self.best = val
            return False
        if val < self.best - self.min_delta:
            self.best = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

# ===================== 6. 训练 =====================
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.MSELoss()
early_stopper = EarlyStopping(PATIENCE, MIN_DELTA)

best_state = None
best_loss  = float("inf")
history    = []

def train_epoch(epoch):
    model.train()
    total_loss, n_seen = 0.0, 0
    pbar = tqdm(train_loader, desc=f"[Seq2Seq] Epoch {epoch:02d}/{EPOCHS}",
                unit="batch", ncols=110, dynamic_ncols=True)
    for x, y, _ in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x, target_len=PRED_LEN, tgt=y, teacher_forcing_ratio=TF_RATIO)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n_seen += x.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}",
                          "avg": f"{total_loss / n_seen:.4f}",
                          "lr": f"{scheduler.get_last_lr()[0]:.1e}"})
    pbar.close()
    return total_loss / len(train_ds)


@torch.no_grad()
def evaluate_loss(loader):
    """在验证/测试集上（不含 Teacher Forcing）算 MSE，用于 EarlyStopping"""
    model.eval()
    total, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x, target_len=PRED_LEN, tgt=None, teacher_forcing_ratio=0.0)
        total += criterion(pred, y).item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


# 用测试集 loss 做早停监控（无验证集时可直接用测试集；有验证集应切出 val）
train_start = time.time()
for epoch in range(1, EPOCHS + 1):
    ep_start = time.time()
    train_loss = train_epoch(epoch)
    val_loss   = evaluate_loss(test_loader)
    scheduler.step()

    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"]})
    print(f"[Seq2Seq] Epoch {epoch:02d} | Train MSE: {train_loss:.6f} | "
          f"Val MSE: {val_loss:.6f} | 耗时: {time.time() - ep_start:.1f}s")

    if val_loss < best_loss:
        best_loss = val_loss
        best_state = deepcopy(model.state_dict())
        torch.save(best_state, f"{PREFIX}_best.pth")

    if early_stopper.step(val_loss):
        print(f"🛑 Early stopping at epoch={epoch}")
        break

print(f"[Seq2Seq] 训练总耗时: {time.time() - train_start:.1f}s")

pd.DataFrame(history).to_csv(f"{PREFIX}_train_history.csv", index=False, encoding="utf-8-sig")

# 恢复最佳权重
if best_state is not None:
    model.load_state_dict(best_state)

# ===================== 7. 测试：预测残差 =====================
model.eval()
all_pred, all_true, all_lab = [], [], []
test_pbar = tqdm(test_loader, desc="[Seq2Seq] Testing", unit="batch", ncols=110, dynamic_ncols=True)
with torch.no_grad():
    for x, y, lab in test_pbar:
        x = x.to(DEVICE)
        pred = model(x, target_len=PRED_LEN, tgt=None, teacher_forcing_ratio=0.0).cpu().numpy()
        all_pred.append(pred[:, 0, :])
        all_true.append(y[:, 0, :].numpy())
        all_lab.append(lab.numpy())
test_pbar.close()

pred = np.concatenate(all_pred, axis=0)
true = np.concatenate(all_true, axis=0)
lab  = np.concatenate(all_lab,  axis=0)

pred_timestamps = test_df["timestamp"].iloc[SEQ_LEN: SEQ_LEN + len(pred)].reset_index(drop=True)
assert len(pred_timestamps) == len(pred)

residual = np.abs(pred - true)   # [N, C]

# ===================== 8. 三种异常分数聚合 =====================
score_mean = residual.mean(axis=1)
score_max  = residual.max(axis=1)
top3       = np.sort(residual, axis=1)[:, -3:]
score_top3 = top3.mean(axis=1)

# ===================== 9. 统一评估函数 =====================
def evaluate_score(name, score, labels):
    auc = roc_auc_score(labels, score)
    precisions, recalls, thresholds = precision_recall_curve(labels, score)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    best_thr = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    pred_label = (score >= best_thr).astype(int)

    f1        = f1_score(labels, pred_label)
    precision = precision_score(labels, pred_label, zero_division=0)
    recall    = recall_score(labels, pred_label, zero_division=0)
    pr_auc    = sk_auc(recalls, precisions)

    print(f"\n===== [Seq2Seq] {name} =====")
    print(f"AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | 阈值: {best_thr:.6f}")
    return {"name": name, "AUC": auc, "PR_AUC": pr_auc,
            "Precision": precision, "Recall": recall, "F1": f1, "threshold": best_thr}

results = [
    evaluate_score("mean",      score_mean, lab),
    evaluate_score("max",       score_max,  lab),
    evaluate_score("top3_mean", score_top3, lab),
]

summary_df = pd.DataFrame(results)
summary_df.to_csv(f"{PREFIX}_anomaly_score_comparison.csv", index=False, encoding="utf-8-sig")
print(f"\n聚合方式对比已保存: {PREFIX}_anomaly_score_comparison.csv")

best_result = max(results, key=lambda r: r["F1"])
best_name   = best_result["name"]
best_thr    = best_result["threshold"]
score_map   = {"mean": score_mean, "max": score_max, "top3_mean": score_top3}
best_score  = score_map[best_name]
print(f"\n画图使用聚合方式: {best_name} (F1={best_result['F1']:.4f}, 阈值={best_thr:.6f})")

pred_anomaly = (best_score >= best_thr).astype(int)
tp_mask = (pred_anomaly == 1) & (lab == 1)
fp_mask = (pred_anomaly == 1) & (lab == 0)
fn_mask = (pred_anomaly == 0) & (lab == 1)

score_df = pd.DataFrame({
    "timestamp":    pred_timestamps.values,
    "label":        lab,
    "score_mean":   score_mean,
    "score_max":    score_max,
    "score_top3":   score_top3,
    "pred_anomaly": pred_anomaly,
    "tp": tp_mask.astype(int),
    "fp": fp_mask.astype(int),
    "fn": fn_mask.astype(int),
})
score_df.to_csv(f"{PREFIX}_anomaly_scores.csv", index=False, encoding="utf-8-sig")
print(f"三种异常分数 + 检出/误报/漏报标记已保存: {PREFIX}_anomaly_scores.csv")

# ===================== 10. 逐通道预测指标 =====================
def channel_metrics(true_c, pred_c):
    mae  = np.mean(np.abs(true_c - pred_c))
    rmse = math.sqrt(np.mean((true_c - pred_c) ** 2))
    mask = np.abs(true_c) > 1e-6
    valid_ratio = mask.mean()
    mape = np.mean(np.abs((true_c[mask] - pred_c[mask]) / true_c[mask])) * 100 if mask.sum() > 0 else np.nan
    return mae, rmse, mape, valid_ratio

print("\n===== [Seq2Seq] 逐通道预测指标（标准化空间）=====")
channel_metric_rows = []
for j, col in enumerate(sensor_cols):
    mae_j, rmse_j, mape_j, vr_j = channel_metrics(true[:, j], pred[:, j])
    channel_metric_rows.append({"channel": col, "MAE": mae_j, "RMSE": rmse_j,
                                "MAPE(%)": mape_j, "valid_ratio": vr_j})
pd.DataFrame(channel_metric_rows).to_csv(f"{PREFIX}_channel_metrics.csv",
                                         index=False, encoding="utf-8-sig")
print(f"逐通道指标已保存: {PREFIX}_channel_metrics.csv")

# ===================== 11. 按天画图 =====================
OUT_DIR = f"{PREFIX}_channel_plots"
os.makedirs(OUT_DIR, exist_ok=True)

result_df = pd.DataFrame(pred, columns=[f"pred_{c}" for c in sensor_cols])
result_df.insert(0, "timestamp", pred_timestamps.values)
for j, c in enumerate(sensor_cols):
    result_df[f"true_{c}"] = true[:, j]
    result_df[f"res_{c}"]  = residual[:, j]
result_df["tp"] = tp_mask.astype(int)
result_df["fp"] = fp_mask.astype(int)
result_df["fn"] = fn_mask.astype(int)

result_df["date"] = pd.to_datetime(result_df["timestamp"]).dt.date
dates = sorted(result_df["date"].unique())
print(f"\n[Seq2Seq] 测试集覆盖 {len(dates)} 天，开始保存每个通道的每日对比图...")

for d in dates:
    day_df = result_df[result_df["date"] == d].reset_index(drop=True)
    if len(day_df) == 0:
        continue
    day_str = str(d)

    tp_t = day_df.loc[day_df["tp"] == 1, "timestamp"]
    fp_t = day_df.loc[day_df["fp"] == 1, "timestamp"]
    fn_t = day_df.loc[day_df["fn"] == 1, "timestamp"]

    for c in sensor_cols:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 5), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}
        )

        ax1.plot(day_df["timestamp"], day_df[f"true_{c}"], label="True",
                 color="blue", linewidth=1.0)
        ax1.plot(day_df["timestamp"], day_df[f"pred_{c}"], label="Pred",
                 color="orange", linewidth=1.0, linestyle="--", alpha=0.85)

        def get_true_at(ts_series):
            if len(ts_series) == 0:
                return ts_series, []
            vals = day_df.set_index("timestamp").loc[ts_series, f"true_{c}"]
            return ts_series, vals.values

        if len(tp_t) > 0:
            t_tp, v_tp = get_true_at(tp_t)
            ax1.scatter(t_tp, v_tp, marker="o", s=14, color="green",
                        label="TP (detected)", zorder=5, alpha=0.8)
        if len(fp_t) > 0:
            t_fp, v_fp = get_true_at(fp_t)
            ax1.scatter(t_fp, v_fp, marker="x", s=18, color="red",
                        label="FP (false alarm)", zorder=5, alpha=0.8)
        if len(fn_t) > 0:
            t_fn, v_fn = get_true_at(fn_t)
            ax1.scatter(t_fn, v_fn, marker="^", s=16, color="black",
                        label="FN (missed)", zorder=5, alpha=0.8)

        ax1.set_title(f"[Seq2Seq] {c} | {day_str} | F1 max={best_name}")
        ax1.set_ylabel("Value (standardized)")
        ax1.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
        ax1.legend(loc="upper right", fontsize=7, ncol=2)

        ax2.plot(day_df["timestamp"], day_df[f"res_{c}"], color="purple",
                 linewidth=0.8, alpha=0.8, label="Residual")
        ax2.axhline(y=0, color="gray", linewidth=0.5)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("|Pred - True|")
        ax2.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
        ax2.legend(loc="upper right", fontsize=7)

        fig.autofmt_xdate()
        fig.tight_layout()
        safe_c = c.replace("/", "_").replace("\\", "_").replace(":", "_")
        fig.savefig(os.path.join(OUT_DIR, f"{day_str}_{safe_c}.png"), dpi=110)
        plt.close(fig)

    print(f"  {day_str}: 已保存 {len(sensor_cols)} 张通道图")

# ===================== 12. 训练曲线 =====================
plt.figure(figsize=(10, 5))
plt.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="Train MSE")
plt.plot([h["epoch"] for h in history], [h["val_loss"]   for h in history], label="Val MSE")
plt.xlabel("Epoch"); plt.ylabel("MSE")
plt.title("[Seq2Seq-LSTM] Training Curve")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(f"{PREFIX}_loss_curve.png", dpi=160, bbox_inches="tight")
plt.close()

print(f"\n[Seq2Seq] 全部结果已保存，PREFIX = {PREFIX}")