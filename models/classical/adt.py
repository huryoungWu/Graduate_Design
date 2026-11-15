# -*- coding: utf-8 -*-
"""ADT：Agent-based Dynamic Thresholding（PyTorch 复现）。

参考实现：https://github.com/XueYang0130/ADT
论文：Yang, Howley & Schukat,
*Agent-based dynamic thresholding for adaptive anomaly detection using
reinforcement learning*, Neural Computing and Applications, 2024；
以及 *ADT: Time series anomaly detection for cyber-physical systems via
deep reinforcement learning*, Computers & Security, 2024.

参考仓库是 TensorFlow/Keras 实现；当前环境没有 TensorFlow，本文件用
PyTorch 复刻同一流程：

1. **AE**：窗口展平 -> Linear(w/2) -> Linear(w/4) -> latent=100 ->
   镜像解码 -> Sigmoid；MSE 训练；每窗口异常分 = 重构 MSE。
2. **DQN**：状态 = 最近 k 个窗口的
   ``[mean_score, var_score, TP_rate, FP_rate, TN_rate, FN_rate]``，
   动作 = 阈值 0/1（分数 >= 阈值判异常），奖励 = 正确 +1 / 错误 -1；
   ε-greedy + replay memory + target network。
3. **评估**：用 DQN 在窗口序列上动态切换阈值，统计 TP/TN/FP/FN、
   Precision/Recall/F1。

两种协议：

- ``paper``：复刻仓库设置——AE 用 70% Normal 窗口训练；DQN 用一段
  带标签的 Attack 窗口（SWaT 1500:2000，WADI 59000:60000）训练；
  在整个 Attack 段上评估。
- ``unified``：AE 用 Normal 训练；DQN 只用 Attack 前 20%（验证段）
  的标签训练；在 Attack 后 80% 测试段评估（项目统一协议）。

重要说明（必须与论文口径一起看）：

- 参考实现的 Env 在**推理时也使用真实标签**计算 TP/FP/TN/FN，
  作为下一条状态/奖励的一部分。因此 ADT 是一个“带在线反馈的
  动态阈值控制器”，不是纯无监督检测器；评估时存在标签反馈。
- ``paper`` 协议中 DQN 的训练片段属于最终评估的 Attack 段
  （transductive），并且仓库的 MinMaxScaler 是在 Normal+Attack
  全体上 fit 的（数据泄漏）。本复现默认保留这两点以对齐论文/
  仓库，并在报告中明确标注；``unified`` 协议去掉了训练片段泄漏。

用法::

    # 论文/仓库协议
    python -m models.classical.adt --protocol paper \
        --csv 08_SWaT/SWaT_all_1s.csv
    python -m models.classical.adt --protocol paper \
        --csv 09_WADI/WADI_all_1s.csv

    # 统一框架协议
    python -m models.classical.adt --protocol unified \
        --csv 08_SWaT/SWaT_all_1s.csv
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, Subset

from models.classical.isolation_forest import _dataset_name
from models.common.config import (PROJECT_ROOT, resolve_device, set_seed)
from models.reconstruction.usad_extended_metrics import best_f1
from models.reconstruction.usad_paper import load_periods

MODEL_NAME = "ADT"


# ----------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------
class WindowDataset(Dataset):
    """按需切分展平窗口，避免在内存中物化 [N, L*C] 大矩阵。"""

    def __init__(self, data: np.ndarray, seq_len: int, labels=None):
        self.data = torch.from_numpy(np.ascontiguousarray(data))
        self.seq_len = int(seq_len)
        self.labels = labels
        self.n = len(data) - self.seq_len + 1

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, i):
        x = self.data[i:i + self.seq_len].reshape(-1)
        if self.labels is None:
            return x
        lab = int(np.max(self.labels[i:i + self.seq_len]))
        return x, lab


def _scale_periods(normal, attack, sensor_cols, fit: str, clip: bool):
    normal_vals = normal[sensor_cols].to_numpy(np.float64)
    attack_vals = attack[sensor_cols].to_numpy(np.float64)
    scaler = MinMaxScaler()
    if fit == "all":
        scaler.fit(np.vstack([normal_vals, attack_vals]))
    else:
        scaler.fit(normal_vals)
    nx = scaler.transform(normal_vals).astype(np.float32)
    ax = scaler.transform(attack_vals).astype(np.float32)
    if clip:
        nx = np.clip(nx, 0.0, 1.0)
        ax = np.clip(ax, 0.0, 1.0)
    return nx, ax, scaler


def _window_labels(labels: np.ndarray, seq_len: int) -> np.ndarray:
    L = int(seq_len)
    return np.lib.stride_tricks.sliding_window_view(
        labels.astype(np.int8), L).max(axis=1)


# ----------------------------------------------------------------------
# AE
# ----------------------------------------------------------------------
class AE(nn.Module):
    def __init__(self, window_dim: int, latent_dim: int = 100):
        super().__init__()
        h1 = int(window_dim / 2)
        h2 = int(window_dim / 4)
        self.encoder = nn.Sequential(
            nn.Linear(window_dim, h1), nn.ReLU(True),
            nn.Linear(h1, h2), nn.ReLU(True),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.ReLU(True),
            nn.Linear(h2, h1), nn.ReLU(True),
            nn.Linear(h1, window_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_ae(model, loader, epochs, lr, device, name="ADT-AE"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.9, patience=20, min_lr=1e-20)
    start = time.time()
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total, seen = 0.0, 0
        for x in loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            recon = model(x)
            loss = F.mse_loss(recon, x)
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.size(0)
            seen += x.size(0)
        sched.step(total / max(seen, 1))
        print(f"[{name}] Epoch {epoch:03d}/{epochs} | "
              f"MSE {total / max(seen, 1):.6f}")
    return time.time() - start


@torch.no_grad()
def score_ae(model, loader, device) -> np.ndarray:
    model.eval()
    scores = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        recon = model(x)
        err = ((recon - x) ** 2).mean(dim=1)
        scores.append(err.cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


# ----------------------------------------------------------------------
# DQN
# ----------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, state_size=6, action_size=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 32), nn.ReLU(True),
            nn.Linear(32, 32), nn.ReLU(True),
            nn.Linear(32, 16), nn.ReLU(True),
            nn.Linear(16, action_size),
        )

    def forward(self, x):
        return self.net(x)


def eval_performance(tp_n, fp_n, fn_n):
    if tp_n == fp_n == fn_n == 0:
        return 1.0, 1.0, 1.0
    if tp_n == 0 and (fp_n + fn_n) > 0:
        return 0.0, 0.0, 0.0
    precision = tp_n / (tp_n + fp_n)
    recall = tp_n / (tp_n + fn_n)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return precision, recall, f1


class Env:
    """与参考实现一致的动态阈值环境（注意：推理也使用真实标签）。"""

    def __init__(self, action_set, y, score_list, verbose=False):
        self.actions = action_set
        self.y = y
        self.score_list = score_list
        self.verbose = verbose
        self.reset()

    def reset(self):
        self.t_env = 0
        self.total_reward = 0
        self.done = False
        self.state = [0, 0, 0, 0, 0, 0]
        self.tp_n = self.tn_n = self.fp_n = self.fn_n = 0
        self.result_hist = []
        return self.state

    def do_step(self, action, t_env, k_state, e=0, episodes=1):
        self.t_env = t_env
        pred_i = 1 if self.score_list[t_env] >= action else 0
        if self.y[t_env] == 0 and pred_i == 0:
            self.tn_n += 1
            self.result_hist.append("TN")
            reward = 1
        elif self.y[t_env] == 1 and pred_i == 1:
            self.tp_n += 1
            self.result_hist.append("TP")
            reward = 1
        elif self.y[t_env] == 0 and pred_i == 1:
            self.fp_n += 1
            self.result_hist.append("FP")
            reward = -1
        else:
            self.fn_n += 1
            self.result_hist.append("FN")
            reward = -1
        self.total_reward += reward

        if self.t_env < 1:
            self.state = [0, 0, 0, 0, 0, 0]
        elif 1 <= self.t_env < len(self.y) - 1:
            lo = max(0, self.t_env - k_state + 1)
            score_windows = self.score_list[lo:self.t_env + 1]
            result_windows = self.result_hist[lo:self.t_env + 1]
            n = len(score_windows)
            self.state = [
                float(np.mean(score_windows)),
                float(np.var(score_windows)),
                result_windows.count("TP") / n,
                result_windows.count("FP") / n,
                result_windows.count("TN") / n,
                result_windows.count("FN") / n,
            ]
        else:
            self.done = True
        return reward, self.state, self.done


class DQNAgent:
    def __init__(self, state_size=6, action_size=2, lr=1e-4,
                 gamma=0.99, memory=20000, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.device = torch.device(device)
        self.q_net = QNet(state_size, action_size).to(self.device)
        self.target_net = QNet(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=int(memory))

    def store(self, state, action_index, reward, next_state, done):
        self.memory.append((state, int(action_index), float(reward),
                            next_state, bool(done)))

    def policy(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            q = self.q_net(torch.tensor(
                state, dtype=torch.float32,
                device=self.device).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def train_batch(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states = torch.tensor([b[0] for b in batch], dtype=torch.float32,
                              device=self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long,
                               device=self.device)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32,
                               device=self.device)
        next_states = torch.tensor([b[3] for b in batch],
                                   dtype=torch.float32, device=self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32,
                             device=self.device)
        q = self.q_net(states)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            target = rewards + self.gamma * next_q * (1.0 - dones)
        q_sel = q.gather(1, actions[:, None]).squeeze(1)
        loss = F.mse_loss(q_sel, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# ----------------------------------------------------------------------
# DQN 训练 / 推理
# ----------------------------------------------------------------------
def train_dqn(scores, labels, episodes, batch_size, l_action, k_state,
              seed=42, verbose=True, dqn_device="cpu"):
    random.seed(seed)
    np.random.seed(seed)
    agent = DQNAgent(state_size=6, action_size=2, device=dqn_device)
    env = Env([0, 1], labels, scores, verbose=verbose)
    epsilon = 1.0
    epsilon_min = 0.001
    target_update = 100
    l_action = max(1, int(l_action))
    k_state = max(1, int(k_state))
    start = time.time()
    rewards = []
    for e in range(int(episodes)):
        state = env.reset()
        e_reward = 0.0
        action_hist = []
        for t in range(len(labels)):
            if t % l_action == 0:
                action_index = agent.policy(state, epsilon)
                action = [0, 1][action_index]
                action_hist.append(action)
            else:
                action = action_hist[-1]
            reward, next_state, done = env.do_step(
                action, t, k_state, e, episodes)
            e_reward += reward
            state = next_state
            agent.store(state, [0, 1].index(action), reward, next_state, done)
            if t == len(labels) - 1 and len(agent.memory) >= batch_size:
                agent.train_batch(batch_size)
                epsilon = max(
                    epsilon - 1.0 / (episodes * 0.99), epsilon_min)
        rewards.append(e_reward)
        if e % target_update == 0:
            agent.update_target()
        if verbose and (e == 0 or e == episodes - 1 or (e + 1) % 200 == 0):
            print(f"  episode {e + 1}/{episodes} | reward={e_reward:.0f} | "
                  f"epsilon={epsilon:.3f}")
    train_time = time.time() - start
    return agent, train_time


def _numpy_q_params(agent):
    """把 Q 网络权重导出为 numpy，用于快速逐点推理。"""
    return [p.detach().cpu().numpy().astype(np.float64)
            for p in agent.q_net.parameters()]


def _numpy_q_values(state, params):
    x = np.asarray(state, dtype=np.float64)
    n_layers = len(params) // 2
    for i in range(n_layers):
        w = params[2 * i]
        b = params[2 * i + 1]
        x = x @ w.T + b
        if i < n_layers - 1:
            x = np.maximum(x, 0.0)
    return x


@torch.no_grad()
def evaluate_dqn(agent, scores, labels, l_action=1, k_state=1):
    env = Env([0, 1], labels, scores, verbose=False)
    # 逐点推理会把 policy 调用数十万次；用 numpy 前向替代 torch 单样本
    # 前向，速度提升一个数量级，数值与原网络一致。
    params = _numpy_q_params(agent)
    action_hist = []
    for t in range(len(labels)):
        if t % max(1, int(l_action)) == 0:
            action_index = int(np.argmax(_numpy_q_values(env.state, params)))
            action = [0, 1][action_index]
            action_hist.append(action)
        else:
            action = action_hist[-1]
        reward, next_state, done = env.do_step(
            action, t, k_state, 0, 1)
        env.state = next_state
        if done:
            break
    p, r, f1 = eval_performance(env.tp_n, env.fp_n, env.fn_n)
    return {
        "Precision": float(p), "Recall": float(r), "F1": float(f1),
        "TP": int(env.tp_n), "TN": int(env.tn_n),
        "FP": int(env.fp_n), "FN": int(env.fn_n),
    }


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def _paper_segment(dataset: str, n: int):
    if dataset == "swat":
        s, e = 1500, 2000
    else:
        s, e = 59000, 60000
    s = min(s, max(0, n - 1))
    e = min(e, n)
    return s, e


def main() -> None:
    p = argparse.ArgumentParser(description="ADT PyTorch 复现（SWaT/WADI）")
    p.add_argument("--csv", required=True)
    p.add_argument("--protocol", choices=["paper", "unified"],
                   default="paper")
    p.add_argument("--seq-len", type=int, default=12)
    p.add_argument("--latent-dim", type=int, default=100)
    p.add_argument("--ae-epochs", type=int, default=30)
    p.add_argument("--ae-batch", type=int, default=8000)
    p.add_argument("--ae-lr", type=float, default=1e-3)
    p.add_argument("--ae-train-frac", type=float, default=0.7)
    p.add_argument("--scaler-fit", choices=["all", "normal"], default=None,
                   help="paper 默认 all（仓库做法）；unified 默认 normal")
    p.add_argument("--clip-input", action="store_true",
                   help="把 MinMax 后的输入裁剪到 [0,1]")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--dqn-batch", type=int, default=256)
    p.add_argument("--l-action", type=int, default=10,
                   help="训练时动作更新间隔")
    p.add_argument("--l-action-infer", type=int, default=1,
                   help="推理时动作更新间隔（README 默认 1）")
    p.add_argument("--k-state", type=int, default=1)
    p.add_argument("--dqn-lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--memory", type=int, default=20000)
    p.add_argument("--dqn-device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", default="")
    p.add_argument("--segment-start", type=int, default=-1,
                   help="覆盖 paper 协议的 DQN 训练片段起点（-1=用仓库默认）")
    p.add_argument("--segment-end", type=int, default=-1,
                   help="覆盖 paper 协议的 DQN 训练片段终点")
    p.add_argument("--stage", default="all",
                   choices=["all", "train", "detect"],
                   help="兼容 models.run_all 的批量入口；ADT 不使用")
    p.add_argument("--no-plot", action="store_true",
                   help="兼容批量入口；ADT 不绘图")
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch AE + 50 episodes，用于流程自检")
    args = p.parse_args()

    set_seed(args.seed)
    if args.smoke:
        args.ae_epochs = 1
        args.episodes = 50
        args.ae_batch = min(args.ae_batch, 2048)

    dataset = _dataset_name(args.csv)
    device = resolve_device(args.device)
    if args.protocol == "paper":
        out_dir = (Path(args.out_dir)
                   if args.out_dir and Path(args.out_dir).resolve()
                   != PROJECT_ROOT.resolve()
                   else Path("runs") / f"adt_paper_{dataset}")
        scaler_fit = args.scaler_fit or "all"
    else:
        out_dir = (Path(args.out_dir)
                   if args.out_dir and Path(args.out_dir).resolve()
                   != PROJECT_ROOT.resolve()
                   else Path("runs") / f"adt_{dataset}")
        scaler_fit = args.scaler_fit or "normal"
    if args.protocol == "unified" and scaler_fit == "normal":
        # 只用 Normal 拟合 MinMax 时，Attack 可能超出 [0,1]；
        # AE 末端是 Sigmoid，裁剪可避免无法重构的越界值。
        args.clip_input = True
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{MODEL_NAME}] protocol={args.protocol} | dataset={dataset} | "
          f"device={device} | scaler_fit={scaler_fit}")
    normal, attack, sensor_cols = load_periods(args.csv)
    print(f"[{MODEL_NAME}] Normal {len(normal):,} | "
          f"Attack {len(attack):,} | 变量 {len(sensor_cols)}")
    normal_X, attack_X, scaler = _scale_periods(
        normal, attack, sensor_cols, scaler_fit, args.clip_input)
    attack_labels = attack["label"].to_numpy(np.int8)
    attack_ts = attack["timestamp"].to_numpy()
    del normal, attack

    L = int(args.seq_len)
    window_dim = L * len(sensor_cols)
    normal_ds = WindowDataset(normal_X, L)
    attack_ds = WindowDataset(attack_X, L)
    del normal_X, attack_X

    n_train_windows = int(len(normal_ds) * args.ae_train_frac)
    train_ds = Subset(normal_ds, range(n_train_windows))
    train_loader = DataLoader(
        train_ds, batch_size=args.ae_batch, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"))
    score_loader = DataLoader(
        attack_ds, batch_size=8192, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"))

    ae = AE(window_dim, args.latent_dim).to(device)
    print(f"[{MODEL_NAME}] AE 参数量="
          f"{sum(p.numel() for p in ae.parameters()):,} | "
          f"window_dim={window_dim} | latent={args.latent_dim}")
    ae_time = train_ae(ae, train_loader, args.ae_epochs, args.ae_lr,
                       device, name="ADT-AE")
    print(f"[{MODEL_NAME}] AE 训练耗时 {ae_time:.1f}s")
    scores = score_ae(ae, score_loader, device)
    y_win = _window_labels(attack_labels, L)
    ts_win = attack_ts[L - 1:]
    print(f"[{MODEL_NAME}] attack windows={len(scores):,} | "
          f"positive rate={y_win.mean():.4f} | "
          f"score range=[{scores.min():.4g}, {scores.max():.4g}]")

    # AE-only 参考指标
    ae_auc = float(roc_auc_score(y_win, scores))
    thr_ae, f1_ae = best_f1(y_win, scores)
    print(f"[{MODEL_NAME}] AE-only: AUC {ae_auc:.4f} | "
          f"best F1 {f1_ae:.4f} @ thr {thr_ae:.6g}")

    # 选择 DQN 训练片段
    if args.protocol == "paper":
        if (args.segment_start >= 0 and args.segment_end
                > args.segment_start):
            seg_start = min(args.segment_start, len(scores) - 1)
            seg_end = min(args.segment_end, len(scores))
        else:
            seg_start, seg_end = _paper_segment(dataset, len(scores))
        eval_scores, eval_y = scores, y_win
        eval_ts = ts_win
        protocol_note = "full attack; DQN trained on a labeled slice"
    else:
        split = int(len(scores) * 0.2)
        # 与仓库规模一致的小训练片段，必须来自验证段且包含正例；
        # 取验证段第一个攻击窗口附近的一段，避免只用全正常片段训练。
        seg_len = min(500 if dataset == "swat" else 1000, max(1, split))
        val_y = y_win[:split]
        pos = np.flatnonzero(val_y == 1)
        if len(pos):
            center = int(pos[0])
            seg_start = max(0, min(center - seg_len // 2,
                                   split - seg_len))
        else:
            seg_start = 0
        seg_end = seg_start + seg_len
        eval_scores, eval_y = scores[split:], y_win[split:]
        eval_ts = ts_win[split:]
        protocol_note = "test = last 80% attack; DQN trained on val slice"
    seg_scores = scores[seg_start:seg_end]
    seg_y = y_win[seg_start:seg_end]
    print(f"[{MODEL_NAME}] DQN 训练片段 [{seg_start}:{seg_end}] | "
          f"n={len(seg_y)} | 正例={int(seg_y.sum())} | "
          f"score range=[{seg_scores.min():.4g}, {seg_scores.max():.4g}]")

    # 在当前评估段（paper=全 Attack，unified=后 80%）上的 AE-only 参考值
    if len(np.unique(eval_y)) > 1:
        ae_auc_eval = float(roc_auc_score(eval_y, eval_scores))
        thr_ae_eval, f1_ae_eval = best_f1(eval_y, eval_scores)
    else:
        ae_auc_eval, thr_ae_eval, f1_ae_eval = (
            float("nan"), float("nan"), float("nan"))
    print(f"[{MODEL_NAME}] AE-only on eval: AUC {ae_auc_eval:.4f} | "
          f"best F1 {f1_ae_eval:.4f}")

    agent, dqn_time = train_dqn(
        seg_scores, seg_y, args.episodes, args.dqn_batch,
        args.l_action, args.k_state, seed=args.seed, verbose=True,
        dqn_device=args.dqn_device)
    print(f"[{MODEL_NAME}] DQN 训练耗时 {dqn_time:.1f}s")
    result = evaluate_dqn(agent, eval_scores, eval_y,
                          l_action=args.l_action_infer,
                          k_state=args.k_state)
    print(f"[{MODEL_NAME}] DQN 评估: P={result['Precision']:.4f} | "
          f"R={result['Recall']:.4f} | F1={result['F1']:.4f} | "
          f"TP={result['TP']} TN={result['TN']} "
          f"FP={result['FP']} FN={result['FN']}")

    # 保存
    np.save(out_dir / "ae_scores.npy", scores)
    np.save(out_dir / "window_labels.npy", y_win)
    np.save(out_dir / "window_timestamps.npy", ts_win)
    torch.save(agent.q_net.state_dict(), out_dir / "dqn.pt")
    torch.save(ae.state_dict(), out_dir / "ae.pt")
    metrics = {
        "dataset": dataset,
        "protocol": args.protocol,
        "protocol_note": protocol_note,
        "n_features": len(sensor_cols),
        "seq_len": L,
        "window_dim": window_dim,
        "latent_dim": args.latent_dim,
        "ae": {
            "epochs": args.ae_epochs, "batch_size": args.ae_batch,
            "lr": args.ae_lr, "train_frac": args.ae_train_frac,
            "train_time_s": float(ae_time),
            "score_auc": ae_auc,
            "score_best_f1": float(f1_ae),
            "score_best_threshold": float(thr_ae),
            "score_auc_eval": ae_auc_eval,
            "score_best_f1_eval": float(f1_ae_eval),
            "score_best_threshold_eval": float(thr_ae_eval),
        },
        "dqn": {
            "episodes": args.episodes,
            "batch_size": args.dqn_batch,
            "l_action_train": args.l_action,
            "l_action_infer": args.l_action_infer,
            "k_state": args.k_state,
            "lr": args.dqn_lr,
            "gamma": args.gamma,
            "training_segment": [int(seg_start), int(seg_end)],
            "train_time_s": float(dqn_time),
            **result,
        },
        "scaler_fit": scaler_fit,
        "clip_input": bool(args.clip_input),
        "label_feedback_at_inference": True,
        "note": "Env 在推理时使用真实标签计算 TP/FP/TN/FN 状态，"
                "这属于在线标签反馈，不是纯无监督检测。",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.DataFrame({
        "timestamp": pd.to_datetime(ts_win),
        "score": scores,
        "label_window": y_win.astype(int),
    }).to_csv(out_dir / "window_scores.csv", index=False,
              encoding="utf-8-sig")
    print(f"[{MODEL_NAME}] 结果已保存: {out_dir}")


if __name__ == "__main__":
    main()
