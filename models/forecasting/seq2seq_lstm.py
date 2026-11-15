# -*- coding: utf-8 -*-
"""Seq2Seq-LSTM 预测类模型（Encoder-Decoder + Teacher Forcing）。

参考根目录 seq2seq.py 的模型结构，数据切分、训练轮询、验证集选阈值、
测试评估和绘图全部复用 models/common/ 共享管线。
"""
from __future__ import annotations

import random

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.forecast_runner import run_forecasting

MODEL_NAME = "Seq2Seq-LSTM"
PREFIX = "seq2seq_lstm_swat"

HIDDEN = 128
NUM_LAYERS = 2
DROPOUT = 0.1
TF_RATIO = 0.5   # Teacher Forcing 比例


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
        return self.fc(out), hidden, cell


class Seq2SeqLSTM(nn.Module):
    """Encoder 读入 seq_len 步历史，Decoder 自回归预测 pred_len 步。"""

    def __init__(self, n_channels, pred_len, hidden_dim=128,
                 num_layers=2, dropout=0.1, tf_ratio=0.5):
        super().__init__()
        self.n_channels = n_channels
        self.pred_len = pred_len
        self.tf_ratio = tf_ratio
        self.encoder = Encoder(n_channels, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(n_channels, hidden_dim, num_layers, dropout)

    def forward(self, src, target_len=None, tgt=None,
                teacher_forcing_ratio=0.0):
        """返回 [B, target_len, C]。

        - tgt=None：完全自回归（推理/验证）
        - tgt 非空：按 teacher_forcing_ratio 使用真实值作为下一步输入
        """
        target_len = self.pred_len if target_len is None else int(target_len)
        batch_size = src.size(0)
        hidden, cell = self.encoder(src)

        decoder_input = torch.zeros(
            batch_size, 1, self.n_channels,
            device=src.device, dtype=src.dtype)
        outputs = []
        for t in range(target_len):
            out, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(out)
            use_tf = (tgt is not None
                      and random.random() < teacher_forcing_ratio)
            decoder_input = tgt[:, t:t + 1, :] if use_tf else out

        return torch.cat(outputs, dim=1)

    def training_loss(self, x, y):
        """训练专用：带 teacher forcing 的 MSE。"""
        pred = self(x, target_len=y.size(1), tgt=y,
                    teacher_forcing_ratio=self.tf_ratio)
        return F.mse_loss(pred, y)


def build_model(n_channels, seq_len, pred_len, data=None):
    return Seq2SeqLSTM(
        n_channels, pred_len, HIDDEN, NUM_LAYERS, DROPOUT, TF_RATIO)


def main():
    run_forecasting(
        MODEL_NAME, PREFIX, build_model,
        description="Seq2Seq-LSTM 预测类模型（SWaT）",
        batch_size=256, epochs=10, lr=1e-4,
    )


if __name__ == "__main__":
    main()
