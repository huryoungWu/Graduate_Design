# LSTM-VAE 复现说明（SWaT / WADI）

参考实现：[TimyadNyda/Variational-Lstm-Autoencoder](https://github.com/TimyadNyda/Variational-Lstm-Autoencoder)
原始论文：Park, Hoshi & Kemp, *A Multimodal Anomaly Detector for
Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder*,
IEEE RA-L 2018。

参考仓库是 TensorFlow/Keras 实现；本目录用 **PyTorch** 逐项复刻，
并接入项目统一两阶段框架。**结论：完整 PyTorch 复现可行，不需要
TensorFlow**；唯一需要手工实现的是 TF 的 ReLU LSTM 单元（PyTorch
`nn.LSTM` 固定 tanh）。

---

## 1. PyTorch 复刻映射

| 参考实现（TF/Keras） | 本实现（PyTorch） |
|---|---|
| `LSTMCell(intermediate_dim, activation=relu, forget_bias=1)` | `ReLULSTMCell`（自定义单元，见 `models/reconstruction/lstm_vae.py`） |
| 编码器取最后时刻输出 | 循环 `seq_len` 步后取 `h_t` |
| `z_mean = Linear(h)` | `z_mean_layer` |
| `z_sigma = softplus(Linear(h))`，按 log-variance 使用 | `logvar = F.softplus(z_logvar_layer(h))` |
| `z = mean + exp(0.5*logvar) * eps` | `reparameterize` |
| `RepeatVector(seq_len)(z)` | `z.unsqueeze(1).repeat(1, T, 1)` |
| `MultiRNNCell([LSTMCell(intermediate_dim), LSTMCell(n_dim)])` | 两层 `nn.LSTM`，forget bias 初始化为 1 |
| 损失 `MSE + kl_coef*KL + reg_lambda*L2(Encoder)` | 同一公式 |
| `MinMaxScaler(feature_range=(-1,1))` | 框架 `--scaler minmax11` |
| Adam lr=1e-3，grad_clip=10 | 自定义 `fit`，同一设置 |

说明：

- TF 的 `activation=relu` 不能直接用 PyTorch `nn.LSTM`（默认 tanh）
  代替，因此编码器使用自定义 `ReLULSTMCell`；解码器保持默认 tanh，
  与参考实现一致。
- 参考实现的 `stateful` 选项默认关闭；本实现只实现 stateless 版本。
- 推理时使用潜变量均值（确定性评分）；训练时重参数采样。
- 可选 `--clip-input` 把输入裁剪到 [-1,1]（WADI 建议开启）。

## 2. 数据与超参

- SWaT：`08_SWaT/SWaT_all_1s.csv`（45 维，Normal 495,000 / Attack 449,919）
- WADI：`09_WADI/WADI_all_1s.csv`（98 维，Normal 784,571 / Attack 172,801）
- 窗口：`seq_len=12`
- 参考实现默认：`intermediate_dim=15`、`z_dim=3`、`kl_coef=0.1`、
  `reg_lambda=0.01`、`lr=1e-3`、`grad_clip=10`、`batch=100`、`epochs=200`
- 本复现为兼顾大数据量：`batch=1024`、`epochs=100`（其余默认）；
  可用 CLI 覆盖。
- 统一框架协议：完整 Normal 段训练；Attack 前 20% 验证、后 80% 测试。
- 全 Attack 评估：复用训练好的权重，在整个 Attack 段上计算 AUC/PA-F1。

## 3. 运行命令

```bash
# SWaT：统一框架（训练 + 检测）
python -m models.reconstruction.lstm_vae --stage all \
    --csv 08_SWaT/SWaT_all_1s.csv --batch-size 1024

# WADI：统一框架（建议输入裁剪）
python -m models.reconstruction.lstm_vae --stage all \
    --csv 09_WADI/WADI_all_1s.csv --batch-size 1024 --clip-input

# 全 Attack 段评估（复用权重）
python -m models.reconstruction.lstm_vae_paper \
    --csv 08_SWaT/SWaT_all_1s.csv
python -m models.reconstruction.lstm_vae_paper \
    --csv 09_WADI/WADI_all_1s.csv --clip-input
```

产物：

```text
runs/lstmvae_swat/                  # artifact + detection
runs/lstmvae_wadi/                  # artifact + detection
runs/lstmvae_paper_swat/            # 全 Attack 段指标
runs/lstmvae_paper_wadi/            # 全 Attack 段指标
```

## 4. 结果

> SWaT 与 WADI 均已完成（论文协议 + 统一框架协议）。

### 4.1 统一框架协议（验证集选阈值）

| 数据集 | 配置 | level | 选中分数 | AUC | PR-AUC | Precision | Recall | F1 | FPR |
|---|---|---|---|---:|---:|---:|---:|---:|
| SWaT | baseline | point | channel_norm_max | 0.7190 | 0.2321 | 0.1414 | 0.9716 | 0.2469 | 0.9394 |
| SWaT | baseline | window | recon_mean | 0.8162 | 0.7492 | 0.1581 | 0.9386 | 0.2706 | 0.8013 |
| SWaT | rolling | point | channel_norm_max | 0.7190 | 0.2321 | 0.1769 | 0.2371 | 0.2026 | 0.1757 |
| SWaT | rolling | window | recon_top3_mean | 0.8161 | 0.7491 | 0.6362 | 0.1225 | 0.2055 | 0.0112 |
| WADI | baseline | point | point_recon_mse_last | 0.8069 | 0.3230 | 0.0966 | 0.9088 | 0.1747 | 0.5551 |
| WADI | baseline | window | recon_mean | 0.8057 | 0.3267 | 0.0987 | 0.9072 | 0.1780 | 0.5512 |
| WADI | rolling | point | point_kl | 0.7301 | 0.1233 | 0.0784 | 0.1294 | 0.0976 | 0.0994 |
| WADI | rolling | window | kl | 0.7290 | 0.1249 | 0.0792 | 0.1287 | 0.0981 | 0.0995 |

> SWaT 的验证集阈值迁移后 F1 只有 0.25–0.27，但窗口 AUC/PR-AUC
> 仍有 0.816/0.749。值得注意的是 **KL 分数**：测试集 F1 0.462
> （P 0.325 / R 0.797 / FPR 0.265），远高于验证集选中的 recon_mean；
> 验证集 F1（0.144）低估了它。说明 LSTM-VAE 的 KL 项也可以作为
> 有效的异常分数，后续可用 rolling FPR 或按目标 FPR 校准。
>
> rolling FPR（目标 1%）把 FPR 压到 1.1%，但 Recall 只有 0.12，
> F1 0.21，低于 baseline 的 0.27；在这个数据上单纯压低 FPR
> 并不能提升 F1，阈值/分数漂移需要更细的校准。
>
> WADI 的窗口 AUC 为 0.8057、PR-AUC 0.3267，明显高于文献
> LSTM-VAE 的 AUC≈0.50；这主要得益于 `--clip-input`（攻击期
> 越界值被裁剪到 MinMax(-1,1) 范围）。窗口 F1 0.178（验证集
> 阈值），KL 分数的测试 F1 0.257，同样存在阈值迁移问题。
> rolling FPR 在 WADI 上 FPR 仍达到 9.95%、F1 0.098，效果差于
> baseline；说明这里的分数漂移不是简单滚动分位数能解决的。

### 4.2 全 Attack 段（oracle 阈值，供文献对照）

| 数据集 | 分数 | AUC | PR-AUC | 窗口 F1@EER | 窗口 F1@oracle | PA-F1@oracle |
|---|---|---:|---:|---:|---:|---:|
| SWaT | recon_mean | 0.7916 | 0.6911 | 0.3395 | 0.7404 | **0.8003** |
| SWaT | recon_max | 0.7915 | 0.6911 | 0.3397 | 0.7404 | 0.8053 |
| SWaT | point_recon_mse_last | 0.7916 | 0.6910 | 0.3394 | 0.7410 | 0.8054 |
| WADI | recon_mean | 0.7993 | 0.2965 | 0.2107 | 0.3654 | **0.6216** |
| WADI | recon_max | 0.8022 | 0.2957 | 0.2145 | 0.3661 | 0.6264 |
| WADI | point_recon_mse_last | 0.7993 | 0.2979 | 0.2125 | 0.3682 | 0.6495 |

> SWaT 的 PA-F1 0.800–0.805 与文献 LSTM-VAE 的 0.8051
> （STAD-GAN 表，point-adjusted）几乎完全一致；窗口 AUC 0.792
> 低于文献的 0.883，说明两者的 AUC 计算口径/数据预处理不同，
> 但 F1 口径对上了。
>
> WADI 的 raw F1@EER 0.211 与文献无 PA F1 0.227 接近；
> window oracle F1 0.365 接近文献 PA F1 0.380；PA-F1 0.62–0.65
> 更高（裁剪 + point-adjust）。窗口 AUC 0.799 远高于文献的
> 0.498，主要来自 `--clip-input` 对越界值的处理。
>
> 注：4.2 的全 Attack 段评估复用统一框架训练好的权重
> （完整 Normal 段训练），没有重新按 80/20 训练/验证；与严格的
> 论文协议略有差异，但预处理、窗口和打分口径一致。

### 4.3 文献中 LSTM-VAE 的报告值

- IJCAI'24 基准表：SWaT AUC 88.3 / F1 80.5；WADI AUC 49.8 / F1 38.0。
- STAD-GAN（ACM 2023）：
  - SWaT：无 PA F1 0.7756（P 0.9897 / R 0.6377）；
    有 PA F1 0.8051（P 0.7123 / R 0.9258）。
  - WADI：无 PA F1 0.2271（P 0.9947 / R 0.1282）；
    有 PA F1 0.3799（P 0.4632 / R 0.3220）。
- 不同论文的 LSTM-VAE 实现和评估口径差异较大，对照时需注明
  是否使用 point-adjust、是否降采样、窗口/预处理是否一致。

## 5. 复现要点与偏差

1. **ReLU LSTM**：自定义单元与 TF 的 `activation=relu` 对齐；
   若改成标准 tanh LSTM，结果会不同。
2. **批量大小/轮数**：参考实现 batch=100、epochs=200；本复现
   batch=1024（适配 50 万级窗口），SWaT 跑 100 epochs、WADI 跑
   50 epochs。loss 在约 10–20 epoch 后已基本不变，可用
   `--epochs` 调整；通过 CLI 可改回参考设置。
3. **stateful**：未实现；参考实现默认也是 stateless。
4. **推理采样**：使用潜变量均值而非单次采样，评分更稳定；
   训练仍使用重参数采样。
5. **WADI 输入越界**：与 USAD 类似，攻击期存在超出 Normal
   MinMax(-1,1) 范围的值；建议 `--clip-input`，否则
   tanh 解码器无法重构越界值，分数会被极端值主导。
6. **模型容量**：参考实现 hidden=15、z=3 很小；若在 SWaT/WADI
   上欠拟合，可尝试 `--intermediate-dim 64 --z-dim 8` 等更大配置，
   但那就偏离参考实现了。
7. **L2 项**：参考实现先创建 `train_op`，之后才把
   `REG_LAMBDA * L2(Encoder)` 加到 `_cost`；在 TF1 中梯度已经
   用旧计算图构建，因此这个 L2 实际上不生效。本实现按 README 的
   意图真正加入 L2；默认 `reg_lambda=0.01`，影响很小，可用
   `--reg-lambda 0` 复刻“无 L2”的行为。

## 6. 结论

1. **PyTorch 复现可行且已完成**：编码器 ReLU LSTM、潜变量头、
   两层 LSTM 解码器、KL 损失、MinMax(-1,1) 预处理和训练循环
   都已用 PyTorch 实现，不依赖 TensorFlow。
2. **SWaT 与文献对齐**：全 Attack 段窗口 AUC 0.7916、PR-AUC
   0.6911、PA-F1 **0.800–0.805**；文献 LSTM-VAE 的 PA-F1 为
   0.8051（STAD-GAN 表），几乎完全一致。窗口 AUC 低于文献的
   0.883，说明 AUC 口径/预处理不同，但 F1 口径对上了。
3. **统一框架的瓶颈是阈值迁移**：SWaT 测试 AUC/PR-AUC 仍有
   0.816/0.749，但验证集阈值迁移后 F1 只有 0.27；KL 分数的
   测试 F1 反而有 0.462。rolling 1% FPR 把 FPR 压到 1.1%，
   但 Recall 只有 0.12、F1 0.21。报告时必须区分排序指标和
   阈值相关指标。
4. **WADI 也与文献同量级**：全 Attack 段 raw F1@EER 0.211
   （文献无 PA F1 0.227）、窗口 oracle F1 0.365（文献 PA F1
   0.380）、PA-F1 0.622–0.650；窗口 AUC 0.799 明显高于文献
   0.498，主要得益于 `--clip-input`。统一框架下窗口 F1 0.178
   （验证集阈值），KL 分数测试 F1 0.257。
5. **建议**：SWaT 可用 `--epochs 20–30`（loss 早已收敛）；
   WADI 建议 `--epochs 50 --clip-input`；若需更高指标，可尝试
   更大的 `--intermediate-dim/--z-dim` 或文献基准超参
   （lr 9.5e-3、λ_kl 0.28、λ_reg 0.55），但那会偏离参考默认。
