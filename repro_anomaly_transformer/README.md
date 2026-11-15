# Anomaly-Transformer 复现（SWaT / WADI）

代码来源：

- 模型与训练/评价逻辑：`https://github.com/Manami-GIWA/Anomaly-Transformer`
  （与官方 `thuml/Anomaly-Transformer` 同一套实现）
- 本地数据：`08_SWaT/SWaT_all_1s.csv`、`09_WADI/WADI_all_1s.csv`

本目录只做两件事：

1. 把本地 CSV 转成官方 loader 所需的 Normal/Attack 数组；
2. 复用官方 `AnomalyTransformer`、minimax 损失、energy 阈值和
   point-adjust 评价，在 SWaT/WADI 上训练并测试。

## 1. 实现逻辑

### 1.1 模型

- 输入窗口 `[B, win_size, C]`；
- `DataEmbedding`：`Conv1d` token embedding + 正弦位置编码；
- 3 层 Encoder，每层包含 **Anomaly-Attention**：
  - series association：`softmax(QKᵀ/√d)`；
  - prior association：每个 head 学习一个 `σ`，
    `prior_ij ∝ 1/(√(2π)σ)·exp(-|i-j|²/(2σ²))`；
  - 两者之间的对称 KL 即 association discrepancy。
- 输出重构值 `x̂`、每层 `series`、`prior`、`σ`。

默认尺寸与官方一致：

```text
win_size=100, d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0
```

### 1.2 Minimax 训练

```text
series_loss = KL(series || prior.detach) + KL(prior.detach || series)
prior_loss  = KL(prior || series.detach) + KL(series.detach || prior)
loss1 = MSE(x, x̂) - k * series_loss
loss2 = MSE(x, x̂) + k * prior_loss
```

先 `loss1.backward(retain_graph=True)`，再 `loss2.backward()`，同一个
optimizer step。官方默认 `k=3`、`lr=1e-4`、`Adam`。

### 1.3 异常分数

```text
loss_t       = mean_c (x_t - x̂_t)^2
series_loss  = KL(...) * temperature      # temperature=50
prior_loss   = KL(...) * temperature
metric_t     = softmax_t(-series_loss - prior_loss)
score_t      = metric_t * loss_t
```

### 1.4 阈值与评价

- 在 Normal 训练段和 Attack 测试段上分别算逐点 score；
- `threshold = percentile(concat(train_score, test_score), 100 - anomaly_ratio)`；
- 预测 `score > threshold`；
- **point-adjusted 评价**：若某个连续异常段内至少有一个点被检出，
  则把整段都视为检出，再算 Precision/Recall/F1。这是 Anomaly-Transformer
  官方代码使用的评价方式，也是论文报告的 F1 口径。
- 另外额外保存未调整的 P/R/F1、AUC、PR-AUC 供参考。

## 2. 数据预处理

`prepare_data.py`：

- `train = period == "normal"`；
- `test = period == "attack"`，带 `label`；
- `StandardScaler` 只在 Normal 训练段 fit；
- 本地 CSV 已删除恒定列，为对齐官方输入维数，补回 0：
  - SWaT：45 + 6 = 51；
  - WADI：98 + 29 = 127。
- 输出 `train_X.npy`、`test_X.npy`、`test_y.npy`、时间戳和 `meta.json`。

```bash
python -m repro_anomaly_transformer.prepare_data --dataset swat
python -m repro_anomaly_transformer.prepare_data --dataset wadi
```

## 3. 训练与测试

### SWaT：20/80 冻结测试协议（推荐用于对比）

```bash
python -m repro_anomaly_transformer.run --dataset swat \
    --win-size 100 --batch-size 32 --epochs 3 --lr 1e-4 --k 3 \
    --val-mode attack_val --val-frac 0.2 \
    --out-dir repro_anomaly_transformer/runs/swat_20_80
```

- 训练：Normal 全部；
- 验证/early stopping：Attack 前 20%；
- **冻结测试**：Attack 后 80%；
- 阈值：只在验证集上按 F1 选（`val_best_f1`）；
- 测试集不参与训练、early stopping 或阈值选择。

### SWaT：官方原始协议（仅用于复现论文）

```bash
python -m repro_anomaly_transformer.run --dataset swat \
    --win-size 100 --batch-size 32 --epochs 3 --lr 1e-4 --k 3 \
    --anomaly-ratio 0.1
```

- `step=1`（官方 SwatSegLoader 的滑动窗口方式）；
- 验证/early stopping 使用 test loader（官方代码行为；不用标签）；
- 阈值 `train+test` 分位数（`anomaly_ratio=0.1`）。
- 这条线**不满足冻结测试集**，不要和 20/80 表混用。

### WADI

```bash
python -m repro_anomaly_transformer.run --dataset wadi \
    --win-size 100 --batch-size 32 --epochs 3 --lr 1e-4 --k 3 \
    --anomaly-ratio 0.1 --clip-scaled 10
```

WADI 没有官方脚本，沿用 SWaT 的同一套协议；`input_c=127`。

**WADI 必须做数值裁剪**：`2B_AIT_002_PV` 在 Normal 段均值约 9.1、
标准差约 0.16，但 Attack 段中位数约 8099。StandardScaler 后该特征
达到 ~49,424，会让验证损失直接爆炸到 `1e7`。`--clip-scaled 10`
把标准化后的窗口裁剪到 `[-10, 10]`，保留“攻击时该特征显著升高”的
信号，同时避免数值溢出；WADI 冒烟测试中验证损失恢复到正常量级。
也可以用 `prepare_data.py --winsorize 0.001` 在 Normal 分位上裁剪。

### SBT：同一套 20/80 协议

```bash
python -m repro_anomaly_transformer.run_sbt --epochs 10
```

- SBT 只输出 60 步窗口分数（`score_max` / `score_mean`），评估单位是窗口；
- 训练只用 Normal，验证用 Attack 前 20%，测试用 Attack 后 80%；
- 阈值只在验证集上按 F1 选。

### 冒烟测试

```bash
python -m repro_anomaly_transformer.run --dataset swat --smoke \
    --epochs 1 --no-plot --val-mode attack_val
python -m repro_anomaly_transformer.run_sbt --smoke --epochs 1
```

## 4. 输出

```text
repro_anomaly_transformer/runs/<dataset>/
  <dataset>_checkpoint.pth   # 验证损失最优的权重
  metrics.json               # PA P/R/F1、未调整 P/R/F1、AUC/PR-AUC、阈值、配置
  test_scores.csv            # timestamp,label,score,pred,pred_adjusted,tp/fp/tn/fn
  test_energy.npy            # 测试段逐点 energy
  train_energy.npy           # Normal 段逐点 energy（用于阈值）
  test_score_overview.png    # 全测试段分数概览
  plots/daily/               # 每天一张四色背景图（TN/TP/FP/FN）
```

## 5. 已知差异与注意

- **训练集行数**：本地 SWaT Normal 段为 495,000 行；上述复现博客的
  Normal 段为 496,800 行。两者相差 0.36%，结果可能略有差异。
- **恒定列**：本地已删除 6/29 个恒定列，这里补 0；标准化后恒定列本来
  就是 0，因此对非恒定特征没有影响，但模型输入维数保持 51/127。
- **early stopping**：官方代码用 test loader（无标签）做验证；这是
  transductive，但不是标签泄漏。若要做严格 inductive 对比，可用
  `--val-mode normal`（训练只用前 80% Normal，最后 20% 做验证）。
- **阈值**：官方用 `concat(train_score, test_score)` 的分位数，属于
  转导式阈值；测试标签没有参与。
- **PA 评价**：论文 F1 是 point-adjusted F1，会发现“只检出一个点”
  也算整段检出；未调整的 P/R/F1 也保存在 `metrics.json` 中。
- **随机性**：官方代码没有固定 seed；本实现默认 `seed=42`。
- **WADI 极值**：`2B_AIT_002_PV` 的 Normal/Attack 分布差异极大；
  不裁剪会导致验证损失 `1e7`。这是 WADI 数据本身的问题，不是模型
  forward 的 bug。
- **内存**：`train_X/test_X` 以只读 memmap 打开，窗口按需 copy；
  WADI 的 `train_energy` 仍会有 ~78M 个 float32（约 314 MB），
  低内存机器需要注意。

## 6. 参考结果

同一份实现/数据格式的公开复现（Tencent Cloud 博客）在 SWaT 上报告：

```text
Threshold ≈ 0.003219
Accuracy 0.9771
Precision 0.8965
Recall    0.9172
F1        0.9067
```

本目录运行完成后，把 `metrics.json` 与这组数字对比即可。
