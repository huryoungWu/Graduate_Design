# USAD 复现说明（SWaT / WADI）

参考实现：[manigalati/usad](https://github.com/manigalati/usad)
论文：Audibert et al., *USAD: UnSupervised Anomaly Detection on Multivariate
Time Series*, KDD 2020, DOI [10.1145/3394486.3403392](https://doi.org/10.1145/3394486.3403392)。

本目录在同一套代码上提供两种复现口径：

| 文件 | 口径 | 训练/验证/测试 | 用途 |
|---|---|---|---|
| `models/reconstruction/usad.py` | 项目统一两阶段协议 | 完整 Normal 段训练；Attack 段前 20% 验证、后 80% 测试 | 与 `runs/` 下其它模型横向对比 |
| `models/reconstruction/usad_paper.py` | 论文 / 官方 notebook 协议 | Normal 窗口 80%/20% 训练验证；Attack 全段测试 | 与论文/文献数值对照 |
| `models/reconstruction/usad_extended_metrics.py` | 补充指标 | 读取已保存的窗口分数 | 计算 PA-F1 等论文常用口径 |

---

## 1. 方法与官方实现对齐点

网络（`UsadModel`）：

```text
Encoder: W -> Linear(W/2) -> ReLU -> Linear(W/4) -> ReLU -> Linear(z) -> ReLU
Decoder: z -> Linear(W/4) -> ReLU -> Linear(W/2) -> ReLU -> Linear(W) -> Sigmoid
AE1 = Encoder + Decoder1
AE2 = Encoder + Decoder2（共享 Encoder）
W = seq_len * n_channels
```

训练（与官方 `training()` 一致，默认 `zeroing="official"`）：

```text
L1 = (1/n) * MSE(W, AE1(W)) + (1 - 1/n) * MSE(W, AE2(AE1(W)))
L2 = (1/n) * MSE(W, AE2(W)) - (1 - 1/n) * MSE(W, AE2(AE1(W)))

optimizer1 = Adam(Encoder + Decoder1)  # 最小化 L1
optimizer2 = Adam(Encoder + Decoder2)  # 最小化 L2
n = epoch（从 1 开始）
```

测试分数：

```text
official : alpha * MSE(W, AE1(W)) + beta * MSE(W, AE2(AE1(W)))
paper    : alpha * MSE(W, AE1(W)) + beta * MSE(W, AE2(W))
alpha = beta = 0.5
```

默认超参（官方 SWaT notebook）：`window=12`、`batch=7919`、
`epochs=100`、Adam `lr=1e-3`、MinMaxScaler（只在 Normal 上 fit）。

> 官方仓库里存在两处需要说明的细节：
> 1. `USAD.ipynb` 当前源码写 `hidden_size=100`（z=12×100=1200），
>    但仓库提交的 `model.pth` 潜变量维度是 **120**（对应
>    `hidden_size=10`）。本实现两种都支持，用 `--hidden-size` 切换。
> 2. 官方 `testing()` 使用 `AE2(AE1(W))`，与论文式 (5) 的
>    `AE2(W)` 不一致；本实现同时输出两种分数。

---

## 2. 数据与预处理

- SWaT：`08_SWaT/SWaT_all_1s.csv`，45 个传感器/执行器（原始 51 列中
  6 个恒定列已删除），Normal 495,000 行，Attack 449,919 行，35 段攻击。
- WADI：`09_WADI/WADI_all_1s.csv`，98 个传感器/执行器（原始 127 列中
  29 个恒定列已删除），Normal 784,571 行，Attack 172,801 行，14 段攻击。
- 缺失：SWaT 无缺失；WADI 有 4 列共 **34 个 NaN**
  （`1_AIT_002_PV` 12、`1_AIT_004_PV` 6、`2B_AIT_004_PV` 10、
  `3_AIT_004_PV` 6；README 四舍五入为 0%）。复现时对传感器列做
  `ffill().fillna(0)`，与框架 `load_swat` 一致；否则 MinMax 后
  训练损失会变成 NaN。
- WADI Normal 段的时间缺口未插值。
- 归一化：`MinMaxScaler.fit(Normal)` 后 transform Attack（官方做法）。

窗口：`seq_len=12`，逐秒滑动；每个窗口展平为 `W = 12 × 通道数`。

---

## 2.1 实现校验

- 官方 notebook `USAD.ipynb` 的训练输出（`Epoch [0] val_loss1=0.0261,
  val_loss2=0.0263`；`Epoch [1] 0.0206/-0.0026`；
  `Epoch [2] 0.0323/-0.0210`）与我们的 `zeroing="official"` 前 3 个
  epoch 基本对齐（约 0.028/0.028、0.028/-0.006、0.040/-0.026）。
  差异主要来自本数据集删除了 6 个恒定列（45 vs 51 维）和随机种子。
  这说明双优化器训练顺序与梯度清零行为复刻正确。
- 官方仓库提交的 `model.pth` 输入维度为 `612=12×51`、潜变量为
  `120`，对应 `hidden_size=10`；而 notebook 源码当前写
  `hidden_size=100`（z=1200）。因此论文表格更可能使用 z=120。
- 官方 notebook 自己输出的 SWaT window-level ROC 只有
  `AUC=0.6825`（见 notebook cell 37），远低于论文表格的高 F1，
  说明 notebook 是演示脚本，论文表格采用的是不同的评估口径
  （很可能是逐点 PA-F1）。

---

## 3. 运行命令

```bash
# 论文协议：SWaT / WADI
python -m models.reconstruction.usad_paper \
    --csv 08_SWaT/SWaT_all_1s.csv
python -m models.reconstruction.usad_paper \
    --csv 09_WADI/WADI_all_1s.csv

# 论文协议 + z=120（仓库 model.pth 的配置）
python -m models.reconstruction.usad_paper \
    --csv 08_SWaT/SWaT_all_1s.csv --hidden-size 10 \
    --out-dir runs/usad_paper_swat_z120

# 统一框架协议
python -m models.reconstruction.usad --stage train \
    --csv 08_SWaT/SWaT_all_1s.csv --hidden-size 10
python -m models.detect --artifact-dir runs/usad_swat

# 补充指标（读取已保存分数，无需重训）
python -m models.reconstruction.usad_extended_metrics \
    --run-dir runs/usad_paper_swat --csv 08_SWaT/SWaT_all_1s.csv
```

输出目录：

```text
runs/usad_paper_swat/          # 论文协议，z=1200
runs/usad_paper_swat_z120/     # 论文协议，z=120
runs/usad_paper_wadi/          # 论文协议，WADI
runs/usad_swat/                # 统一框架协议
runs/usad_wadi/                # 统一框架协议
```

每个论文协议目录包含：`model.pt`、`metrics.json`、
`training_history.csv`、`window_scores.csv`、`scaler.npz`、
`roc.png`、`score_hist.png`。

---

## 4. 结果

> 完整训练、评估与统一框架检测均已结束；下面按论文协议和统一框架
> 协议分别汇总。

### 4.1 论文协议（Attack 全段测试）

| 数据集 | 潜在维度 | 窗口 AUC | 窗口 F1（EER 阈值） | 窗口 F1（oracle） | PA-F1（oracle） |
|---|---|---:|---:|---:|---:|
| SWaT | z=1200, 100ep | 0.7476 | 0.3336 | 0.4998 | 0.7374 |
| SWaT | z=120, 100ep | 0.7955 | 0.3596 | 0.7399 | **0.7996** |
| SWaT | z=120, 25ep | 0.7924 | 0.3428 | 0.7390 | 0.8004 |
| WADI | z=120, 100ep | 0.5108 | 0.0853 | 0.2604 | 0.3608 |
| WADI | z=120, 25ep | 0.5484 | 0.0853 | 0.2604 | 0.3608 |
| WADI | z=120, 25ep + 输入裁剪到 [0,1] | **0.7832** | **0.2092** | **0.3806** | **0.7357** |

> 初步结论：z=120（仓库 `model.pth` 的实际配置）明显优于
> notebook 源码写的 z=1200。z=1200 在 3 epochs 时窗口 AUC 0.7928、
> PA-F1 0.8040，100 epochs 后降到 0.7476/0.7374；早期模型反而更好，
> 说明 100 epochs 的对抗阶段对 z=1200 有害。z=120 则稳定。
> 两种测试分数几乎一致：z=120 官方分数 AUC 0.7955 / PA-F1 0.7996，
> 论文式(5)分数 AUC 0.7928 / PA-F1 0.8002，说明 `AE2(AE1(W))` 与
> `AE2(W)` 的差异不是主要因素。25 epochs 与 100 epochs 的结果也几乎
> 相同（AUC 0.7924 / PA-F1 0.8004），说明 z=120 的训练在 25 epoch
> 内已基本收敛，后续对抗阶段没有带来额外收益。
> WADI 则出现了明显的对抗崩溃：100 epochs 时窗口 AUC 仅 0.5108
> （接近随机），PA-F1 0.3608；损失在 epoch 35–40 左右从约 0.048
> 跳到约 0.148。25-epoch 模型损失稳定（L1≈0.0455，val≈0.0466），
> 但整体 AUC 仍只有 0.5484——说明 WADI 还有第二个问题：攻击期
> 数据超出 Normal 的 MinMax 范围，Sigmoid Decoder 无法重构，窗口
> 分数被极端值主导。
>
> **输入裁剪**：把 MinMax 后的输入裁剪到 [0,1] 后（复用 25-epoch
> 权重，不重训），WADI AUC 从 0.5484 提升到 **0.7832**，PA-F1 从
> 0.3608 提升到 **0.7357**。按天看：
>
> | 日期 | 未裁剪 AUC | 裁剪后 AUC | 未裁剪正常/攻击分数均值 |
> |---|---:|---:|---:|
> | 10-09 | 0.9049 | 0.9055 | 0.0366 / 0.0548 |
> | 10-10 | 0.5923 | 0.7710 | 259.96 / 85.46 |
> | 10-11 | 0.6466 | 0.8414 | 731.70 / 755.93 |
>
> 10-10/10-11 的未裁剪正常分数高达 260/732，攻击分数并没有更高；
> 裁剪后回到 0.05 左右，排序恢复正常。这是 WADI 上 USAD 复现的
> 关键预处理修正。

### 4.2 统一框架协议（Attack 前 20% 验证 / 后 80% 测试）

| 数据集 | 配置 | level | 分数 | AUC | PR-AUC | Precision | Recall | F1 | FPR |
|---|---|---|---|---:|---:|---:|---:|---:|
| SWaT | baseline | point | channel_norm_max | 0.8279 | 0.7447 | 0.1406 | 0.9729 | 0.2457 | 0.9466 |
| SWaT | baseline | window | recon_mean | 0.8168 | 0.7479 | 0.1556 | 0.9414 | 0.2671 | 0.8188 |
| SWaT | rolling | point | point_recon_mse_last | 0.8188 | 0.7496 | 0.6629 | 0.1414 | 0.2330 | 0.0114 |
| SWaT | rolling | window | recon_max | 0.8163 | 0.7472 | 0.6616 | 0.1395 | 0.2304 | 0.0114 |
| SWaT | rolling_all | point | channel_norm_top3 | 0.8299 | 0.7512 | 0.5544 | 0.1698 | 0.2599 | 0.0217 |
| SWaT | rolling_all | window | recon_top3_mean | 0.8165 | 0.7476 | 0.6662 | 0.1425 | 0.2347 | 0.0114 |
| WADI | baseline | point | channel_norm_top3 | 0.4981 | 0.2340 | 0.0648 | 1.0000 | 0.1217 | 0.9432 |
| WADI | baseline | window | usad_score | 0.7847 | 0.3358 | 0.0945 | 0.8045 | 0.1691 | 0.5126 |
| WADI | rolling | window | usad_score_paper | 0.7855 | 0.3426 | 0.3006 | 0.1190 | 0.1705 | 0.0184 |

> - `baseline`：验证集 max-F1 阈值；`rolling`：1% 目标 FPR 的因果滚动阈值；
>   `rolling_all`：rolling + 全部候选分数。
> - SWaT 测试 AUC/PR-AUC 约 0.82/0.75，但验证集阈值迁移到测试段后
>   F1 只有 0.23–0.27。`rolling` 把 FPR 压到约 1% 并提高 Precision，
>   但 Recall 降到约 0.14；F1 仍在 0.23–0.26。说明瓶颈是阈值选择/
>   分数漂移，而不是排序能力。
> - WADI 的 window AUC 0.7847，与论文协议裁剪版 0.7832 一致；但
>   point 级候选（窗口分数对齐到最后一点）AUC 约 0.50，说明 USAD
>   本质上是窗口级方法，强行逐点评估不合适。WADI 阈值迁移同样严重：
>   val F1 0.280 → test F1 0.169。

### 4.3 文献中 USAD 的报告值（供对照）

不同复现的数值差异很大，常见区间：

- SWaT：窗口/PA F1 约 **0.79–0.87**，AUC 约 **0.91**（IJCAI'24 表）。
- WADI：F1 约 **0.43**，AUC 约 **0.55**（IJCAI'24 表）；部分复现更低。
- 官方 notebook 提交的 checkpoint 在 SWaT Attack 全段上的
  window-level ROC AUC 仅 **0.682**，说明 notebook 本身不等于论文表格口径。

因此对照时应优先看 **逐点 PA-F1**，并注明是否降采样、是否使用全部特征、
阈值如何选择；窗口级 F1 与论文表格不可直接比较。

---

## 5. 复现要点与偏差

1. **特征数**：原始 SWaT 51 维 / WADI 127 维；本数据集删除了恒定列
   （SWaT 45 维、WADI 98 维）。恒定列在正常/攻击段都不变，理论上不
   提供判别信息，但会改变展平维度与训练损失尺度。
2. **未降采样**：部分文献对 WADI/SWaT 每 5 个点取 1 个（论文补充材料
   的做法）以降低计算量；本复现使用 1 秒全分辨率。
3. **训练细节**：官方 `training()` 的 `optimizer.zero_grad()` 只清
   本优化器的参数，两个 Decoder 会通过 `AE2(AE1(W))` 保留部分梯度；
   本实现默认严格复刻该行为（`zeroing="official"`），并提供
   `zeroing="clean"` 作为消融。
4. **阈值**：论文协议使用官方 `ROC()` 的等错误率点（TPR=1-FPR）。
   同时给出 oracle 最优阈值与 PA-F1，便于与不同文献口径对照。
5. **评估口径**：窗口级标签为“窗口内是否含攻击点”；PA-F1 把窗口分数
   对齐到窗口最后一点，并在攻击段内做 point-adjust。两者数值可能相差
   很大，报告中必须注明。
6. **WADI 输入裁剪**：官方 MinMaxScaler 不做裁剪，但 WADI 攻击期有
   大量值超出 Normal 的 [min,max]，而 Decoder 末端是 Sigmoid。未裁剪
   时窗口分数可达数百~数千，10-10/10-11 的正常窗口分数甚至高于攻击
   窗口。裁剪到 [0,1] 后 AUC 0.5484→0.7832、PA-F1 0.3608→0.7357。
   本复现同时报告未裁剪（官方口径）与裁剪（修正口径）结果。
7. **WADI 训练轮数**：100 epochs 在 epoch 35–40 发生对抗崩溃（loss1
   0.048→0.148），10-09 的 AUC 从 0.905 掉到 0.082；25 epochs 损失
   稳定但整体 AUC 仍受分布漂移影响。建议 WADI 使用 25–50 epochs，
   或保存中间 checkpoint 按验证集选择。

---

## 6. 结论

1. **实现正确**：双优化器训练循环、`zeroing="official"` 的梯度行为、
   两种测试分数、MinMax 预处理均与官方实现对齐；前几个 epoch 的
   val loss 与官方 notebook 输出基本一致。
2. **最佳配置是 z=120（`hidden_size=10`）**，而不是 notebook 源码的
   z=1200。SWaT 上 z=120 的窗口 AUC 0.7955、oracle 窗口 F1 0.7399、
   PA-F1 0.7996；z=1200 在 100 epochs 后退化到 AUC 0.7476、PA-F1
   0.7374。25 epochs 与 100 epochs 几乎一致，z=120 收敛稳定。
3. **SWaT 达到论文同一数量级**：文献中 USAD 的 PA-F1 约 0.846–0.869
   （也有 0.791 的报告），本复现 0.7996，低约 0.05。差异主要来自
   恒定列删除（45 vs 51 维）、全分辨率 vs 降采样、阈值/PA 实现和
   随机种子；窗口级 F1（0.74）不能与论文的 PA-F1 直接比较。
4. **WADI 的关键是输入裁剪**：官方 MinMax 不裁剪，但 WADI 攻击期
   大量值超出 Normal 范围，Sigmoid Decoder 无法输出，导致正常窗口
   分数爆炸（260/732）并高于攻击窗口。裁剪到 [0,1] 后（复用
   25-epoch 权重，不重训），AUC 0.5484→0.7832、PA-F1 0.3608→0.7357、
   窗口 oracle F1 0.2604→0.3806。裁剪后的窗口 F1 与文献 WADI
   USAD 的约 0.43 接近；未裁剪结果则接近随机。
5. **100-epoch 崩溃是真实的**：WADI 损失在 epoch 35–40 从 0.048
   跳到 0.148，10-09 的 AUC 从 0.905 掉到 0.082。原因是 USAD 的
   对抗 min-max 目标（AE2 最大化 `AE2(AE1(W))` 误差）在共享 Encoder、
   交替更新、跨 Decoder 梯度耦合下失稳；容量越大越容易崩
   （WADI 2.7M 参数 > SWaT z=1200 1.03M > SWaT z=120 0.6M）。
   WADI 建议 25–50 epochs 或保存中间 checkpoint 按验证集选择。
6. **统一框架协议暴露阈值迁移问题**：SWaT 测试 AUC/PR-AUC 仍有
   0.82/0.75，但验证集阈值迁移后 F1 只有 0.23–0.27；rolling 1% FPR
   把 FPR 压到约 1%、Precision 提到约 0.66，但 Recall 仅约 0.14。
   WADI 同理（AUC 0.78，F1 0.17）。因此报告 USAD 时必须同时给出
   排序指标（AUC/PR-AUC）和阈值相关指标（P/R/F1/FPR），并说明
   阈值来源，否则容易误读。
7. **复现建议**：SWaT 用 z=120、25–100 epochs；WADI 用 z=120、
   25–50 epochs + 输入裁剪；论文对照用逐点 PA-F1，并注明是否裁剪/
   降采样；统一基准中优先看窗口级 AUC/PR-AUC，阈值用 rolling FPR
   或按目标 FPR 校准。
