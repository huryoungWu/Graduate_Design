# 模型代码结构（两阶段流程）

训练/推理与异常检测已经解耦：

```text
Stage 1  train : 训练 -> 推理 val/test -> 保存 artifact + 预测/重构精度
Stage 2  detect: 读取 artifact -> 计算残差 -> 聚合 -> 选阈值 -> 检测评估
```

两个阶段之间用 `runs/<prefix>/` 下的 `npz + meta.json` 连接。Stage 2 不导入模型，
也不重新读取原始 CSV，因此可以反复换阈值、换聚合方式，而不用重新训练。

## 目录

```text
models/
├─ common/
│  ├─ config.py            # 参数（--stage / --artifact-dir / --scores / --threshold ...）
│  ├─ data.py              # SWaT 读取、切分、标准化、窗口 Dataset
│  ├─ train.py             # 训练循环
│  ├─ inference.py         # val/test 推理收集
│  ├─ artifacts.py         # artifact 保存/加载
│  ├─ scores.py            # 残差计算、通道/时间步聚合、平滑、归一化
│  ├─ detection.py         # Stage 2：选阈值 + 测试集评估
│  ├─ forecast_runner.py   # 预测类两阶段入口
│  ├─ recon_runner.py      # 重构/对比类两阶段入口
│  └─ eval.py / protocol.py / plots.py
├─ forecasting/            # 只保留各模型网络结构
├─ reconstruction/         # 只保留各模型网络结构
├─ detect.py               # 独立检测 CLI
└─ run_all.py              # 批量运行（--stage all/train/detect）
```

## 运行方式

### 预测类

```bash
# Stage 1：训练 + 保存 artifact + prediction_metrics.csv
python -m models.forecasting.lstm --stage train --epochs 50

# Stage 2：读取 artifact，做异常检测
python -m models.detect --artifact-dir runs/lstm_swat \
    --scores mean,max,top3_mean --threshold val_best_f1

# 一条命令完成两阶段
python -m models.forecasting.lstm --stage all --epochs 50

# 复用已有 checkpoint，不重新训练
python -m models.forecasting.lstm --stage train \
    --load-model lstm_swat_model.pt --artifact-dir runs/lstm_swat
```

### 重构 / 对比类

```bash
# AE / VAE / AnomalyTransformer
python -m models.reconstruction.ae --stage train --epochs 50
python -m models.detect --artifact-dir runs/ae_swat \
    --scores recon_mean,recon_max --threshold val_best_f1

# DCdetector / SBT（无重构，只有表示差异分数）
python -m models.reconstruction.dcdetector --stage train --epochs 50
python -m models.detect --artifact-dir runs/dcdetector_swat \
    --scores score_mean,score_max --threshold val_best_f1

# USAD（默认即官方超参：window=12、hidden=100、MinMaxScaler、
#       batch=7919、epochs=100、lr=1e-3；w_size=seq_len*通道数，
#       z_size=seq_len*100）
python -m models.reconstruction.usad --stage all \
    --csv 08_SWaT/SWaT_all_1s.csv          # -> runs/usad_swat
python -m models.reconstruction.usad --stage all \
    --csv 09_WADI/WADI_all_1s.csv          # -> runs/usad_wadi

# 复用论文协议训练好的权重，只做统一框架推理+检测（跳过训练）
python -m models.reconstruction.usad --stage all \
    --csv 08_SWaT/SWaT_all_1s.csv --hidden-size 10 \
    --load-model runs/usad_paper_swat_z120/model.pt

# USAD 论文/官方 notebook 协议复现（Normal 80/20 训练验证、
# Attack 全段测试、窗口级标签、ROC 等错误率阈值）
python -m models.reconstruction.usad_paper \
    --csv 08_SWaT/SWaT_all_1s.csv          # -> runs/usad_paper_swat
python -m models.reconstruction.usad_paper \
    --csv 09_WADI/WADI_all_1s.csv          # -> runs/usad_paper_wadi

# 可选参数：
#   --hidden-size 10    z=120（官方仓库 model.pth 的实际配置）
#   --eval-only         只加载 <out-dir>/model.pt 重新评估
#   --downsample 5      降采样（部分文献的 WADI 设置）
#   --zeroing clean     每个子步骤清空全部梯度（消融）

# 补充指标（PA-F1 等）与结果汇总
python -m models.reconstruction.usad_extended_metrics \
    --run-dir runs/usad_paper_swat --csv 08_SWaT/SWaT_all_1s.csv
python -m models.reconstruction.summarize_usad_runs

# LSTM-VAE（参考实现 TimyadNyda/Variational-Lstm-Autoencoder）
# 默认：MinMax(-1,1), hidden=15, z=3, kl=0.1, reg=0.01, lr=1e-3,
#       batch=256, epochs=100, grad_clip=10
python -m models.reconstruction.lstm_vae --stage all \
    --csv 08_SWaT/SWaT_all_1s.csv --batch-size 1024
python -m models.reconstruction.lstm_vae --stage all \
    --csv 09_WADI/WADI_all_1s.csv --batch-size 1024 --epochs 50 --clip-input

# Isolation Forest（sklearn，经典基线；Normal 拟合，逐点打分后聚合窗口）
python -m models.classical.isolation_forest --protocol framework \
    --csv 08_SWaT/SWaT_all_1s.csv      # -> runs/iforest_swat
python -m models.classical.isolation_forest --protocol framework \
    --csv 09_WADI/WADI_all_1s.csv      # -> runs/iforest_wadi
python -m models.classical.isolation_forest --protocol paper \
    --csv 08_SWaT/SWaT_all_1s.csv      # -> runs/iforest_paper_swat

# DAGMM（PyTorch 复现；压缩网络 + 估计网络 + GMM 能量）
python -m models.classical.dagmm --protocol framework \
    --csv 08_SWaT/SWaT_all_1s.csv      # -> runs/dagmm_swat
python -m models.classical.dagmm --protocol framework \
    --csv 09_WADI/WADI_all_1s.csv      # -> runs/dagmm_wadi
python -m models.classical.dagmm --protocol paper \
    --csv 08_SWaT/SWaT_all_1s.csv      # -> runs/dagmm_paper_swat

# ADT（PyTorch 复现；AE 打分 + DQN 动态阈值）
python -m models.classical.adt --protocol paper \
    --csv 08_SWaT/SWaT_all_1s.csv      # -> runs/adt_paper_swat
python -m models.classical.adt --protocol paper \
    --csv 09_WADI/WADI_all_1s.csv      # -> runs/adt_paper_wadi
python -m models.classical.adt --protocol unified \
    --csv 08_SWaT/SWaT_all_1s.csv      # 只用 val 标签训练 DQN

# 统一可部署口径：Normal 99% 分位阈值 + 全 Attack 段评估
# （ADT 使用 AE-only；DQN 依赖标签反馈，不参与该阈值协议）
python -m models.classical.normal99_full_attack
# 产物：runs/normal99_full_attack/comparison.csv

# 扩展：目录内其它预测/重构模型（现有 SWaT checkpoint）
python -m models.classical.normal99_project_models
# 可只跑部分模型：--models LSTM AE AnomalyTransformer
# 产物：runs/normal99_project_models/comparison.csv
# 两个脚本还会输出 comparison_methods.csv：
#   q99 / q99.9 / held-out q99 / alarm-budget 多阈值对比
```

详细说明见 `models/reconstruction/USAD_复现说明.md` 与
`models/reconstruction/LSTM_VAE_复现说明.md`。

`usad.py` 使用项目统一协议，`usad_paper.py` 使用论文协议，两者结果
不直接等同；论文对照请用后者。USAD 需要 `--scaler minmax`
（`usad.py` 已自动设置），其它模型默认仍为 `--scaler standard`。

### 批量运行

```bash
python -m models.run_all --group forecasting --stage train
python -m models.run_all --group reconstruction --stage train
python -m models.run_all --group all --stage detect
```

## 统一评估协议

| 片段 | 范围 | 用途 |
|---|---|---|
| train | 前 495,000 行（Normal 段） | 训练、标准化 |
| val | Attack 段前 20% | 选阈值、选评分配置 |
| test | Attack 段后 80% | 只报告最终指标 |

- 测试集标签不参与阈值、评分配置或模型选择。
- `val_frac` 默认 0.20；它是“Attack 段中验证集比例”，不是训练/验证比例。
- 训练集始终是 Normal 段，改 `val_frac` 不会增加训练数据。

## 重构模型统一对比

`compare_reconstruction_models.py` 用完全相同的口径比较 SWaT 上的
AE / VAE / AnomalyTransformer / USAD / LSTM-VAE：

- 取所有模型验证集/测试集时间戳的交集；
- 统一逐点分数：窗口最后时刻的通道平均平方重构误差；
- 统一 12 步滑窗评估；
- 只在验证集上按 F1 选阈值（`val_best_f1`）。

```bash
python -I compare_reconstruction_models.py
# -> runs/recon_comparison/COMPARISON.md
# -> runs/recon_comparison/comparison_unified_point.csv
# -> runs/recon_comparison/comparison_unified_window12.csv
# -> runs/recon_comparison/comparison_recon_accuracy.csv
```

同时输出原始物理量下的重构精度（MAE/RMSE、按通道量程归一化的
nMAE/nRMSE）以及 IsolationForest 参考基线。DCdetector / SBT 目前没有
可复现的 checkpoint/artifact，暂不纳入统一对比。

## artifact 结构

```text
runs/<prefix>/
  meta.json                  # 模型、通道、切分、超参数、scaler
  model.pt                   # 模型权重
  val.npz                    # 验证集输出
  test.npz                   # 测试集输出
  prediction_metrics.csv     # 预测类：逐通道 MAE/RMSE/MAPE
  recon_metrics.csv          # 重构类：逐通道重构误差
  detection/
    detection_metrics.csv    # 所有分数配置的测试集指标
    selected_config.json     # 验证集选中的配置和阈值
    scores_<level>_<name>.csv
    plots/
      <level>_<name>/              # 残差/异常分数按天曲线
      <level>_<name>_true_vs_pred/ # 预测类：True vs Pred（含残差和 TN/TP/FP/FN）
      <level>_<name>_true_vs_recon/# 重构类：True vs Recon（含残差和 TN/TP/FP/FN）
```

### 预测类 npz

| 键 | 形状 | 含义 |
|---|---|---|
| `pred` | `[N,C]` 或 `[N,P,C]` | 预测值 |
| `true` | `[N,C]` 或 `[N,P,C]` | 真值 |
| `label` | `[N]` 或 `[N,P]` | 标签 |
| `timestamp` | `[N]` | 预测时刻 |
| `score_<name>` | `[N]` | 模型特有的额外分数（可选） |

### 重构类 npz

| 键 | 形状 | 含义 |
|---|---|---|
| `recon_last` / `true_last` | `[N,C]` | 最后时刻重构/真值 |
| `abs_last` / `sq_last` | `[N,C]` | 最后时刻绝对/平方误差 |
| `abs_ts` / `sq_ts` | `[N,L]` | 逐时刻通道平均误差 |
| `label_window` | `[N]` | 窗口内是否含攻击 |
| `label_point` | `[N]` | 窗口最后一点标签 |
| `score_<name>` | `[N]` 或 `[N,L]` | 窗口/点级模型分数 |

## Stage 2 常用参数

```bash
python -m models.detect \
  --artifact-dir runs/lstm_swat \
  --scores mean,max,top3_mean \     # 只评估这些分数；空=全部
  --threshold val_best_f1 \         # val_best_f1 | val_fpr | fixed
  --target-fpr 0.01 \               # threshold=val_fpr 时使用
  --normalize none \                # none | rolling
  --smooth-window 1 \               # 平滑窗口
  --select-by val_f1                # val_f1 | val_pr_auc | val_auc
```

`detection_metrics.csv` 的列：

```text
level, name, AUC, PR_AUC, Precision, Recall, F1, FPR,
threshold, threshold_source, val_F1, val_PR_AUC, val_AUC,
val_precision, val_recall, val_FPR
```

检测阶段会同时输出两类按天图：

- `plots/<level>_<name>/`：残差/异常分数随时间变化；
- `plots/<level>_<name>_true_vs_pred/`：预测类的**预测值 vs 真实值**；
- `plots/<level>_<name>_true_vs_recon/`：重构类的**重构值 vs 真实值**。

两组图都带 TN/TP/FP/FN 四色背景色带和残差子图。预测/重构值 vs 真实值图使用**原始物理量**
（通过 `meta.json` 里的 scaler 反标准化），异常分数图仍使用标准化空间。
DCdetector/SBT 没有预测量或重构量，只输出分数图，并打印跳过真实值对比图的提示。

## 模型文件需要实现什么

### 预测类

```python
def build_model(n_channels, seq_len, pred_len, data=None):
    return MyForecaster(...)

def main():
    run_forecasting("MyModel", "my_swat", build_model)
```

网络 `forward(x: [B,L,C]) -> [B,P,C]`，训练损失统一为 MSE。

### 重构 / 对比类

```python
def build_model(n_channels, seq_len, data=None):
    return MyModel(...)

def main():
    run_reconstruction("MyModel", "my_swat", build_model)
```

模型需实现：

```python
def training_loss(self, x): ...
def score_batch(self, x): ...
```

`score_batch` 返回：

```python
{
  "window": {"score_name": [B]},        # 窗口级分数
  "point":  {"point_name": [B]},        # 最后一点分数（可选）
  "ts":     {"sq_ts": [B,L], ...},      # 逐时刻误差（可选）
  "channel": {"recon_last": [B,C], ...} # 逐通道重构/误差（可选）
}
```

- AE/VAE/AnomalyTransformer 提供 `ts` 和 `channel`，Stage 1 会生成 `recon_metrics.csv`。
- DCdetector/SBT 只有 `window`，Stage 1 标记为 `task="score"`，不做重构精度。
- 可选 `make_fused(reference, target)`：用验证集统计量构造融合分数。

## 迁移说明

- 旧的根目录 CSV（`*_swat_anomaly_scores.csv`、`*_channel_metrics.csv` 等）已移到
  `_old_results_csv/` 备份；新流程不再写这些文件名。
- 旧命令 `python -m models.forecasting.lstm` 仍可用，默认 `--stage all`，
  会训练并检测，结果写在 `runs/lstm_swat/`。
- 旧的 `forecast_runner` 多步 lead/window 评估代码已移除；如需多步检测，
  可在 Stage 1 保存全部预测步（`--save-all-steps`），再在 Stage 2 扩展。
- 旧的 `*_channel_plots/` 和根目录 `*_model.pt` 仍保留，可手动删除；
  新流程的图片在 `runs/<prefix>/detection/plots/`。

## 数值验证

用已有 checkpoint 重新生成 artifact 后，与旧 CSV 对比：

- LSTM、Transformer：`prediction_metrics.csv` 的逐通道 MAE/RMSE/MAPE 与旧
  `*_channel_metrics.csv` **完全一致**；Stage 2 的 `mean` 行与旧
  `*_anomaly_score_comparison.csv` 一致。
- AE、VAE：`recon_metrics.csv` 与旧 `*_channel_metrics.csv` 完全一致；
  `recon_mean`/`point_*` 等旧分数行与旧 CSV 一致。
- AnomalyTransformer：`recon_max`、`assoc_max`、`fused` 等窗口分数可复现。

新增的 `abs_*`/`recon_top3_mean` 等候选分数是 Stage 2 的扩展，可能会被
验证集选中，因此最终选中配置不一定和旧 CSV 相同，但所有旧配置仍然保留在
`detection_metrics.csv` 中。
