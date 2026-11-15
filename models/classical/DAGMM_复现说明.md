# DAGMM 复现说明（SWaT / WADI）

参考实现：[tnakae/DAGMM](https://github.com/tnakae/DAGMM)
原始论文：Zong et al., *Deep Autoencoding Gaussian Mixture Model for
Unsupervised Anomaly Detection*, ICLR 2018。

参考仓库是 TensorFlow 1.x 实现；本目录用 **PyTorch** 逐项复刻，
并接入项目统一两阶段框架。**结论：完整 PyTorch 复现可行，不需要
TensorFlow。**

---

## 1. PyTorch 复刻映射

| 参考实现（TF1） | 本实现（PyTorch） |
|---|---|
| CompressionNet：Encoder MLP(tanh) + Decoder MLP(tanh) | `CompressionNet`（`nn.Sequential`） |
| 重构特征 `loss_E = ||x-x'||/(||x||+1e-3)`、`loss_C = 0.5(1-cos)` | 同一公式 |
| 特征拼接 `z=[z_c, loss_E, loss_C]` | `torch.cat` |
| EstimationNet：MLP + dropout + softmax | `EstimationNet` |
| GMM：phi/mu/sigma 由 batch 内 z/gamma 估计 | `GMM.estimate` |
| Cholesky + `-logsumexp` 计算 energy | `GMM.energy`（`torch.linalg.cholesky` / `solve_triangular`） |
| 损失 `recon + λ1*mean(energy) + λ2*sum(1/diag(σ))` | 同一公式 |
| 训练结束后用完整训练集固定 GMM | `DAGMM.fix_gmm` |
| StandardScaler 归一化 | 框架 `--scaler standard` |

默认超参（参考实现 / KDDCup 示例）：

```text
comp_hiddens = [60,30,10,1]
est_hiddens  = [10,4]      # 最后一维 = GMM 组件数
dropout      = 0.5
lr           = 1e-4
batch_size   = 1024
epochs       = 100
lambda1      = 0.1
lambda2      = 1e-4
```

DAGMM 原论文是逐样本（非时序）模型；本复现把每个时刻的传感器向量
作为一个样本，逐点计算 energy，再用 12 步滑动窗口聚合成窗口分数
（mean / max / top3_mean），与项目统一评估协议对齐。

## 2. 运行命令

```bash
# 统一框架协议
python -m models.classical.dagmm --protocol framework \
    --csv 08_SWaT/SWaT_all_1s.csv
python -m models.classical.dagmm --protocol framework \
    --csv 09_WADI/WADI_all_1s.csv

# 全 Attack 段评估
python -m models.classical.dagmm --protocol paper \
    --csv 08_SWaT/SWaT_all_1s.csv
python -m models.classical.dagmm --protocol paper \
    --csv 09_WADI/WADI_all_1s.csv

# 复用已训练 artifact 只做检测
python -m models.classical.dagmm --protocol framework \
    --stage detect --artifact-dir runs/dagmm_swat
```

产物：

```text
runs/dagmm_swat/            # artifact + detection + training_history.csv
runs/dagmm_wadi/
runs/dagmm_paper_swat/      # metrics.json / window_scores.csv / roc.png
runs/dagmm_paper_wadi/
```

## 3. 结果

> SWaT/WADI 训练/评估完成后补入。

### 3.1 统一框架协议（验证集选阈值）

| 数据集 | level | 选中分数 | AUC | PR-AUC | Precision | Recall | F1 | FPR |
|---|---|---|---|---:|---:|---:|---:|---:|
| SWaT | window | dagmm_max | 0.8084 | 0.3693 | 0.4019 | 0.7698 | 0.5281 | 0.1837 |
| SWaT | point | point_dagmm_window_max | 0.8095 | 0.3687 | 0.4006 | 0.7718 | 0.5274 | 0.1839 |
| WADI | window | dagmm_mean | 0.5197 | 0.1168 | 0.0689 | 0.9966 | 0.1288 | 0.8960 |
| WADI | point | point_dagmm | 0.5188 | 0.1124 | 0.0678 | 0.9962 | 0.1269 | 0.8953 |

> WADI 上 DAGMM 的 AUC 约 0.52（接近随机），F1 0.129；验证集阈值
> 几乎把所有点判为异常（Recall 0.997、FPR 0.896）。这与文献中
> DAGMM 在 WADI 上的弱表现一致（IJCAI'24：AUC 29.2 / F1 20.9）。
> DAGMM 是非时序模型，WADI 维度约为 SWaT 两倍且分布漂移严重。

> SWaT 的 `dagmm_mean` 其实有更高的 AUC 0.8222 / PR-AUC 0.4186，
> 但验证集 F1（0.381）略低于 `dagmm_max`（0.390），最终按 val_F1
> 选中了 `dagmm_max`。窗口 F1 0.528，与 Isolation Forest 的 0.533
> 很接近；DAGMM 的 AUC 0.808 低于 IF 的 0.871。

### 3.2 全 Attack 段（oracle 阈值）

| 数据集 | 分数 | AUC | PR-AUC | 窗口 F1@EER | 窗口 F1@oracle | PA-F1@oracle |
|---|---|---:|---:|---:|---:|---:|
| SWaT | dagmm_mean | 0.8180 | 0.4080 | 0.4551 | 0.6561 | 0.8014 |
| SWaT | dagmm_max | 0.8133 | 0.3656 | 0.4627 | 0.6268 | **0.8400** |
| SWaT | dagmm_top3_mean | 0.8147 | 0.3731 | 0.4635 | 0.6324 | 0.8244 |
| WADI | dagmm_mean | 0.5214 | 0.0900 | 0.0819 | 0.2226 | **0.5130** |
| WADI | dagmm_max | 0.5192 | 0.0839 | 0.0807 | 0.2124 | 0.4857 |
| WADI | dagmm_top3_mean | 0.5196 | 0.0848 | 0.0807 | 0.2142 | 0.4862 |

> SWaT 的 PA-F1 0.801–0.840 与文献 85.3 同一量级；
> AUC 0.818 低于文献 91.6（AUC/预处理口径不同）。统一框架下
> 窗口 F1 0.528（val 阈值），与 Isolation Forest 的 0.533 接近。
>
> WADI 的全 Attack 指标由 ``val.npz + test.npz`` 拼接计算
> （``models/classical/full_attack_from_artifact.py``）：窗口分数
> 直接拼接，逐点分数按窗口结束时间对齐并零阶保持；与重新加载
> 原始数据的 paper 协议在跨界处有极小差异。
>
> WADI 全 Attack 段 AUC 0.521、oracle 窗口 F1 0.223、
> PA-F1 0.513；文献 DAGMM WADI 为 AUC 29.2 / F1 20.9，
> 窗口 F1 与文献一致，AUC/PA 口径不同。DAGMM 在 WADI 上
> 基本失效，符合“非时序 + 高维 + 分布漂移”的预期。

### 3.3 文献中 DAGMM 的报告值

- IJCAI'24 基准表：SWaT AUC 91.6 / F1 85.3；
  WADI AUC 29.2 / F1 20.9。
- Time-series AD survey（CSA/CSDN 转引）：
  SWaT P 90.60 / R 80.72 / F1 85.38；
  WADI P 22.28 / R 19.76 / F1 20.94。
- 其他论文因协议不同差异很大：SWaT F1 0.55–0.85，
  WADI F1 0.14–0.70（有的使用 point-adjust/不同预处理）。
- 共同结论：DAGMM 是非时序模型，在 WADI（维度约为 SWaT 两倍）
  上明显退化；IJCAI 表给出的 WADI F1 仅约 0.21。

## 4. 复现要点与偏差

1. **逐点而非窗口输入**：参考实现处理表格样本；本复现把每个时刻
   的传感器向量作为样本。窗口分数只是评估用的聚合，不参与训练。
2. **GMM 固定化**：训练时每个 batch 重新估计 phi/mu/sigma；
   训练结束后用完整训练集的 z/gamma 重新估计并固定，再打分。
3. **数值稳定性**：协方差 Cholesky 前加 1e-6 I；gamma 权重
   clamp 到 1e-6，避免空组件除零。
4. **批量/轮数**：batch=1024、epochs=100（参考 KDDCup 200）；
   可通过 CLI 改回。
5. **lambda2**：参考实现默认 1e-4（论文里 0.005 容易让协方差
   过大、检测退化）；本实现沿用 1e-4。
6. **输入归一化**：框架 StandardScaler；paper 协议在 Normal 上
   fit StandardScaler 后 transform Attack。
7. **WADI 越界值**：DAGMM 的重构网络是线性输出，不会像 sigmoid/
   tanh 模型那样被 [0,1] 边界卡住；但仍建议关注极端值对 energy
   的影响。

## 5. 结论

1. **PyTorch 复现可行且已完成**：压缩网络、估计网络、批内 GMM、
   Cholesky 能量、GMM 固定化和训练循环全部用 PyTorch 实现，
   不依赖 TensorFlow。
2. **SWaT 达到文献量级**：统一框架窗口 AUC 0.808 / F1 0.528；
   全 Attack 段 AUC 0.818 / PR-AUC 0.408、oracle 窗口 F1 0.656、
   PA-F1 0.801–0.840。文献 DAGMM SWaT 为 AUC 91.6 / F1 85.3，
   F1 同一量级，AUC 偏低（预处理/评估口径不同）。
3. **WADI 上 DAGMM 基本失效**：统一框架 AUC 0.520 / F1 0.129；
   全 Attack 段 AUC 0.521 / oracle 窗口 F1 0.223 / PA-F1 0.513。
   文献 DAGMM WADI 为 AUC 29.2 / F1 20.9，窗口 F1 与之一致。
   这是非时序模型在高维、分布漂移数据上的典型退化。
4. **与 Isolation Forest 对比**：SWaT 框架 F1 0.528 vs IF 0.533，
   WADI 0.129 vs IF 0.138，两者接近；DAGMM 的 SWaT 全 Attack
   PA-F1（0.84）略高于 IF（0.824）。DAGMM 没有显著优于经典 IF。
5. **建议**：SWaT 上 DAGMM 可作为强基线；WADI 上应优先使用时序
   模型或先做分布对齐；报告时必须同时给出 AUC/PR-AUC 与
   P/R/F1/FPR，并注明阈值是否来自验证集。
