# Isolation Forest 复现说明（SWaT / WADI）

参考实现：[sklearn.ensemble.IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
原始论文：Liu, Ting & Zhou, *Isolation Forest*, ICDM 2008。

本目录把 sklearn IsolationForest 作为经典无监督基线接入项目统一
两阶段框架，并在整个 Attack 段上做补充评估。

---

## 1. 方法

1. **训练**：只使用 Normal 段拟合 `IsolationForest`（无监督，不需要标签）。
2. **逐点打分**：
   ```text
   score = -model.score_samples(X)
   ```
   sklearn 的 `score_samples` 越小越异常，取负后**越大越异常**。
3. **窗口聚合**：用长度 `seq_len=12` 的滑动窗口把逐点分数聚合成
   `iforest_mean`、`iforest_max`、`iforest_top3_mean`，与项目其它
   重构模型的窗口分数对齐；窗口标签为“窗口内是否含攻击点”。
4. **阈值**：
   - framework 协议：Attack 前 20% 验证集选阈值，后 80% 测试；
   - paper 协议：整个 Attack 段，报告 EER / oracle / PA-F1。

默认参数：

```text
n_estimators=100
max_samples='auto'（sklearn 取 min(256, n_samples)）
contamination='auto'（只影响 offset_；本实现用 score_samples，排序不受影响）
max_features=1.0
bootstrap=False
random_state=42
n_jobs=-1
```

预处理：StandardScaler（IsolationForest 对单调缩放不敏感，保留 scaler
只为与框架的 artifact 格式一致）。

## 2. 运行命令

```bash
# 统一框架协议
python -m models.classical.isolation_forest --protocol framework \
    --csv 08_SWaT/SWaT_all_1s.csv
python -m models.classical.isolation_forest --protocol framework \
    --csv 09_WADI/WADI_all_1s.csv

# 全 Attack 段评估
python -m models.classical.isolation_forest --protocol paper \
    --csv 08_SWaT/SWaT_all_1s.csv
python -m models.classical.isolation_forest --protocol paper \
    --csv 09_WADI/WADI_all_1s.csv

# 复用已拟合模型只做检测
python -m models.classical.isolation_forest --protocol framework \
    --stage detect --artifact-dir runs/iforest_swat
```

产物：

```text
runs/iforest_swat/            # artifact + detection
runs/iforest_wadi/            # artifact + detection
runs/iforest_paper_swat/      # 全 Attack 段 metrics.json / window_scores.csv
runs/iforest_paper_wadi/
```

## 3. 结果

> SWaT 与 WADI 均已完成（统一框架协议 + 全 Attack 段协议）。

### 3.1 统一框架协议（验证集选阈值）

| 数据集 | level | 选中分数 | AUC | PR-AUC | Precision | Recall | F1 | FPR |
|---|---|---|---|---:|---:|---:|---:|---:|
| SWaT | window | iforest_mean | 0.8713 | 0.7871 | 0.4104 | 0.7616 | 0.5334 | 0.1754 |
| SWaT | point | point_iforest | 0.8727 | 0.7886 | 0.4076 | 0.7650 | 0.5318 | 0.1770 |
| WADI | window | iforest_mean | 0.7824 | 0.2239 | 0.0747 | 0.9125 | 0.1380 | 0.7520 |
| WADI | point | point_iforest | 0.7828 | 0.2158 | 0.0737 | 0.9175 | 0.1365 | 0.7530 |

### 3.2 全 Attack 段（oracle 阈值）

| 数据集 | 分数 | AUC | PR-AUC | 窗口 F1@EER | 窗口 F1@oracle | PA-F1@oracle |
|---|---|---:|---:|---:|---:|---:|
| SWaT | iforest_mean | 0.8321 | 0.7248 | 0.4195 | 0.7306 | **0.8238** |
| SWaT | iforest_max | 0.8331 | 0.7173 | 0.4225 | 0.7232 | 0.8132 |
| SWaT | iforest_top3_mean | 0.8331 | 0.7176 | 0.4235 | 0.7233 | 0.8131 |
| WADI | iforest_mean | 0.7529 | 0.2343 | 0.2202 | 0.3444 | **0.4942** |
| WADI | iforest_max | 0.7350 | 0.1662 | 0.2137 | 0.2963 | 0.5807 |
| WADI | iforest_top3_mean | 0.7358 | 0.1677 | 0.2134 | 0.2982 | 0.5804 |

> SWaT 的结果落在文献报告的区间内：AUC 0.832（IEEE CAI
> 0.8426）、oracle 窗口 F1 0.731（HFN 0.727）、PA-F1 0.824；
> 统一框架下窗口 F1 0.533、point F1 0.532，优于 USAD/LSTM-VAE
> 在该协议下的 0.25–0.27。说明 IF 在 SWaT 上仍然是很强的经典
> 基线。
>
> WADI 的 AUC 0.782（文献 0.708–0.77 区间）、框架 F1 0.138
> （文献 0.055–0.25 区间）；Precision 只有 0.075、Recall 0.91，
> 说明验证集阈值在 WADI 上严重偏低，导致大量误报。IF 在 WADI
> 上的排序能力尚可，但阈值迁移和类别不平衡使其 F1 很低。
>
> WADI 全 Attack 段：AUC 0.753（文献 0.708–0.77）、F1@EER 0.220
> （文献无 PA F1 0.207–0.25）、oracle 窗口 F1 0.344、
> PA-F1 0.494。与文献区间一致，说明 IF 在 WADI 上确实只能达到
> 这个量级（远低于 SWaT）。

### 3.3 文献中 Isolation Forest 的报告值

不同论文差异很大（评估口径、预处理、point-adjust 不一致）：

- IEEE CAI 2024：SWaT F1 0.3502 / AUC 0.8426 / AUPR 0.7577；
  WADI F1 0.0554 / AUC 0.7080 / AUPR 0.0987。
- HFN（Lancaster）：SWaT P 0.951 / R 0.588 / F1 0.727；
  WADI P 0.299 / R 0.158 / F1 0.207。
- PLOS ONE：SWaT P 49.29% / R 44.95% / F1 47.02%。
- RNTI：SWaT F1 0.28 / AUC 0.35；WADI F1 0.25 / AUC 0.77。

因此对照时应优先看 AUC/PR-AUC，并注明阈值和 point-adjust 口径。

## 4. 复现要点

1. **逐点 vs 窗口**：sklearn IF 是表格模型；本实现先对逐点特征打
   分数，再用滑动窗口聚合。这样计算量小（无需把 12×C 展平后
   拟合），也能直接复用统一框架的窗口评估。
2. **无监督**：IF 只用 Normal 拟合；验证/测试标签不参与训练。
3. **阈值**：framework 协议用验证集 max-F1；paper 协议额外给出
   EER/oracle/PA-F1。oracle 阈值使用测试标签，仅作上界。
4. **max_samples**：sklearn 默认 `auto` = min(256, n_samples)；
   对 50 万级样本，每棵树只用 256 个点，训练很快。
5. **contamination**：本实现使用 `score_samples` 排序，
   `contamination` 只影响 `offset_/decision_function`，不影响
   AUC/F1 的排序结果；保留该参数是为了完整复现 sklearn API。

## 5. 结论

1. **SWaT 达到文献量级且偏强**：全 Attack 段 AUC 0.832、PR-AUC
   0.725、oracle 窗口 F1 0.731、PA-F1 0.824；文献 IF 的 AUC
   约 0.84、F1 在 0.35–0.73 之间。统一框架（验证集阈值）下
   窗口 F1 0.533、point F1 0.532，优于 USAD/LSTM-VAE 在该协议下
   的 0.25–0.27。
2. **WADI 明显更难**：全 Attack 段 AUC 0.753、F1@EER 0.220、
   oracle 窗口 F1 0.344、PA-F1 0.494；文献 IF 的 AUC 0.708–0.77、
   F1 0.055–0.25。与文献一致，WADI 上 IF 只能达到这个量级。
3. **统一框架的阈值迁移**：WADI 验证集阈值对应 test Precision
   仅 0.075、Recall 0.91、FPR 0.75；SWaT 的验证集阈值则较好地
   迁移（F1 0.53）。WADI 的分数漂移/类别不平衡是主要问题。
4. **方法本身**：IF 只用 Normal 拟合、无需训练迭代，速度远快于
   深度模型；在 SWaT 上 F1 甚至高于 USAD/LSTM-VAE 的统一框架
   结果，是很强的经典基线。
5. **建议**：SWaT 上 IF 可直接作为强 baseline；WADI 上应结合
   目标 FPR 校准或滚动阈值，并报告 AUC/PR-AUC 与 F1/FPR 两组
   指标，避免只看单一 F1。
