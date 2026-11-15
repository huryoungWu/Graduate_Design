# ADT 复现说明（SWaT / WADI）

参考实现：[XueYang0130/ADT](https://github.com/XueYang0130/ADT)
论文：

- Yang, Howley & Schukat, *Agent-based dynamic thresholding for adaptive
  anomaly detection using reinforcement learning*, Neural Computing and
  Applications, 2024.
- Yang, Howley & Schukat, *ADT: Time series anomaly detection for
  cyber-physical systems via deep reinforcement learning*,
  Computers & Security 141 (2024) 103825.

参考仓库是 TensorFlow/Keras 实现；当前环境没有安装 TensorFlow，
本目录用 **PyTorch** 逐项复刻 AE 与 DQN，并保留仓库的评估流程。

---

## 1. 方法

1. **数据**：窗口长度 12，展平为 `L × C`；窗口标签 = 窗口内是否含攻击点。
2. **AE**（PyTorch MLP）：
   ```text
   window_dim -> window_dim/2 -> window_dim/4 -> latent=100
              -> window_dim/4 -> window_dim/2 -> window_dim (Sigmoid)
   ```
   MSE 训练；每个窗口的异常分 = 重构 MSE。
3. **DQN**（PyTorch MLP 6→32→32→16→2）：
   - 状态：最近 k 个窗口的
     `[mean_score, var_score, TP_rate, FP_rate, TN_rate, FN_rate]`；
   - 动作：阈值 0/1，`score >= threshold` 判异常；
   - 奖励：预测正确 +1，错误 -1；
   - ε-greedy、replay memory、target network。
4. **评估**：DQN 在窗口序列上动态切换阈值，统计 TP/TN/FP/FN、
   Precision/Recall/F1。

默认超参（仓库设置）：

```text
seq_len=12, latent=100
AE: epochs=30, batch=8000, lr=1e-3, 70% Normal 窗口
DQN: episodes=2000, batch=256, lr=1e-4, gamma=0.99,
     l_action=10 (训练), l_action_infer=1 (README), k_state=1
```

## 2. 两种协议

### paper（复刻仓库/论文）

- AE 用 70% Normal 窗口训练；
- DQN 用一段带标签的 Attack 窗口训练：
  **SWaT [1500:2000]，WADI [59000:60000]**；
- 在整个 Attack 段上评估。

### unified（项目统一协议）

- AE 用 Normal 训练；
- DQN 只用 Attack 前 20%（验证段）的 500/1000 个窗口训练；
- 在 Attack 后 80% 测试段评估；
- MinMaxScaler 只在 Normal 上 fit，Attack 输入裁剪到 [0,1]。

## 3. 重要说明（必须与结果一起看）

1. **推理阶段使用真实标签**：仓库的 `Env.do_step` 在计算 TP/FP/TN/FN
   状态和奖励时使用 `self.y`（真实标签）。因此 ADT 是一个**带在线
   标签反馈的动态阈值控制器**，不是纯无监督检测器；高 F1 部分来自
   这一反馈。配套论文专门研究了噪声/延迟/部分反馈下的鲁棒性。
2. **训练片段泄漏（paper 协议）**：DQN 的训练片段属于最终评估的
   Attack 段，属于 transductive 设置；unified 协议去掉了这一点。
3. **归一化泄漏（paper 协议）**：仓库在 Normal+Attack 全体上 fit
   MinMaxScaler；本复现默认保留以对齐论文，unified 改为只在
   Normal 上 fit。
4. **AE-only 参考**：报告中同时给出 AE 重构分数的 AUC 和 best-F1，
   便于区分“AE 本身的能力”和“DQN 动态阈值带来的提升”。

## 4. 运行命令

```bash
# 论文/仓库协议
python -m models.classical.adt --protocol paper \
    --csv 08_SWaT/SWaT_all_1s.csv
python -m models.classical.adt --protocol paper \
    --csv 09_WADI/WADI_all_1s.csv

# 统一协议（DQN 只用 val 标签）
python -m models.classical.adt --protocol unified \
    --csv 08_SWaT/SWaT_all_1s.csv
python -m models.classical.adt --protocol unified \
    --csv 09_WADI/WADI_all_1s.csv

# 快速自检
python -m models.classical.adt --protocol paper \
    --csv 08_SWaT/SWaT_all_1s.csv --smoke
```

产物：

```text
runs/adt_paper_swat/     # metrics.json / ae_scores.npy / dqn.pt / ae.pt
runs/adt_paper_wadi/
runs/adt_swat/           # unified
runs/adt_wadi/
```

## 5. 结果

> SWaT 已完成；WADI 完成后补入。

### 5.1 paper 协议（全 Attack 段）

| 数据集 | AE AUC | AE best F1 | DQN Precision | DQN Recall | DQN F1 | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---|
| SWaT | 0.8582 | 0.7348 | 0.9993 | 0.9994 | **0.9993** | TP 54971 / TN 394865 / FP 37 / FN 35 |
| WADI | 0.6272 | 0.3481 | 0.1107 | 0.9996 | 0.1994 | TP 10127 / TN 81327 / FP 81332 / FN 4 |

### 5.2 unified 协议（Attack 后 80%）

| 数据集 | AE AUC | AE best F1 | DQN Precision | DQN Recall | DQN F1 | TP/TN/FP/FN |
|---|---:|---:|---:|---:|---:|---|
| SWaT | 0.8720 | 0.7670 | 0.9994 | 0.9995 | **0.9995** | TP 49694 / TN 310179 / FP 28 / FN 26 |
| WADI | 0.7154 | 0.4269 | 0.1173 | 0.9995 | 0.2100 | TP 8615 / TN 64804 / FP 64809 / FN 4 |

> SWaT 在统一协议下仍然 F1 0.9995：DQN 只用验证段的一个
> 小片段训练，但推理时的标签反馈使其能在测试段动态切换。
> AE-only 在测试段 AUC 0.872、best F1 0.767；提升同样来自
> 动态阈值/标签反馈，而不是分数本身。
>
> WADI 统一协议仍然失败（F1 0.210）：AE 在测试段的 AUC 0.715、
> best F1 0.427，DQN 再次退化为“几乎全部报警”（FP 64,809 /
> FN 4）。这说明问题不是 DQN 训练片段落在测试段（unified 已经
> 去掉这一点），而是 WADI 上 AE 分数判别力弱、策略无法从验证段
> 迁移到测试段。

> SWaT：AE-only AUC 0.8582 / best F1 0.7348；DQN 动态阈值把 F1
> 提升到 **0.9993**，与论文 SWaT 的 0.995 一致。需要注意：DQN 的
> 状态/奖励使用真实标签计算 TP/FP/TN/FN，因此这是带在线标签反馈
> 的动态阈值控制器，不是纯无监督检测；AE-only 的 0.735 更能代表
> 无标签反馈时的检测能力。
>
> **逐点评估（paper, SWaT）**：`models/classical/adt_point_eval.py`
> 把窗口预测映射到点后：
>
> | 口径 | Precision | Recall | F1 |
> |---|---:|---:|---:|
> | 窗口级 | 0.9993 | 0.9994 | 0.9993 |
> | 逐点 raw（窗口末点映射） | 0.9923 | 0.9994 | **0.9958** |
> | 逐点 PA（窗口末点映射） | 0.9923 | 1.0000 | 0.9962 |
> | 逐点 raw（窗口内膨胀） | 0.9859 | 1.0000 | 0.9929 |
> | 逐点 PA（窗口内膨胀） | 0.9859 | 1.0000 | 0.9929 |
>
> 逐点 AE 分数 AUC 0.8602。逐点 F1 只比窗口级低约
> 0.003–0.006，因为 DQN 检测到了攻击段内几乎所有窗口；
> 差距主要是每段结束后约 11 个边界点被误报。也就是说，
> **0.999 主要来自标签反馈/段跟踪，窗口级标签只是次要的
> 放大因素**。但逐点 PA 仍会因 point-adjust 而偏高。
>
> **WADI 未能复现论文的 0.999**：AE-only AUC 仅 0.627、best F1
> 0.348；DQN 在完整 Attack 段上退化为“几乎全部报警”
> （P 0.111 / R 0.9996 / F1 0.199，FP 81,332 / FN 4）。原因：
> 1) 本数据集的 AE 分数在 WADI 上判别力弱；
> 2) 仓库的 DQN 训练片段 [59000:60000] 正例率约 **60%**，
>    而完整 WADI 只有 **5.86%**；策略在该高正例片段上学到
>    “多报警”，迁移到低正例的全段后大量误报；
> 3) 本数据集删除了 29 个恒定列（98 vs 123 维），与仓库原始
>    WADI 预处理不同。
> 这与论文“ADT 不依赖分数质量”的强结论不一致；至少在本数据
> 和该训练片段下，ADT 的表现高度依赖 AE 分数与片段分布。

### 5.3 WADI 训练片段敏感性（诊断，非主协议）

固定 paper 协议的 AE 分数，只改变 DQN 的训练片段（长度 1000），
在完整 WADI Attack 段上评估：

| 片段起点 | 片段正例率 | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 4151 | 0.059 | 0.9998 | 0.5000 | 0.6666 |
| 4192 | 0.100 | 0.9986 | 0.9986 | **0.9986** |
| 4292 | 0.200 | 0.9998 | 0.5000 | 0.6666 |
| 4492 | 0.400 | 0.9998 | 0.5000 | 0.6666 |
| 4692 | 0.600 | 0.9984 | 0.9986 | **0.9985** |
| 59000（仓库默认） | ≈0.60 | 0.1107 | 0.9996 | 0.1994 |

结论：WADI 上 ADT 的表现**对训练片段极其敏感**。同一 AE 分数下，
起点 4192（10% 正例）和 4692（60% 正例）的片段都能达到
F1 ≈ 0.999，与论文一致；而仓库默认的 59000 片段在本数据上退化
为几乎全报警。正例率与全段一致的片段只有 F1 0.667（漏掉一半
攻击），说明“片段分布匹配”不是充分条件，片段的分数/攻击模式
也会影响策略泛化。

注意：该诊断是**事后选择片段**，存在用测试结果选超参的嫌疑，
不能当作论文协议的正式结果；它只用于解释 WADI 失败的原因。
诊断原始数据见 `runs/adt_paper_wadi/segment_diagnostic.json`。

### 5.4 论文报告值

- Computers & Security 2024：ADT 在 SWaT/WADI/HAI 上 F1
  **0.995–0.999**；摘要称即使只用少量标签数据也能达到该水平。
- arXiv:2312.01488：SWaT 分类准确率约 **0.9994**；α/β/l 消融中
  F1 约 0.999。
- 这些数字是在“带标签反馈 + 训练片段属于评估段”的设置下得到的；
  对照时必须注明。

## 6. 复现要点与偏差

1. **TensorFlow -> PyTorch**：AE 和 DQN 都是小 MLP，PyTorch 复刻
   完全可行；DQN 逐点推理用 numpy 前向加速（数值等价）。
2. **数据**：使用项目处理后的 45/98 维数据，而不是仓库原始的
   51/123 维；窗口数也略有差异（T-L+1 vs T-L）。
3. **Scaler**：paper 协议默认 all-data MinMax（仓库做法）；
   unified 只用 Normal。ADT 的 AE 末端是 Sigmoid，Normal-fit 时
   对 Attack 输入做 [0,1] 裁剪。
4. **DQN 训练规模**：episodes=2000、l_action=10、batch=256
   与仓库一致；推理 l_action=1（README 默认），也可用
   `--l-action-infer 10` 对齐代码默认。
5. **随机性**：DQN 的 ε-greedy/replay 采样随机；固定 seed=42，
   但不同实现/硬件可能略有差异。

## 7. 结论

1. **PyTorch 复现可行且已完成**：AE 与 DQN 都是小 MLP；PyTorch
   实现与仓库结构一致，无需 TensorFlow。
2. **SWaT 完全复现论文量级**：
   - paper 协议：AE-only AUC 0.8582 / best F1 0.7348；DQN
     P 0.9993 / R 0.9994 / **F1 0.9993**；
   - unified 协议：AE-only AUC 0.8720 / best F1 0.7670；DQN
     P 0.9994 / R 0.9995 / **F1 0.9995**。
   与论文 SWaT 的 F1 0.995 一致。提升完全来自 DQN 动态阈值
   （以及推理时的标签反馈），而不是 AE 分数本身。
3. **WADI 没有复现论文的 0.999**：
   - paper 协议：AE-only AUC 0.6272 / best F1 0.3481；DQN
     **F1 0.1994**（P 0.1107 / R 0.9996，FP 81,332 / FN 4）；
   - unified 协议：AE-only AUC 0.7154 / best F1 0.4269；DQN
     **F1 0.2100**（P 0.1173 / R 0.9995，FP 64,809 / FN 4）。
   两次都退化为“几乎全部报警”。可能原因：
   - WADI 上 AE 分数判别力弱（AUC 0.63–0.72）；
   - 仓库 DQN 训练片段 [59000:60000] 正例率约 60%，完整 WADI
     只有 5.86%，策略在低正例全段上严重过报；
   - 本数据集删除 29 个恒定列（98 vs 123 维），与仓库原始
     WADI 预处理不同；
   - DQN 的 replay 存储的是动作后的 state（参考实现的写法），
     也会影响策略泛化。
   - **片段敏感性诊断**（5.3）：同一 AE 分数下，起点 4192（10%
     正例）和 4692（60% 正例）的片段都能达到 F1 ≈ 0.999，说明
     仓库默认片段 59000 在本数据上恰好学到了坏策略；WADI 的
     0.999 并非不可复现，但对训练片段/数据分布高度敏感。
4. **必须注明的评估性质**：推理时 Env 使用真实标签计算
   TP/FP/TN/FN 状态/奖励，ADT 是“带在线反馈的动态阈值控制器”，
   不是纯无监督检测器；paper 协议还存在训练片段属于评估段、
   MinMax 在全体数据上 fit 的泄漏。SWaT 的近 1.0 F1 不能与
   无监督基线的 F1 直接比较。
5. **建议**：SWaT 上 ADT 可以复现论文水平；WADI 上需要更好的
   AE 分数/更代表性的训练片段，或使用论文配套的反馈鲁棒性设置；
   论文“ADT 不依赖分数质量”的强结论至少在本数据/片段下不成立。
