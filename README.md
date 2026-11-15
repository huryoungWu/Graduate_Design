# GraduateDesign 处理后数据集

生成日期：2026-11-04  
原始数据：`D:\GraduateDesign\Dataset`  
输出目录：`C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset`

> 说明：`D:\GraduateDesign` 的 NTFS 权限对当前用户只有读取权限（Users: RX），无法写入，因此输出放在 Documents。若需放回项目目录，请用资源管理器或管理员权限复制。

## 1. 统一格式

- 全部为 CSV（UTF-8 with BOM），Excel 可直接打开。
- 第一列 `timestamp`，格式 `YYYY-MM-DD HH:MM:SS`，按时间升序。
  - 例外：`10_WDSEventDB` 原始数据无绝对时间戳，使用 `relative_time_s`（按 1 Hz 假设，第 n 行 = n 秒）。
- 每一行是一个时间点；每一列是一个变量。
- 缺失值留空（空字符串），不插值、不填充。
- 浮点数保留约 10 位有效数字。
- 标签/元数据列：
  - `label`：0/1 异常或攻击标签；
  - `leak_label`：LeakDB 的逐点泄漏标签；
  - `any_leak`：BattLeDIM 是否至少一条管段泄漏；
  - `event`：GECCO 事件标签；
  - `subset` / `split` / `period`：数据来源或划分。

## 2. 文件清单

完整清单见 `MANIFEST.csv`（自动生成，含行数、列数、大小、时间范围）。主要文件：

| 目录 | 文件 | 行 × 列 | 说明 |
|---|---|---|---|
| 01_2026_multisource | `2026_multisource_hourly.csv` | 8,737 × 21 | 18 个异常分数 + `fault_d7` + `label_reliable` |
| 02_AquaSense | `aquasense_real_labeled.csv` | 23,688 × 15 | 4 个真实传感器 + 质量/攻击标签 |
| 02_AquaSense | `aquasense_attack_segments.csv` | 14 × 6 | 攻击段起止/类型/通道 |
| 03_BATADAL | `BATADAL_all.csv` | 15,027 × 46 | 43 个特征 + label + subset |
| 04_BattLeDIM | `BattLeDIM_2018_5min.csv` | 105,120 × 138 | 119 条 SCADA + 14 条泄漏 + 汇总/起始标记 |
| 04_BattLeDIM | `BattLeDIM_2019_5min.csv` | 105,120 × 147 | 119 条 SCADA + 23 条泄漏 + 汇总/起始标记 |
| 04_BattLeDIM | `BattLeDIM_*_leak_events.csv` | 14 / 23 × 9 | 泄漏事件起止/峰值/时长 |
| 05_GECCO | `GECCO_all_1min.csv` | 218,880 × 9 | 6 变量 + event + split |
| 05_GECCO | `GECCO_events.csv` | 37 × 4 | 37 个连续事件区间 |
| 06_LeakageRepairs | `LeakageRepairs_flows_5min.csv` | 331,776 × 8 | 6 个 DMA 流量 + 有效区数 |
| 06_LeakageRepairs | `LeakageRepairs_reports.csv` | 2,577 × 7 | 维修记录 |
| 07_LeakDB | 9 个场景 CSV | 17,520 × 37~102 | 逐点标签 + 需求/流量/压力/泄漏 |
| 07_LeakDB | `LeakDB_all_scenarios_index.csv` | 2,500 × 4 | 全部场景是否含泄漏 |
| 08_SWaT | `SWaT_all_1s.csv` | 944,919 × 48 | 正常+攻击，1 s，51 特征减恒定列 |
| 08_SWaT | `SWaT_attack_segments.csv` | 35 × 3 | 攻击段 |
| 09_WADI | `WADI_all_1s.csv` | 957,372 × 101 | 正常+攻击，1 s，127 特征减恒定列 |
| 09_WADI | `WADI_attack_segments.csv` | 14 × 3 | 攻击段 |
| 10_WDSEventDB | `WDSEventDB_all_events.csv` | 25,165 × 21 | 15 变量 + label + 元数据 |
| 10_WDSEventDB | `WDSEventDB_file_summary.csv` | 9 × 5 | 每文件行数/正类率 |

## 3. 各数据集处理说明

### 01_2026_multisource

- 按 `DD-MM-YYYY HH:00` 解析时间，按时间升序排序。
- 新增 `label_reliable`：原始文件前 289 行是各月 1 日块，排序后这些行的 `fault_d7` 可能与时间不对应，标记为 0；其余为 1。
- 注意：正类占 73.6%，标签是“泄漏前后 ±7 天”的粗区间，不适合逐点异常检测；18 列均为已算好的异常分数，不是原始传感器读数。

### 02_AquaSense

- 仅保留 `fused_data.csv` 中 `origin=real` 的 23,688 行；不包含 TimeGAN 合成数据和 EPANET 仿真列。
- 列：4 个真实传感器 + `quality_legacy`（Real_data 的 4 类质量标签）/ `quality_final` / `wq_score` / `wq_class` + 攻击/故障标签；新增 `attack_label = attack_id > 0`。
- `aquasense_attack_segments.csv`：14 个攻击段。
- 注意：有效非缺口时长约 7.9 h；`splits.json` 的 test 切分只有 replay 攻击。

### 03_BATADAL

- `BATADAL_all.csv`：dataset03（2014-01-06 ~ 2015-01-06，正常，label=0）+ dataset04（2016-07-04 ~ 2016-12-25，label 0/1，219 个攻击点）+ 2017 无标签测试集（2017-01-04 ~ 2017-04-01）。
- `subset` 区分三段；2017 段 `label` 为空。
- 日期按 `dayfirst=True` 解析，标签 -999 已映射为 0。

### 04_BattLeDIM

- 2018/2019 各 105,120 行、5 min；119 条 SCADA 序列（33 压力 + 3 流量 + 1 液位 + 82 需求）。
- 列名前缀：`flow_`、`pressure_`、`level_`、`demand_`、`leak_`；另有 `total_leak_m3h`、`any_leak`、`leak_onset_count`、`leak_onset`。
- `leak_onset` = 该 5 min 时刻是否有管段泄漏从 0 变为正（事件起始）；2018 年 14 个、2019 年 19 个（2019 年 4 条延续泄漏的起始未观测到）。
- `BattLeDIM_*_leak_events.csv`：每条管段泄漏的起止、持续时长、最大/平均泄漏、峰值时间、`onset_observed`（0 表示年初已存在、起始未观测到）。
- 注意：2018 年 97.8% 时间点至少一条管段泄漏，2019 年 100%；`any_leak` 不能直接做逐点二分类，应使用 `leak_onset` 或泄漏事件表。

### 05_GECCO

- `GECCO_all_1min.csv`：1 min，6 个水质变量，`event` 0/1，`split` 为 train/valid/test。
- 6 个变量的 0 值按缺失处理；valid 段缺失约 4.1%，总体约 1.15%。
- `GECCO_events.csv`：**37 个连续事件区间（628 个事件分钟）**，时长 4~41 min。
- 修正：之前口头说“628 个事件”不准确，628 是 `Event=1` 的分钟数，独立连续事件区间是 37 个。

### 06_LeakageRepairs

- `LeakageRepairs_flows_5min.csv`：6 个 DMA 的 5 min 流量，331,776 行。
- 清洗：负值/零值置空；每区按 `max(q99.9×3, median×10)` 截断极端值；按 5 min 重采样；缺口保留为空，不插值。
- `LeakageRepairs_reports.csv`：2,577 条维修记录，`severity` 缺失填 `unknown`。
- `LeakageRepairs_area_summary.csv`：每区的阈值、截断行数、缺失情况。
- 注意：标签是维修工单时间，不是泄漏起止区间；需要按流量下降与工单匹配。

### 07_LeakDB

- 导出 9 个代表场景：Hanoi500 3 个、Hanoi1000 3 个、Net1 3 个；每个 17,520 行（1 年，30 min）。
- 列名前缀：`demand_`、`flow_`、`pressure_`、`leak_`；`leak_label` 为逐点泄漏标签（0/1）。
- `LeakDB_all_scenarios_index.csv`：全部 2,500 个场景的 `has_leak` 索引。
- `LeakDB_selected_scenarios.csv`：9 个导出场景的摘要（是否泄漏、正类比例、文件）。
- 未导出全部场景的原因：全部解压约 64 GB，转成宽表 CSV 会更大。需要更多场景时，修改 `_scripts/p07_leakdb.py` 的 `select_scenarios` 后重跑。
- 注意：1000 场景版本原始数据只有 `Index` 无时间戳，已按 2017-01-01 起、30 min 步长补 `timestamp`；500 场景版本使用原始 `Timestamp`。

### 08_SWaT

- `SWaT_all_1s.csv`：Normal_v1 + Attack_v0 合并，1 s，`label` 0/1，`period` normal/attack。
- 已修正 `A ttack` 拼写错误（37 行）；删除恒定列，列表见 `SWaT_dropped_constant_columns.txt`。
- `SWaT_attack_segments.csv`：35 段攻击。
- 注意：`SWaT_Dataset_Normal_v0.xlsx` 文件损坏，未使用；时间戳 2015/2016 格式不统一，已用 dayfirst + mixed 解析。

### 09_WADI

- `WADI_all_1s.csv`：Normal + Attack 合并，1 s，`label` 0/1，`period` normal/attack。
- 原始标签 1=正常、-1=攻击，已映射为 0/1；删除 29 个恒定列，列表见 `WADI_dropped_constant_columns.txt`。
- `WADI_attack_segments.csv`：14 段攻击。
- 注意：原始 `Time` 是 `MM:SS.t` 且每小时清零，已按“每个日期块从首个时间起逐秒累加”重建绝对时间；攻击文件末尾 2 行空行已丢弃。正常段本身有缺失日期（9/30~10/1 等），未插值；`WADI.A2` 与 `WADI.A1` 重复，只用了 A1。

### 10_WDSEventDB

- `WDSEventDB_all_events.csv`：9 个文件合并，15 个变量 + `label` + 元数据（`event_type`、`source_file`、`segment_id`、`row_index`、`relative_time_s`）。
- 原始数据无绝对时间戳；`relative_time_s` 按 1 Hz 假设生成（第 n 行 = n 秒），仅用于相对定位。
- `WDSEventDB_file_summary.csv`：每个文件的行数/正类率。

## 4. 读取示例

```python
import pandas as pd

base = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"

# 单文件时间序列
df = pd.read_csv(f"{base}/04_BattLeDIM/BattLeDIM_2019_5min.csv", parse_dates=["timestamp"])

# 按 period 取正常/攻击
swat = pd.read_csv(f"{base}/08_SWaT/SWaT_all_1s.csv", parse_dates=["timestamp"])
normal = swat[swat.period == "normal"]
attack = swat[swat.period == "attack"]

# LeakDB 单场景
lk = pd.read_csv(f"{base}/07_LeakDB/Hanoi500_Scenario-007.csv", parse_dates=["timestamp"])
X = lk.drop(columns=["timestamp", "leak_label"])
y = lk["leak_label"]

# 滑动窗口（示例：SWaT，窗口 60 秒）
X = normal.drop(columns=["timestamp", "label", "period"]).to_numpy()
windows = [X[i:i+60] for i in range(0, len(X)-60, 60)]
```

## 5. 使用建议

- **预测**：优先 BattLeDIM（5 min，119 维）、BATADAL（1 h，43 维）、GECCO（1 min，6 维）；高维秒级用 SWaT/WADI。
- **异常检测**：SWaT/WADI/BATADAL/GECCO 直接用 `label`；BattLeDIM 用 `leak_events` 的起始点；LeakDB 用 `leak_label`；Leakage-Repairs 需自行用工单匹配。
- **不要**把 BattLeDIM 的 `any_leak` 直接做逐点二分类；2019 年全为 1。
- **不要**把 2026 的 `fault_d7` 直接做逐点标签；先过滤 `label_reliable==0` 或重建标签。
- **不要**把 WDSEventDB 的 `relative_time_s` 当作真实绝对时间；它只是文件内相对秒数。
- **LeakDB** 应按场景划分 train/test，不能把同一场景的时间点随机打乱。
