import os, json, re
import pandas as pd

OUT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"
STATS = json.load(open(r"C:\Users\ASUS\AppData\Local\Temp\dataset_analysis\readme_stats.json", encoding="utf-8"))

def pct(x):
    if x is None:
        return "-"
    return f"{x*100:.2f}%"

def table(headers, rows):
    s = "| " + " | ".join(headers) + " |\n"
    s += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s

def hdr(folder):
    return pd.read_csv(os.path.join(OUT, folder), nrows=0).columns.tolist()

# ---------------------------------------------------------------- 01
def readme_2026():
    cols = hdr("01_2026_multisource/2026_multisource_hourly.csv")
    miss = STATS["2026"]["missing"]
    desc = {
        "timestamp": "小时时间戳（YYYY-MM-DD HH:00）",
        "fault_d7": "泄漏邻近标签：1=该时刻在真实泄漏事件前后 ±7 天内，0=不在",
        "4700008966_kW": "设备 4700008966 有功功率异常分数（0–100）",
        "4700008966_power_hour": "设备 4700008966 小时电量异常分数（0–100）",
        "4408411600_kW": "设备 4408411600 有功功率异常分数（0–100）",
        "4408411600_power_hour": "设备 4408411600 小时电量异常分数（0–100）",
        "4608927500_kvar_hour": "设备 4608927500 小时无功电量异常分数（0–100）",
        "238045_ws_temp": "水源 238045 温度异常分数（0–100）",
        "279805_ws_vigor": "水源 279805 活力/运行强度异常分数（0–100）",
        "279804_ws_level": "水源 279804 水位异常分数（0–100）",
        "279625_ws_temp": "水源 279625 温度异常分数（0–100）",
        "279623_ws_vigor": "水源 279623 活力/运行强度异常分数（0–100）",
        "319235_ws_temp": "水源 319235 温度异常分数（0–100）",
        "279831_ws_temp": "水源 279831 温度异常分数（0–100）",
        "326417_ws_temp": "水源 326417 温度异常分数（0–100）",
        "326416_ws_vigor": "水源 326416 活力/运行强度异常分数（0–100）",
        "temp_site1_anomaly_score": "站点 1 环境温度异常分数（0–100）",
        "sfc_temp_site1_anomaly_score": "站点 1 地表温度异常分数（0–100）",
        "gw_lvl_site2_anomaly_score": "站点 2 地下水位异常分数（0–100）",
        "gw_temp_site2_anomaly_score": "站点 2 地下水温异常分数（0–100）",
        "label_reliable": "标签可靠性：1=可用；0=原始文件各月 1 日块，排序后标签可能与时间不对应",
    }
    rows = [(c, desc.get(c, "见列名"), pct(miss.get(c, 0))) for c in cols]
    return f"""# 2026 多源城市供水管网数据集

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 文件 | `2026_multisource_hourly.csv` |
| 行 × 列 | 8,737 × 21 |
| 时间跨度 | 2021-12-31 23:00 ~ 2022-12-30 23:00 |
| 时间粒度 | 1 小时 |
| 变量数 | 18 个异常分数 + `fault_d7` + `label_reliable` |
| 来源 | 匿名化城市供水管网的 SCADA、能耗和环境数据；异常分数由 Elastic ML / Isolation Forest 预计算 |
| 缺失率 | 整体约 9.16%；14 列有缺失，最高 48.84% |

## 2. 数据质量

- 原始文件行序被打乱：前 289 行是各月 1 日（及 2021-12-31 23:00）的 24 小时数据，之后才是其余日期；已按时间排序。
- **排序后，被移到文件顶部的月首日数据标签可能与时间不对应**，例如 2022-02-01 的 `fault_d7` 全为 0，而 01-31、02-02 全为 1。`label_reliable=0` 标记这些行。
- `fault_d7` 是“泄漏事件前后 ±7 天”的粗粒度区间标签，正类占 **73.63%**（6,432/8,737），不是逐点异常标签。
- 18 列都是已经算好的异常分数，不是原始传感器读数；分数已经包含模型对异常的判断，拿来做异常检测存在“用分数检测分数”的循环性。
- 缺失值表示传感器停机或数据不可用；未插值。

## 3. 标签说明

| 标签 | 含义 | 分布 |
|---|---|---|
| `fault_d7` | 1=该时刻距真实泄漏事件在 ±7 天内；0=不在 | 1: 6,432；0: 2,305 |
| `label_reliable` | 1=标签与时间对应可信；0=原始月首日块，标签可能错位 | 0: 289 行；1: 8,448 行 |

## 4. 列说明

{table(["列名", "含义", "缺失率"], rows)}
## 5. 时序预测

**⚠️ 有条件可用。**

- 可以预测 18 条异常分数序列（小时级、约 1 年），能构造日/周周期。
- 但预测对象是“异常分数”而非物理量，结果缺乏物理解释；分数大量为 0、尖峰重尾。
- 不建议作为主预测基准，可作为真实场景的补充案例。

## 6. 异常检测

**❌ 不建议作为主基准。**

- 标签粒度太粗（±7 天）、正类 73.63%、存在月首日标签错位，无法可靠做逐点异常检测。
- 若一定要用，只能做事件级早期预警，并需要回到原始 PTIS 泄漏记录重新生成标签。
- 使用时应先过滤 `label_reliable == 1`，或把 `fault_d7` 当作事件级弱标签。

## 7. 使用注意

- 时间列已按 `DD-MM-YYYY HH:00` 解析并升序排序。
- 不要直接把 `fault_d7` 当作 point-wise 标签计算 F1/AUC。
- 缺失值未插值；建模时建议按列做缺失指示或前向填充并加 mask。
"""

# ---------------------------------------------------------------- 02
def readme_aqua():
    cols = hdr("02_AquaSense/aquasense_real_labeled.csv")
    miss = STATS["aqua"]["missing"]
    desc = {
        "timestamp": "NTP 同步采样时间（UTC）",
        "temperature": "BME680 环境温度（°C）",
        "humidity": "BME680 相对湿度（%RH）",
        "gas_resistance": "BME680 气体电阻（Ω）；越低表示 VOC 越高",
        "tds_value": "TDS 探头电压缩放值（未校准，不是 ppm）",
        "quality_final": "规则化水质等级（Poor / Good / Very Good）",
        "wq_score": "水质规则分数",
        "wq_class": "水质类别（Very Good / Good）",
        "attack_id": "攻击事件 ID；0=无攻击",
        "attack_type": "攻击类型：DoS / FDI / FDI-stealth / drift / replay / none",
        "attack_channel": "被攻击的传感器/通道",
        "fault_type": "故障类型：white_noise / spike / extreme / none",
        "fault_channel": "故障注入通道",
        "attack_label": "二值攻击标签：1=attack_id>0",
        "quality_legacy": "Real_data.csv 的 4 类传统质量标签（Good / Very good / Poor / mauvais）",
    }
    rows = [(c, desc.get(c, "见列名"), pct(miss.get(c, 0))) for c in cols]
    return f"""# AquaSense-ICS 数据集

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 文件 | `aquasense_real_labeled.csv` |
| 行 × 列 | 23,688 × 15 |
| 时间跨度 | 2025-05-19 12:39:08 ~ 2025-05-20 14:01:00（名义 25.4 h） |
| 时间粒度 | 约 1 秒（中位 1 s），但存在 34 个 >5 s 的缺口 |
| 有效时长 | 约 **7.9 小时**（扣除缺口后） |
| 变量数 | 4 个真实传感器 + 质量/攻击/故障标签 |
| 来源 | 水质 IoT / ICS 试验台；仅保留 `fused_data.csv` 中 `origin=real` 的行 |
| 缺失率 | 整体约 1.63%；`wq_score` / `wq_class` 缺失 12.21% |

## 2. 数据质量

- 真实数据名义跨度 25.4 h，但 34 个 >5 s 的缺口，最大缺口 10.1 h，**有效非缺口时长只有约 7.9 h**。
- 已排除 `origin=timegan` 的 23,688 行 TimeGAN 合成数据和 EPANET 仿真列。
- 攻击标签共 14 段、2,368 个正样本（约 10.0%）：
  - DoS 515；FDI 102；FDI-stealth 265；drift 442；replay 1,044。
- `splits.json` 明确说明 **test 切分只有 replay 攻击**，其他攻击类型不在测试集。
- 缺失值未插值；缺口保留为空。

## 3. 标签说明

| 标签 | 含义 | 分布 |
|---|---|---|
| `attack_label` | 1=攻击；0=正常 | 1: 2,368；0: 21,320 |
| `attack_id` | 攻击事件 ID，0=无攻击 | 14 个事件 |
| `attack_type` | 攻击类型 | replay 1,044；DoS 515；drift 442；FDI-stealth 265；FDI 102；none 21,320 |
| `fault_type` | 故障类型 | white_noise 2,368；extreme 47；spike 118；none 21,155 |

## 4. 列说明

{table(["列名", "含义", "缺失率"], rows)}
### 攻击段表 `aquasense_attack_segments.csv`

{table(["列名", "含义"], [("start", "攻击段开始时间"), ("end", "攻击段结束时间"), ("duration_s", "持续秒数"), ("attack_id", "攻击事件 ID"), ("attack_type", "攻击类型"), ("attack_channel", "被攻击通道")])}
## 5. 时序预测

**❌ 不建议作为主基准。**

- 真实数据只有 4 个变量、有效约 8 小时、缺口大；不足以支撑有意义的日/周周期或长时预测。
- 可以做秒级短时预测 demo，但不适合论文主实验。

## 6. 异常检测

**⚠️ 有条件可用。**

- 优点：逐点攻击标签、攻击类型/通道/故障类型齐全，窗口 30（约 30 秒），适合小规模 ICS 攻击检测和消融实验。
- 缺点：样本少、只有 4 个真实通道、测试集攻击类型单一（只有 replay）、缺口多、合成数据不能参与真实评估。
- 适合作为“小型案例验证”，不适合作为唯一或主异常检测基准。

## 7. 使用注意

- 只使用 `origin=real` 的数据；不要把 TimeGAN 合成行混入真实评估。
- 建议按 `splits.json` 的时间顺序切分，不要随机打乱。
- 14 个攻击段的起止/类型/通道见 `aquasense_attack_segments.csv`。
"""

# ---------------------------------------------------------------- 03
def readme_batadal():
    cols = hdr("03_BATADAL/BATADAL_all.csv")
    miss = STATS["batadal"]["missing"]
    def d(c):
        if c == "timestamp":
            return "逐小时时间戳"
        if c.startswith("L_T"):
            return f"水箱 {c[2:]} 的液位"
        if c.startswith("F_PU"):
            return f"泵 {c[4:]} 的流量"
        if c.startswith("S_PU"):
            return f"泵 {c[4:]} 的状态（0/1）"
        if c == "F_V2":
            return "阀门 V2 的流量"
        if c == "S_V2":
            return "阀门 V2 的状态（0/1）"
        if c.startswith("P_J"):
            return f"节点 J{c[3:]} 的压力"
        if c == "label":
            return "0=正常；1=攻击；2017 子集为空（无标签）"
        if c == "subset":
            return "数据子集：train_normal_2014_2015 / test_labeled_2016 / test_unlabeled_2017"
        return "见列名"
    rows = [(c, d(c), pct(miss.get(c, 0))) for c in cols]
    return f"""# BATADAL 数据集

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 文件 | `BATADAL_all.csv` |
| 行 × 列 | 15,027 × 46 |
| 时间跨度 | 2014-01-06 00:00 ~ 2017-04-01 00:00（三个不连续子集） |
| 时间粒度 | 1 小时 |
| 变量数 | 43 个 SCADA 特征 + `label` + `subset` |
| 子集 | 2014-01-06~2015-01-06 正常（8,761 行）；2016-07-04~2016-12-25 有标签（4,177 行）；2017-01-04~2017-04-01 无标签（2,089 行） |
| 缺失率 | 特征无缺失；`label` 缺失 13.90%（2017 子集无标签） |

## 2. 数据质量

- 43 个特征的三个子集均无缺失、无重复时间戳、严格 1 小时。
- dataset03 为正常训练年（`ATT_FLAG` 全 0）；dataset04 为 2016 年测试集，219 个攻击点（5.24%），共 5 段攻击；2017 CSV 为官方测试集，**无标签**。
- 正常期（2014–2015）与测试期（2016）存在明显跨年分布漂移。
- 攻击类型未细分。

## 3. 标签说明

| 标签 | 含义 | 分布 |
|---|---|---|
| `label` | 0=正常；1=攻击；2017 子集为空 | 0: 12,719；1: 219；空: 2,089 |
| `subset` | 数据来源/用途 | 正常 8,761；2016 有标签 4,177；2017 无标签 2,089 |

## 4. 列说明

{table(["列名", "含义", "缺失率"], rows)}
## 5. 时序预测

**✅ 适合。**

- 1 年逐小时、43 变量、完整无缺失，日/周周期清晰，是经典多变量预测基准。
- 可用 2014–2015 训练、2016 测试，天然形成跨年分布漂移评估。
- 预测目标可选液位、压力、泵流量等。

## 6. 异常检测

**✅ 适合。**

- 经典半监督/有监督基准：dataset03 作正常训练集，dataset04 作带标签测试集。
- 219 个攻击点、5 段，正类 5.24%，适合 point-wise 与 event-wise 指标。
- 2017 子集无标签，只能用于无监督/迁移评估。
- 攻击段数偏少（5 段），统计显著性有限，建议与 SWaT/WADI 配合。

## 7. 使用注意

- `label` 中的 `-999` 已映射为 0；2017 子集 `label` 为空。
- 不要随机打乱时间点；按 `subset` 做时间顺序切分。
- 2017 子集不要用于有监督指标。
"""

# ---------------------------------------------------------------- 04
def readme_battledim():
    cols18 = hdr("04_BattLeDIM/BattLeDIM_2018_5min.csv")
    cols19 = hdr("04_BattLeDIM/BattLeDIM_2019_5min.csv")
    press = [c for c in cols18 if c.startswith("pressure_")]
    dem = [c for c in cols18 if c.startswith("demand_")]
    leak18 = [c for c in cols18 if c.startswith("leak_") and c not in ("leak_onset_count", "leak_onset")]
    leak19 = [c for c in cols19 if c.startswith("leak_") and c not in ("leak_onset_count", "leak_onset")]
    def names(xs):
        return "、".join(xs)
    flow = [c for c in cols18 if c.startswith("flow_")]
    return f"""# BattLeDIM (L-Town) 数据集

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 文件 | `BattLeDIM_2018_5min.csv`（105,120 × 138）；`BattLeDIM_2019_5min.csv`（105,120 × 147） |
| 时间跨度 | 2018-01-01 00:00 ~ 2018-12-31 23:55；2019-01-01 00:00 ~ 2019-12-31 23:55 |
| 时间粒度 | 5 分钟 |
| 变量数 | 119 条 SCADA（33 压力 + 3 流量 + 1 液位 + 82 需求）+ 各管段泄漏流量 + 汇总/起始标记 |
| 来源 | L-Town 供水管网 EPANET 仿真 SCADA，BattLeDIM 2020 竞赛数据 |
| 缺失率 | 0%（两年度均无缺失、无重复时间戳） |

## 2. 数据质量

- 2018/2019 各 105,120 行，严格 5 min，完整无缺失。
- 泄漏标签是各管段的**连续泄漏流量**（m³/h），2018 年 14 条、2019 年 23 条。
- **2018 年 97.8% 的时间点至少有一条管段泄漏**（p257 从 1 月 9 日持续到年底）；**2019 年 100%** 的时间点都有泄漏（p257、p427、p654、p810 从 2018 年延续）。
- 2018 年基本无泄漏的时段只有 **1 月 1~8 日**；2019 年没有无泄漏时段。
- 2019 年 23 条泄漏中 4 条为 2018 年延续，19 条为新发；2018 年 14 条中 10 条在年内修复，4 条延续。

## 3. 标签说明

| 列 | 含义 |
|---|---|
| `leak_p*` | 管段 p* 的泄漏流量（m³/h），0=无泄漏 |
| `total_leak_m3h` | 所有管段泄漏流量之和（m³/h） |
| `any_leak` | 1=该时刻至少一条管段泄漏；0=全部无泄漏 |
| `leak_onset_count` | 该 5 min 时刻新发生泄漏的管段数 |
| `leak_onset` | 1=该时刻有新泄漏起始；0=无 |
| `BattLeDIM_*_leak_events.csv` | 每条管段泄漏的起止、持续时长、最大/平均泄漏、峰值时间、`onset_observed` |

- 2018 年 14 个泄漏起始；2019 年 19 个新起始（4 条延续泄漏的起始未观测到）。
- **不要用 `any_leak` 做逐点二分类**：2019 年全为 1。做异常检测应使用 `leak_onset` 或事件表。

## 4. 列说明

### 共用 SCADA 列

| 列名 | 含义 |
|---|---|
| `timestamp` | 5 min 时间戳 |
| `flow_p227`、`flow_p235`、`flow_PUMP_1` | 3 个流量计（m³/h）：管段 p227、p235、泵 PUMP_1 |
| `pressure_*` | 33 个压力传感器（m），节点：{names(press)} |
| `level_T1` | 水箱 T1 液位（m） |
| `demand_*` | 82 个 AMR 需求（L/h），节点：{names(dem)} |

### 泄漏与汇总列

| 列名 | 含义 |
|---|---|
| `leak_p*`（2018） | 14 条管段泄漏流量（m³/h）：{names(leak18)} |
| `leak_p*`（2019） | 23 条管段泄漏流量（m³/h）：{names(leak19)} |
| `total_leak_m3h` | 所有管段泄漏流量之和 |
| `any_leak` | 是否至少一条管段泄漏（1/0） |
| `leak_onset_count` | 新泄漏管段数 |
| `leak_onset` | 是否有新泄漏起始（1/0） |

> 所有 SCADA 列缺失率均为 0%。

### 泄漏事件表 `BattLeDIM_*_leak_events.csv`

{table(["列名", "含义"], [("year", "年份（2018/2019）"), ("link", "泄漏管段 ID"), ("start", "泄漏开始时间"), ("end", "泄漏结束时间"), ("duration_h", "持续小时数"), ("max_leak_m3h", "最大泄漏流量（m³/h）"), ("mean_leak_m3h", "平均泄漏流量（m³/h）"), ("peak_time", "峰值时间"), ("onset_observed", "1=起始被观测到；0=年初已存在、起始未观测到")])}
## 5. 时序预测

**✅ 非常适合。**

- 5 min、2 年、119 条序列、完整无缺失，日/周周期明显。
- 可预测压力、流量、液位、需求；适合长上下文、多变量、跨年预测。
- 2018 训练 / 2019 测试天然形成分布漂移场景。

## 6. 异常检测

**✅ 适合，但必须用事件级标签。**

- 不能直接用 `any_leak` 做逐点二分类（2019 全为 1，2018 97.8% 为 1）。
- 推荐使用 `leak_onset` 做泄漏起始检测，或用 `leak_p*` 做泄漏流量回归/定位。
- 正常基线：2018 年 1 月 1~8 日（以及各事件前的局部窗口）；2019 年不能提供无泄漏基线。
- 事件表提供每条管段的起止、峰值、持续时长和 `onset_observed`。

## 7. 使用注意

- 原始 CSV 使用 `;` 分隔、`,` 为小数点；本目录已转换为标准 CSV。
- 2019 年 4 条延续泄漏的 `onset_observed=0`，不要把它们当作新起始。
- 评估建议用事件级指标（time-to-detection、定位命中率），而不是只看 point-wise F1。
"""

# ---------------------------------------------------------------- 05
def readme_gecco():
    cols = hdr("05_GECCO/GECCO_all_1min.csv")
    miss = STATS["gecco"]["missing"]
    desc = {
        "timestamp": "逐分钟时间戳",
        "Tp": "水温（Temperature）",
        "pH": "pH 值",
        "Cond": "电导率（Conductivity）",
        "Turb": "浊度（Turbidity）",
        "SAC": "光谱吸收系数（Spectral Absorption Coefficient）",
        "PFM": "GECCO 原始流量/压力相关监测字段（PFM）",
        "event": "0=正常；1=污染事件",
        "split": "官方时间切分：train / valid / test",
    }
    rows = [(c, desc.get(c, "见列名"), pct(miss.get(c, 0))) for c in cols]
    return f"""# GECCO 2019 水质数据集

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 文件 | `GECCO_all_1min.csv` |
| 行 × 列 | 218,880 × 9 |
| 时间跨度 | 2017-07-01 00:00 ~ 2017-11-29 23:59（152 天） |
| 时间粒度 | 1 分钟 |
| 变量数 | 6 个水质变量 + `event` + `split` |
| 官方切分 | train 132,480 行（7/1~9/30）；valid 54,750 行（10/1~11/8 00:29）；test 31,650 行（11/8 00:30~11/29） |
| 缺失率 | 各变量约 1.13%~1.15%；valid 段约 4.1%（原始 0 值按缺失处理） |

## 2. 数据质量

- 6 个变量存在 0 值，已按缺失处理；其余缺失为原始缺测。
- 事件标签为逐点 0/1，事件持续时间短（4~41 min），整体正类比例极低。
- **独立连续事件区间 37 个，共 628 个事件分钟**（train 13 段/212 分钟；valid 12 段/225 分钟；test 12 段/191 分钟）。
- 训练/验证/测试按时间切分，测试期（11 月）的 `Tp` 分布与训练期（7~9 月）明显不同，存在季节/分布漂移。

## 3. 标签说明

| 标签 | 含义 | 分布 |
|---|---|---|
| `event` | 1=该分钟处于污染事件内；0=正常 | 1: 628；0: 218,252 |
| `split` | 官方时间切分 | train 132,480；valid 54,750；test 31,650 |

- `GECCO_events.csv` 给出 37 个连续事件区间的起止、持续分钟数和所属 split。

## 4. 列说明

{table(["列名", "含义", "缺失率"], rows)}
### 事件表 `GECCO_events.csv`

{table(["列名", "含义"], [("start", "事件开始时间"), ("end", "事件结束时间"), ("duration_min", "持续分钟数"), ("split", "所属官方切分：train/valid/test")])}
## 5. 时序预测

**✅ 适合。**

- 1 min、5 个月、6 个变量，适合分钟~小时级水质预测。
- 数据量适中、变量少，适合快速对比 ARIMA/LSTM/Transformer/时序大模型。
- 需注意跨月分布漂移；这反而适合做漂移鲁棒性实验。

## 6. 异常检测

**✅ 适合。**

- 37 个事件区间、628 个事件分钟、逐点标签、官方切分明确，是水质污染事件检测的经典基准。
- 极端不平衡，适合 F1、AUPRC、事件级检测率、误报率；不适合只看 accuracy。
- 建议同时报告 point-wise 和 event-wise 指标。

## 7. 使用注意

- 628 是**事件分钟数**，不是独立事件数；独立连续事件区间是 37 个。
- 不要随机打乱时间点；按 `split` 使用官方时间切分。
- 0 值已在处理中转为缺失，不要再当作真实 0。
"""

# ---------------------------------------------------------------- 06
def readme_leakage():
    cols = hdr("06_LeakageRepairs/LeakageRepairs_flows_5min.csv")
    miss = STATS["leakage"]["missing"]
    desc = {c: f"DMA {c.replace('_flow','')} 的 5 min 平均入口流量" for c in cols if c.endswith("_flow")}
    desc["timestamp"] = "5 min 时间戳"
    desc["n_areas_reporting"] = "该 5 min 时刻有数据的 DMA 数量（0~6）"
    rows = [(c, desc.get(c, "见列名"), pct(miss.get(c, 0))) for c in cols]
    rep = hdr("06_LeakageRepairs/LeakageRepairs_reports.csv")
    repdesc = {
        "topic": "工单主题（希腊语原始文本）",
        "reason": "维修原因（原始文本）",
        "action": "维修动作（原始文本）",
        "area": "DMA 区域（Area_1~Area_6）",
        "timestamp": "维修工单时间戳",
        "severity": "严重度：High / Med / Low / unknown（原缺失已填 unknown）",
        "repeated": "重复工单次数",
    }
    reprows = [(c, repdesc.get(c, "见列名")) for c in rep]
    return f"""# Leakage-Repairs 数据集

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 文件 | `LeakageRepairs_flows_5min.csv`（331,776 × 8）；`LeakageRepairs_reports.csv`（2,577 × 7） |
| 时间跨度 | 2022-01-01 00:00 ~ 2025-02-25 23:55（Area_6 到 2024-12-22） |
| 时间粒度 | 5 分钟（由原始约 5 min 不规则采样重采样得到） |
| 变量数 | 6 个 DMA 流量 + `n_areas_reporting`；另有 2,577 条维修记录 |
| 来源 | 塞浦路斯真实供水管网 DMA 流量 + 泄漏维修工单 |
| 缺失率 | 整体约 9.95%；Area_2 33.77%、Area_1 17.14%、Area_6 12.51%、Area_4 6.77%、Area_5 4.86%、Area_3 4.53% |

## 2. 数据质量

- 原始流量数据缺口多、异常值多：Area_1 有 16,395 个 >10 min 缺口，Area_2 有 21,903 个；最大缺口 Area_4 达 1,872 h。
- 已做清洗：负值/零值置空；每区按 `max(q99.9×3, median×10)` 截断极端值；按 5 min 重采样；**缺口保留为空，不插值**。
- 标签不是逐点/逐区间标签，而是维修工单时间戳；泄漏起止时间未知，需把“流量显著下降”与附近工单匹配。
- `severity` 原始缺失 28.4%，已填 `unknown`。
- `processed_flow_data/` 原始为空；本目录的流量表是重新清洗后的结果。

## 3. 标签说明

| 列 | 含义 |
|---|---|
| `LeakageRepairs_reports.csv: timestamp` | 维修工单时间（不是泄漏开始时间） |
| `severity` | 严重度：High 199 / Med 927 / Low 719 / unknown 732 |
| `repeated` | 同一泄漏的重复工单次数 |
| `Area_*_flow` | DMA 入口流量；流量下降可能对应泄漏或维修 |

## 4. 列说明

### 流量表

{table(["列名", "含义", "缺失率"], rows)}
### 维修记录表

{table(["列名", "含义"], reprows)}
### 区域摘要表 `LeakageRepairs_area_summary.csv`

| 列名 | 含义 |
|---|---|
| `area` | DMA 区域 |
| `raw_rows` | 原始行数 |
| `median_flow` | 清洗后流量中位数 |
| `q999` | 流量 99.9% 分位数 |
| `cap` | 极端值截断阈值 |
| `capped_rows` | 被截断的行数 |
| `reporting_rows_5min` | 5 min 网格中有数据的行数 |
| `missing_5min` | 5 min 网格中缺失的行数 |
| `start` / `end` | 原始数据起止时间 |

## 5. 时序预测

**⚠️ 有条件可用。**

- 单变量 5 min、3~4 年，长度足够，能学日/周周期。
- 但缺口、零值、极端异常值必须先清洗；本目录已完成基本清洗。
- 只有流量一个变量/区，没有压力/需求等外生变量，预测上限有限。

## 6. 异常检测

**⚠️ 有条件可用。**

- 真实漏损场景 + 真实维修记录，价值在真实性和弱标签，适合做“真实世界验证”。
- 标签噪声大：维修记录时间不等于泄漏开始时间；同一泄漏可能有多条重复工单；需要标签工程。
- 更适合做“基于流量下降的漏损检测”而非标准 point-wise AD 基准；建议用事件级召回/误报率、维修确认率评估。

## 7. 使用注意

- 不要把维修工单时间直接当作泄漏开始时间。
- 缺口不要强行插值；建模时可用 mask 或只使用连续段。
- 原始数据是倒序的，本目录已按时间升序排序。
"""

# ---------------------------------------------------------------- 07
def readme_leakdb():
    sel = pd.read_csv(os.path.join(OUT, "07_LeakDB/LeakDB_selected_scenarios.csv"))
    selrows = [(r.network, r.archive, r.scenario, r.has_leak, f"{r.leak_fraction*100:.2f}%", f"{r.rows}×{r.columns}", r.output_file) for _, r in sel.iterrows()]
    hanoi_cols = hdr("07_LeakDB/Hanoi500_Scenario-010.csv")
    net1_cols = hdr("07_LeakDB/Net1_1000_Scenario-001.csv")
    return f"""# LeakDB 数据集

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 文件 | 9 个场景 CSV：Hanoi500 ×3、Hanoi1000 ×3、Net1 ×3 |
| 行 × 列 | 17,520 × 37~102（每个场景） |
| 时间跨度 | 2017-01-01 00:00 ~ 2017-12-31 23:30（每场景 1 年） |
| 时间粒度 | 30 分钟 |
| 变量数 | Hanoi：32 需求 + 34 流量 + 32 压力 = 98 条序列；Net1：11 + 13 + 11 = 35 条序列 |
| 来源 | EPANET 仿真供水管网（Hanoi_CMH / Net1_CMH），含管长/管径/粗糙度不确定性 |
| 缺失率 | 0%（仿真数据完整） |

## 2. 导出范围与数据质量

- 完整数据包含 **2,500 个场景**（Hanoi 500 + Hanoi 1000 + Net1 1000）。本目录只导出 **9 个代表场景**（每个网络 1 个无泄漏 + 2 个泄漏，覆盖短/长泄漏）。
- 全部 2,500 个场景的 `has_leak` 索引见 `LeakDB_all_scenarios_index.csv`；9 个导出场景摘要见 `LeakDB_selected_scenarios.csv`。
- 每个场景的 `leak_label` 来自 `Labels.csv`，是仿真真值；泄漏流量列 `leak_*` 在早期小泄漏时可能为 0（例如 Net1 的 incipient 泄漏），此时以 `leak_label` 为准。
- 1000 场景版本原始数据只有 `Index` 无时间戳，已按 2017-01-01 起、30 min 步长补 `timestamp`；500 场景版本使用原始 `Timestamp`。
- 未导出全部场景的原因：全部解压约 64 GB，转宽表 CSV 会更大；需要更多场景可修改 `_scripts/p07_leakdb.py` 后重跑。

## 3. 标签说明

| 列 | 含义 |
|---|---|
| `leak_label` | 逐点泄漏标签：1=该 30 min 时刻处于泄漏区间；0=正常 |
| `has_leak`（索引表） | 场景级：1=该场景包含泄漏；0=无泄漏 |
| `leak_fraction`（摘要表） | 场景内 `leak_label=1` 的比例 |
| `leak_Leak_N` | 第 N 个泄漏源的泄漏流量 |

## 4. 列说明

### 命名规则

| 列名前缀 | 含义 |
|---|---|
| `demand_Node_N` | 节点 N 的需求量 |
| `flow_Link_N` | 管段 N 的流量 |
| `pressure_Node_N` | 节点 N 的压力 |
| `leak_Leak_N` | 第 N 个泄漏源的泄漏流量 |
| `leak_label` | 逐点 0/1 泄漏标签 |

- Hanoi 场景列：`timestamp`、32 个 `demand_Node_*`、34 个 `flow_Link_*`、32 个 `pressure_Node_*`、若干 `leak_Leak_*`、`leak_label`。
- Net1 场景列：`timestamp`、11 个 `demand_Node_*`、13 个 `flow_Link_*`、11 个 `pressure_Node_*`、若干 `leak_Leak_*`、`leak_label`。
- 以 `Hanoi500_Scenario-010.csv` 为例，列名完整列表：{', '.join(hanoi_cols)}
- 以 `Net1_1000_Scenario-001.csv` 为例，列名完整列表：{', '.join(net1_cols)}

### 导出场景

{table(["网络", "压缩包", "场景", "has_leak", "泄漏比例", "行×列", "文件"], selrows)}
## 5. 时序预测

**✅ 非常适合。**

- 每个场景 1 年、30 min、35~98 条序列，数据完整、可控。
- 可按场景做多变量预测，也可跨场景训练；适合研究多场景/多网络泛化。
- 仿真数据没有真实噪声，预测指标会偏乐观；建议与 BattLeDIM、Leakage-Repairs 配合。

## 6. 异常检测

**✅ 非常适合。**

- 逐点泄漏标签 + 场景级泄漏/无泄漏标签，支持有监督、半监督、无监督（用无泄漏场景作正常基线）。
- 场景数量大，适合统计显著性检验和多种基线对比。
- 应按**场景**划分 train/test，不能把同一场景的时间点随机打乱，否则会泄漏场景信息。
- 仿真标签是“真值”，适合方法对比，但不代表真实管网的可检测性。

## 7. 使用注意

- 9 个导出场景只是代表子集；需要全量 2,500 场景时，用 `_scripts/p07_leakdb.py` 修改选择。
- 1000 场景版本的 `timestamp` 是按 30 min 步长补的，原始只有 `Index`。
- 早期 incipient 泄漏的 `leak_*` 可能为 0，但 `leak_label` 仍为 1；以 `leak_label` 为准。
"""

# ---------------------------------------------------------------- 08
def readme_swat():
    cols = hdr("08_SWaT/SWaT_all_1s.csv")
    sensor_desc = {
        "timestamp": "1 秒时间戳",
        "FIT101": "阶段1 原水进水流量",
        "LIT101": "阶段1 原水罐液位",
        "MV101": "阶段1 原水进水电动阀",
        "P101": "阶段1 原水泵 1",
        "P102": "阶段1 原水泵 2",
        "AIT201": "阶段2 化学加药/水质分析仪 1",
        "AIT202": "阶段2 化学加药/水质分析仪 2",
        "AIT203": "阶段2 化学加药/水质分析仪 3",
        "FIT201": "阶段2 化学加药流量",
        "MV201": "阶段2 化学加药电动阀",
        "P201": "阶段2 加药泵 1",
        "P203": "阶段2 加药泵 3",
        "P204": "阶段2 加药泵 4",
        "P205": "阶段2 加药泵 5",
        "P206": "阶段2 加药泵 6",
        "DPIT301": "阶段3 超滤/反渗透差压",
        "FIT301": "阶段3 超滤产水流量",
        "LIT301": "阶段3 超滤产水罐液位",
        "MV301": "阶段3 超滤/反渗透电动阀 1",
        "MV302": "阶段3 超滤/反渗透电动阀 2",
        "MV303": "阶段3 超滤/反渗透电动阀 3",
        "MV304": "阶段3 超滤/反渗透电动阀 4",
        "P301": "阶段3 超滤泵 1",
        "P302": "阶段3 超滤泵 2",
        "AIT401": "阶段4 反渗透水质分析仪 1",
        "AIT402": "阶段4 反渗透水质分析仪 2",
        "FIT401": "阶段4 反渗透产水流量",
        "LIT401": "阶段4 反渗透产水罐液位",
        "P402": "阶段4 反渗透泵 2",
        "P403": "阶段4 反渗透泵 3",
        "UV401": "阶段4 UV 消毒",
        "AIT501": "阶段5 水质分析仪 1",
        "AIT502": "阶段5 水质分析仪 2",
        "AIT503": "阶段5 水质分析仪 3",
        "AIT504": "阶段5 水质分析仪 4",
        "FIT501": "阶段5 流量 1",
        "FIT502": "阶段5 流量 2",
        "FIT503": "阶段5 流量 3",
        "FIT504": "阶段5 流量 4",
        "P501": "阶段5 泵 1",
        "PIT501": "阶段5 压力 1",
        "PIT502": "阶段5 压力 2",
        "PIT503": "阶段5 压力 3",
        "FIT601": "阶段6 供水流量",
        "P602": "阶段6 供水泵 2",
        "label": "0=正常；1=攻击",
        "period": "normal / attack",
    }
    rows = [(c, sensor_desc.get(c, "见命名规则"), "0.00%") for c in cols]
    dropped = open(os.path.join(OUT, "08_SWaT/SWaT_dropped_constant_columns.txt"), encoding="utf-8").read().strip().splitlines()
    return f"""# SWaT 数据集

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 文件 | `SWaT_all_1s.csv`（944,919 × 48）；`SWaT_attack_segments.csv`（35 × 3） |
| 时间跨度 | 2015-12-22 16:30:00 ~ 2016-01-02 14:59:59 |
| 时间粒度 | 1 秒 |
| 变量数 | 45 个传感器/执行器 + `label` + `period`（原始 51 个特征中 6 个恒定列已删除） |
| 子段 | 正常 495,000 行（2015-12-22 16:30 ~ 2015-12-28 09:59:59）；攻击文件 449,919 行（2015-12-28 10:00 ~ 2016-01-02 14:59:59） |
| 缺失率 | 0%（删除恒定列后无缺失） |

## 2. 数据质量

- 正常文件与攻击文件均无缺失、无重复时间戳、严格 1 秒。
- 攻击文件标签原始值：`Normal` 395,298；`Attack` 54,584；`A ttack` 37。**`A ttack` 是拼写错误，已按 Attack 处理**。
- 攻击点合计 54,621（整体 5.78%；攻击文件内 12.14%），共 **35 段攻击**。
- 6 个恒定列已删除：{', '.join(dropped)}。
- `SWaT_Dataset_Normal_v0.xlsx` 文件损坏（zip 中央目录缺失），未使用；只使用 Normal_v1。
- 正常期与攻击期是不同时间段，存在时间分布漂移。

## 3. 标签说明

| 标签 | 含义 | 分布 |
|---|---|---|
| `label` | 0=正常；1=攻击 | 0: 890,298；1: 54,621 |
| `period` | normal / attack | normal 495,000；attack 449,919 |
| `SWaT_attack_segments.csv` | 35 段攻击的起止和持续秒数 | 35 行 |

## 4. 列说明

{table(["列名", "含义", "缺失率"], rows)}
> 传感器物理含义按 SWaT 公开 P&ID/命名规则推断；不同文献对具体设备描述可能略有差异。`FIT`=流量、`LIT`=液位、`MV`=电动阀、`P`=泵、`AIT`=分析仪、`DPIT`=差压、`UV`=UV 消毒、`PIT`=压力。

### 攻击段表 `SWaT_attack_segments.csv`

{table(["列名", "含义"], [("start", "攻击段开始时间"), ("end", "攻击段结束时间"), ("duration_s", "持续秒数")])}
## 5. 时序预测

**✅ 适合。**

- 1 s、45 个变量、约 11 天总数据，适合秒级~分钟级多变量预测。
- 数据量足够训练深度模型；但只有约 6 天正常 + 5 天攻击，不适合年周期/季节预测。
- 可用 Normal 段训练预测模型，在 Attack 段评估残差异常检测。

## 6. 异常检测

**✅ 非常适合。**

- ICS 异常检测经典基准：正常/攻击分离，35 段攻击、54,621 个攻击点。
- 适合半监督（正常训练、攻击测试）和有监督评估；point-wise 与 event-wise 指标都可做。
- 注意正常期与攻击期是不同时间段，存在漂移；不要随机打乱时间点。

## 7. 使用注意

- 原始 xlsx 有两行表头，本目录已转为标准 CSV。
- 时间戳 2015 年零填充、2016 年不填充；本目录已统一解析。
- `A ttack` 拼写错误已修正；不要再用原始字符串直接匹配。
"""

# ---------------------------------------------------------------- 09
def readme_wadi():
    cols = hdr("09_WADI/WADI_all_1s.csv")
    tag_map = {"AIT":"分析仪", "FIT":"流量变送器", "LIT":"液位变送器", "LT":"液位变送器", "LS":"液位开关",
               "MV":"电动阀", "P":"泵", "DPIT":"差压变送器", "FIC":"流量控制器",
               "FQ":"流量累积", "MCV":"电动控制阀", "SV":"电磁阀", "PIT":"压力变送器",
               "PIC":"压力控制器"}
    suffix_map = {"PV":"过程值", "STATUS":"状态", "AL":"低报警", "AH":"高报警",
                  "CO":"控制输出", "SP":"设定值", "SPEED":"转速"}
    def desc(c):
        if c == "timestamp": return "1 秒时间戳"
        if c == "label": return "0=正常；1=攻击"
        if c == "period": return "normal / attack"
        if c == "LEAK_DIFF_PRESSURE": return "泄漏差压"
        if c == "TOTAL_CONS_REQUIRED_FLOW": return "总需求流量"
        m = re.match(r"^(1|2A|2B|2|3)_([A-Z]+)_(\d+)_([A-Z0-9\-]+)$", c)
        if m:
            stage, tag, num, suf = m.groups()
            return f"阶段{stage} {tag_map.get(tag, tag)} {tag}-{num} {suffix_map.get(suf, suf)}"
        return "见命名规则"
    rows = [(c, desc(c), "0.00%") for c in cols]
    dropped = open(os.path.join(OUT, "09_WADI/WADI_dropped_constant_columns.txt"), encoding="utf-8").read().strip().splitlines()
    return f"""# WADI 数据集

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 文件 | `WADI_all_1s.csv`（957,372 × 101）；`WADI_attack_segments.csv`（14 × 3） |
| 时间跨度 | 2017-09-25 00:00:00 ~ 2017-10-11 18:00:00 |
| 时间粒度 | 1 秒 |
| 变量数 | 98 个传感器/执行器 + `label` + `period`（原始 127 个特征中 29 个恒定列已删除） |
| 子段 | 正常 784,571 行（2017-09-25 ~ 2017-10-07，含缺失日期）；攻击 172,801 行（2017-10-09 ~ 2017-10-11） |
| 缺失率 | 0%（删除恒定列后无缺失；原始缺失主要集中在被删除的恒定列） |

## 2. 数据质量

- 原始 `Time` 是 `MM:SS.t` 且每小时清零；本目录已按“每个日期块从首个时间起逐秒累加”重建绝对时间。
- 攻击文件末尾有 2 行空行，已丢弃；攻击点 9,977 个（整体 1.04%；攻击文件内 5.77%），共 **14 段攻击**。
- 原始标签 1=正常、-1=攻击，已映射为 0/1。
- 29 个恒定列已删除：{', '.join(dropped)}。
- 正常段本身有缺失日期（9/30~10/1 等），未插值；`WADI.A1` 与 `WADI.A2` 文件重复，只使用 A1。
- 正常期与攻击期是不同时间段，存在分布漂移。

## 3. 标签说明

| 标签 | 含义 | 分布 |
|---|---|---|
| `label` | 0=正常；1=攻击 | 0: 947,395；1: 9,977 |
| `period` | normal / attack | normal 784,571；attack 172,801 |
| `WADI_attack_segments.csv` | 14 段攻击的起止和持续秒数 | 14 行 |

## 4. 列说明

> 命名规则：`阶段_仪表类型_编号_后缀`。阶段 1/2/2A/2B/3 对应 WADI 的不同处理阶段；后缀 `PV`=过程值、`STATUS`=状态、`AL`=低报警、`AH`=高报警、`CO`=控制输出、`SP`=设定值、`SPEED`=转速。具体化学/设备名称原始数据未文档化，以下按仪表类型说明。

{table(["列名", "含义", "缺失率"], rows)}
### 攻击段表 `WADI_attack_segments.csv`

{table(["列名", "含义"], [("start", "攻击段开始时间"), ("end", "攻击段结束时间"), ("duration_s", "持续秒数")])}
## 5. 时序预测

**✅ 适合。**

- 1 s、98 个变量，适合高维多变量短时预测。
- 正常约 9 天有效数据 + 攻击 2 天；数据量足够训练深度模型，但时间跨度短，不适合季节/长周期预测。
- 可用正常段训练、攻击段测试残差。

## 6. 异常检测

**✅ 非常适合。**

- 经典 ICS 异常检测基准：14 段攻击、9,977 个攻击点、98 个变量。
- 适合半监督/有监督、point-wise/event-wise 评估。
- 注意正常/攻击时间段不同，存在漂移；不要随机打乱时间点。

## 7. 使用注意

- 原始 `Time` 每小时清零，不能直接按 `HH:MM` 解析；本目录已重建。
- 攻击文件末尾 2 行空行已丢弃；不要再用原始行数。
- 正常段有缺失日期，未插值；建模时可用 mask。
"""

# ---------------------------------------------------------------- 10
def readme_wdsevent():
    cols = hdr("10_WDSEventDB/WDSEventDB_all_events.csv")
    desc = {
        "event_type": "事件类型：clean / cyberattack / leakage / sensor_failure",
        "source_file": "原始 Excel 文件名",
        "segment_id": "事件段 ID（event_type:文件名）",
        "row_index": "文件内行号（0 起）",
        "relative_time_s": "文件内相对秒数（按 1 Hz 假设，第 n 行 = n 秒）",
        "Pressure 1 Out": "压力 1（出口）",
        "Pressure 2 Out": "压力 2（出口）",
        "Pressure 3 In": "压力 3（入口）",
        "Pressure 4 In": "压力 4（入口）",
        "Water Flow 1": "水流量 1",
        "Water Flow 2": "水流量 2",
        "Water Flow 3": "水流量 3",
        "Water Flow 4": "水流量 4",
        "VFD 1": "变频器 1",
        "VFD 2": "变频器 2",
        "VFD 3": "变频器 3",
        "VFD 4-1": "变频器 4-1",
        "VFD 4-2": "变频器 4-2",
        "Analog Valve 1": "模拟阀 1",
        "Analog Valve 2": "模拟阀 2",
        "label": "0=正常；1=事件（攻击/泄漏/传感器故障）",
    }
    rows = [(c, desc.get(c, "见列名"), "0.00%") for c in cols]
    summ = pd.read_csv(os.path.join(OUT, "10_WDSEventDB/WDSEventDB_file_summary.csv"))
    srows = [(r.event_type, r.source_file, r.rows, r.positive, f"{r.positive_pct:.2f}%") for _, r in summ.iterrows()]
    return f"""# WDSEventDB 数据集

## 1. 基本信息

| 项目 | 内容 |
|---|---|
| 文件 | `WDSEventDB_all_events.csv`（25,165 × 21）；`WDSEventDB_file_summary.csv`（9 × 5） |
| 时间跨度 | 无绝对时间戳；每个文件内 `relative_time_s` 从 0 开始 |
| 时间粒度 | 按 1 Hz 假设（原始数据未提供时间列） |
| 变量数 | 15 个传感器/执行器 + `label` + 5 个元数据列 |
| 来源 | 供水系统试验台（WDS）事件记录：正常、网络攻击、泄漏、传感器故障 |
| 缺失率 | 0% |

## 2. 数据质量

- 9 个文件：正常 1 个（9,743 行）、网络攻击 1 个（5,869 行，4 段）、泄漏 3 个（1,355~1,385 行）、传感器故障 4 个（1,360~1,362 行）。
- 每个文件都有逐点 `label`，事件段清晰；无缺失。
- **原始数据没有绝对时间戳**，`relative_time_s` 是按 1 Hz 假设生成的相对秒数，仅用于文件内定位；不能当作真实绝对时间。
- 文件之间是独立实验记录，不是连续时间序列，无法直接拼接成长期序列。
- 事件数量少：4 个网络攻击、3 个泄漏、4 个传感器故障文件。

## 3. 标签说明

| 标签 | 含义 |
|---|---|
| `label` | 0=正常；1=事件（网络攻击/泄漏/传感器故障） |
| `event_type` | clean / cyberattack / leakage / sensor_failure |
| `segment_id` | 事件段 ID；同一文件内共享 |
| `relative_time_s` | 文件内相对秒数（假设 1 Hz） |

## 4. 列说明

{table(["列名", "含义", "缺失率"], rows)}
### 每文件摘要

{table(["事件类型", "文件", "行数", "正类数", "正类比例"], srows)}
## 5. 时序预测

**❌ 不建议作为主基准。**

- 每个文件只有 20~23 分钟，最长 `CleanData` 也仅约 2.7 小时；不足以做有意义的时序预测。
- 只能在单个事件文件内做秒级短时预测 demo。

## 6. 异常检测

**⚠️ 有条件可用。**

- 优点：15 个变量、逐点标签、正常/攻击/泄漏/传感器故障四类事件齐全，适合小规模事件级检测和故障分类。
- 缺点：样本量小、事件少、无绝对时间戳、采样率未文档化；不适合大规模统计显著性主基准。
- 适合作为“真实试验台案例”或补充实验。

## 7. 使用注意

- `relative_time_s` 不是真实时间；不要跨文件拼接。
- 按 `segment_id` 分组做切分，避免同文件事件同时出现在 train/test。
- 采样率 1 Hz 是推断值，使用前建议向数据提供方确认。
"""

# ---------------------------------------------------------------- write
writers = [
    ("01_2026_multisource", readme_2026),
    ("02_AquaSense", readme_aqua),
    ("03_BATADAL", readme_batadal),
    ("04_BattLeDIM", readme_battledim),
    ("05_GECCO", readme_gecco),
    ("06_LeakageRepairs", readme_leakage),
    ("07_LeakDB", readme_leakdb),
    ("08_SWaT", readme_swat),
    ("09_WADI", readme_wadi),
    ("10_WDSEventDB", readme_wdsevent),
]
for folder, fn in writers:
    text = fn()
    path = os.path.join(OUT, folder, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("WROTE", path, len(text), "chars", flush=True)
