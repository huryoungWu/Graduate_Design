# -*- coding: utf-8 -*-
"""
汇总 itransformer_improved.py 生成的各变体结果，输出：
  improved_results/all_variants_prediction.csv
  improved_results/all_variants_detection.csv
  improved_results/improvement_report.md

用法：python compare_itransformer_variants.py
"""

import json
import os

import numpy as np
import pandas as pd

OUT_DIR = "improved_results_v2"
ORDER = ["baseline", "graph", "dual", "multiscale",
         "grouped", "patch", "condition", "tcdit"]
LABELS = {
    "baseline": "baseline（原始 iTransformer，同管线重跑）",
    "graph": "Graph-iTransformer",
    "dual": "Dual-iTransformer",
    "multiscale": "Multi-scale iTransformer",
    "grouped": "Grouped iTransformer",
    "patch": "Patch-iTransformer",
    "condition": "Condition-aware iTransformer",
    "tcdit": "TC-DiTransformer（组合模型）",
}


def load_metrics():
    out = {}
    for v in ORDER:
        p = os.path.join(OUT_DIR, f"{v}_metrics.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out[v] = json.load(f)
    return out


def load_historical_baseline():
    """原始 itransformer_model.py 保存的结果，仅作参考。"""
    ref = {}
    ch_path = "itransformer_swat_channel_metrics.csv"
    det_path = "itransformer_swat_anomaly_score_comparison.csv"
    if os.path.exists(ch_path):
        ch = pd.read_csv(ch_path)
        ref["MAE"] = float(ch["MAE"].mean())
        ref["MSE"] = float((ch["RMSE"] ** 2).mean())
        ref["RMSE"] = float(np.sqrt(ref["MSE"]))
    if os.path.exists(det_path):
        det = pd.read_csv(det_path)
        row = det[det["name"] == "mean"]
        if len(row):
            row = row.iloc[0]
            ref["detection_mean"] = {
                "AUC": float(row["AUC"]), "PR_AUC": float(row["PR_AUC"]),
                "Precision": float(row["Precision"]),
                "Recall": float(row["Recall"]), "F1": float(row["F1"]),
            }
    return ref


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}"


def pct(x):
    if x is None or not np.isfinite(x):
        return "-"
    return f"{x:+.2f}%"


def main():
    metrics = load_metrics()
    if not metrics:
        print("未找到任何结果，请先运行 itransformer_improved.py")
        return

    missing = [v for v in ORDER if v not in metrics]
    if missing:
        print(f"警告：以下变体结果缺失：{missing}")

    hist = load_historical_baseline()
    base = metrics.get("baseline")
    base_rmse = base["prediction"]["RMSE"] if base else None
    base_mae = base["prediction"]["MAE"] if base else None
    base_f1 = None
    base_auc = None
    base_val_f1 = None
    if base:
        for d in base["detection"]:
            if d["name"] == "raw_mean":
                base_f1, base_auc = d["F1"], d["AUC"]
                base_val_f1 = d.get("val_F1")
                break

    # ---------- 预测精度表 ----------
    pred_rows = []
    for v in ORDER:
        if v not in metrics:
            continue
        m = metrics[v]
        p = m["prediction"]
        pred_rows.append({
            "variant": v,
            "label": LABELS[v],
            "params": m["params"],
            "MAE": p["MAE"],
            "MSE": p["MSE"],
            "RMSE": p["RMSE"],
            "MAPE_mean_channels": p.get("MAPE_mean_channels"),
            "train_time_s": m["train_time_s"],
            "delta_MAE_vs_baseline_pct": (100 * (p["MAE"] - base_mae) / base_mae
                                          if base_mae else np.nan),
            "delta_RMSE_vs_baseline_pct": (100 * (p["RMSE"] - base_rmse) / base_rmse
                                           if base_rmse else np.nan),
        })
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(os.path.join(OUT_DIR, "all_variants_prediction.csv"),
                   index=False, encoding="utf-8-sig")

    # ---------- 异常检测长表 ----------
    det_rows = []
    for v in ORDER:
        if v not in metrics:
            continue
        m = metrics[v]
        best_name = m.get("protocol", {}).get(
            "selected_score", m["best_detection"]["name"])
        for d in m["detection"]:
            det_rows.append({
                "variant": v,
                "label": LABELS[v],
                "score": d["name"],
                "AUC": d["AUC"],
                "PR_AUC": d["PR_AUC"],
                "Precision": d["Precision"],
                "Recall": d["Recall"],
                "F1": d["F1"],
                "FPR": d.get("FPR", np.nan),
                "val_F1": d.get("val_F1", np.nan),
                "val_PR_AUC": d.get("val_PR_AUC", np.nan),
                "threshold": d["threshold"],
                "threshold_source": d.get("threshold_source", ""),
                "is_best_of_variant": d["name"] == best_name,
                "delta_F1_vs_baseline_raw_mean": (d["F1"] - base_f1
                                                  if base_f1 is not None else np.nan),
                "delta_val_F1_vs_baseline_raw_mean": (
                    d.get("val_F1", np.nan) - base_val_f1
                    if base_val_f1 is not None else np.nan),
            })
    det_df = pd.DataFrame(det_rows)
    det_df.to_csv(os.path.join(OUT_DIR, "all_variants_detection.csv"),
                  index=False, encoding="utf-8-sig")

    # ---------- 生成报告 ----------
    lines = []
    lines.append("# iTransformer 改进变体实验结果")
    lines.append("")
    lines.append("## 1. 实验设置")
    lines.append("")
    b0 = base or {}
    seq_len = b0.get("seq_len", 60)
    pred_len = b0.get("pred_len", "?")
    eps = b0.get("epochs", "?")
    bs = b0.get("batch_size", "?")
    amp_txt = "；启用 AMP 混合精度" if b0.get("amp") else ""
    lines.append("- 数据：SWaT 1 秒重采样数据，45 个传感器/执行器通道。")
    lines.append("- 切分：前 495,000 行（原始 Normal 段）训练；原始 Attack 段前 20% 验证、后 80% 测试。")
    lines.append(f"- 任务：输入 {seq_len} 秒窗口，预测未来 {pred_len} 个时刻的全部 45 个通道"
                 f"（异常检测只使用第 1 步预测）。")
    lines.append(f"- 训练：仅使用 Normal 段；每个变体 {eps} epochs、batch {bs}、Adam、lr 1e-4、"
                 f"CosineAnnealing、MSE{amp_txt}。")
    lines.append("- 公平性：所有变体（含 baseline 重跑）使用同一数据划分、窗口、超参、随机种子与训练预算。")
    lines.append("- 校准：从整个 Normal 段均匀抽样 10% 正常窗口估计逐通道残差尺度（不参与训练）；"
                 "校准期恒定的通道不参与 z/az 聚合。")
    lines.append("- 异常检测：评分配置 raw/z/az；主指标为 AUC/PR-AUC（与阈值无关），"
                 "F1 采用固定 FPR=1% 口径：阈值 = 验证集正常点分数的 99% 分位，"
                 "测试集标签不参与任何选择。")
    lines.append("")
    lines.append("## 2. 预测精度（标准化空间）")
    lines.append("")
    lines.append("| 变体 | 参数量 | MAE | MSE | RMSE | ΔMAE | ΔRMSE | 训练耗时 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in pred_df.sort_values("RMSE").iterrows():
        lines.append(
            f"| {r['label']} | {int(r['params']):,} | {fmt(r['MAE'], 6)} | "
            f"{fmt(r['MSE'], 6)} | {fmt(r['RMSE'], 6)} | "
            f"{pct(r['delta_MAE_vs_baseline_pct'])} | "
            f"{pct(r['delta_RMSE_vs_baseline_pct'])} | {r['train_time_s']:.0f}s |")
    lines.append("")
    if "MAE" in hist:
        lines.append(f"参考：旧版脚本保存结果（旧 70/30 划分、测试集最优阈值）"
                     f"MAE={hist['MAE']:.6f}，RMSE={hist['RMSE']:.6f}；"
                     f"仅作历史参考，与新协议结果不可直接比较。")
        lines.append("")

    lines.append("## 3. 异常检测精度（baseline 同协议：raw_mean）")
    lines.append("")
    lines.append("| 变体 | AUC | PR-AUC | val F1 | Precision | Recall | test F1 | ΔF1(vs baseline) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    raw_mean = det_df[det_df["score"] == "raw_mean"]
    for _, r in raw_mean.sort_values("val_F1", ascending=False).iterrows():
        lines.append(
            f"| {r['label']} | {fmt(r['AUC'])} | {fmt(r['PR_AUC'])} | "
            f"{fmt(r['val_F1'])} | {fmt(r['Precision'])} | {fmt(r['Recall'])} | "
            f"{fmt(r['F1'])} | {r['delta_F1_vs_baseline_raw_mean']:+.4f} |")
    lines.append("")
    lines.append("说明：F1 为验证集正常点 99% 分位阈值（目标 FPR=1%）下的测试集结果；"
                 "FPR 为测试集实际误报率，受域漂移影响可能高于 1%。")
    lines.append("")
    if "detection_mean" in hist:
        d = hist["detection_mean"]
        lines.append(f"参考：旧版脚本保存的 mean 聚合结果 AUC={d['AUC']:.4f}，"
                     f"PR-AUC={d['PR_AUC']:.4f}，F1={d['F1']:.4f}"
                     f"（旧 70/30 划分、测试集最优阈值，仅作历史参考）。")
        lines.append("")

    lines.append("## 4. 各变体验证集选出的评分配置")
    lines.append("")
    lines.append("| 变体 | 选定分数 | val PR-AUC | val F1 | AUC | PR-AUC | Precision | Recall | test F1 | test FPR |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in ORDER:
        if v not in metrics:
            continue
        b = metrics[v]["best_detection"]
        lines.append(
            f"| {LABELS[v]} | {b['name']} | {fmt(b.get('val_PR_AUC'))} | "
            f"{fmt(b.get('val_F1'))} | "
            f"{fmt(b['AUC'])} | {fmt(b['PR_AUC'])} | "
            f"{fmt(b['Precision'])} | {fmt(b['Recall'])} | "
            f"{fmt(b['F1'])} | {fmt(b.get('FPR'))} |")
    lines.append("")
    n_cfg = len(next(iter(metrics.values()))["detection"])
    lines.append(f"说明：raw_* 为原始绝对残差聚合；z_* 为按校准集逐通道残差尺度标准化并逐通道裁剪"
                 f"（--z-clip，默认 20）后的聚合，用于抑制单通道分布漂移；"
                 f"az_* 为可选的滚动自适应版本（默认关闭）。"
                 f"每个变体在 {n_cfg} 种评分配置中、默认只在 mean 类（raw_mean/z_mean）里"
                 f"按**验证集 PR-AUC**（与阈值无关）选最优；"
                 f"阈值取验证集正常点分数的 99% 分位（目标 FPR=1%），测试集指标只报告，不参与选择。")
    lines.append("")

    lines.append("## 5. 残差标准化（z-score）的影响")
    lines.append("")
    lines.append("| 变体 | raw_mean F1 | z_mean F1 | az_mean F1 | ΔF1(z-raw) | raw_mean AUC | z_mean AUC | az_mean AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for v in ORDER:
        if v not in metrics:
            continue
        d = {x["name"]: x for x in metrics[v]["detection"]}
        lines.append(
            f"| {LABELS[v]} | {fmt(d['raw_mean']['F1'])} | {fmt(d['z_mean']['F1'])} | "
            f"{fmt(d.get('az_mean', {}).get('F1'))} | "
            f"{d['z_mean']['F1'] - d['raw_mean']['F1']:+.4f} | "
            f"{fmt(d['raw_mean']['AUC'])} | {fmt(d['z_mean']['AUC'])} | "
            f"{fmt(d.get('az_mean', {}).get('AUC'))} |")
    lines.append("")

    # ---------- 自动结论 ----------
    lines.append("## 6. 结论摘要")
    lines.append("")
    best_pred = pred_df.sort_values("RMSE").iloc[0]
    lines.append(f"1. 预测精度最优变体：**{best_pred['label']}**，RMSE={best_pred['RMSE']:.6f}，"
                 f"相对 baseline 变化 {best_pred['delta_RMSE_vs_baseline_pct']:+.2f}%，"
                 f"MAE 变化 {best_pred['delta_MAE_vs_baseline_pct']:+.2f}%。")
    pred_improved = pred_df[(pred_df["variant"] != "baseline") &
                            (pred_df["delta_RMSE_vs_baseline_pct"] < 0)]
    if len(pred_improved):
        names = "、".join(pred_improved["label"].tolist())
        lines.append(f"2. 预测 RMSE 优于 baseline 的变体：{names}。")
    else:
        lines.append("2. 没有任何变体的预测 RMSE 优于 baseline（同管线重跑）。")

    sort_col = "val_PR_AUC" if "val_PR_AUC" in raw_mean.columns else "val_F1"
    raw_sorted = raw_mean.sort_values(sort_col, ascending=False)
    best_det = raw_sorted.iloc[0]
    lines.append(f"3. baseline 同协议（raw_mean）下按验证集 {sort_col} 选出的变体：**{best_det['label']}**，"
                 f"val_PR-AUC={fmt(best_det.get('val_PR_AUC'))}，val_F1={fmt(best_det.get('val_F1'))}，"
                 f"test_F1={best_det['F1']:.4f}，test_FPR={fmt(best_det.get('FPR'))}，"
                 f"AUC={best_det['AUC']:.4f}。")
    det_improved = raw_mean[(raw_mean["variant"] != "baseline") &
                            (raw_mean["delta_val_F1_vs_baseline_raw_mean"] > 0)]
    if len(det_improved):
        names = "、".join(
            f"{r['label']}（val {r['delta_val_F1_vs_baseline_raw_mean']:+.4f}）"
            for _, r in det_improved.sort_values(
                "delta_val_F1_vs_baseline_raw_mean", ascending=False).iterrows())
        lines.append(f"4. raw_mean 下验证集 F1 高于 baseline 的变体：{names}。")
    else:
        lines.append("4. raw_mean 下没有变体的验证集 F1 高于 baseline。")

    lines.append("")
    lines.append("## 7. 注意事项")
    lines.append("")
    lines.append("- 单随机种子、单次训练；小于 1% 的差异可能来自训练噪声，正式结论建议多种子重复。")
    lines.append("- 评分配置按验证集 PR-AUC（与阈值无关）选取；阈值取验证集正常点分数的 99% 分位"
                 "（固定 FPR=1% 口径），测试集只报告最终指标。"
                 "如需完全无标签的阈值，可用 `--threshold-mode calib` 按 Normal 校准集分位数确定阈值。")
    lines.append("- az_* 为可选的滚动自适应分数（--roll-window>0）；SWaT 上实测会吸收持续攻击信号，"
                 "默认关闭。z_* 已采用逐通道裁剪（--z-clip，默认 20）抑制单通道分布漂移。")
    lines.append("- 训练只使用原始 Normal 段（前 495,000 行），验证/测试均来自原始 Attack 段，"
                 "与 SWaT 官方 Normal/Attack 划分一致。")
    lines.append("- 异常检测只使用预测残差分数，未实现 improve.md 5.2 的变量关联偏差分数；"
                 "该分数需要额外的正常注意力参考与校准流程，建议作为后续工作。")
    lines.append("- 所有指标在标准化空间计算；未生成逐通道逐日对比图。")
    lines.append("")

    report_path = os.path.join(OUT_DIR, "improvement_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已生成: {report_path}")
    print(f"已生成: {os.path.join(OUT_DIR, 'all_variants_prediction.csv')}")
    print(f"已生成: {os.path.join(OUT_DIR, 'all_variants_detection.csv')}")


if __name__ == "__main__":
    main()
