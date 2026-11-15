import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd

SRC = r"D:\GraduateDesign\Dataset"
OUT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"
os.makedirs(os.path.join(OUT, "01_2026_multisource"), exist_ok=True)

df = pd.read_csv(os.path.join(SRC, "2026 多源城市供水管网数据集", "input_model_potenc_predXfault7_A.csv"))
# original first 289 rows are the moved month-first-day block
df["label_reliable"] = 1
df.loc[df.index < 289, "label_reliable"] = 0
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y %H:%M", errors="raise")
df = df.sort_values("timestamp").reset_index(drop=True)

out = os.path.join(OUT, "01_2026_multisource", "2026_multisource_hourly.csv")
df.to_csv(out, index=False, encoding="utf-8-sig", float_format="%.10g",
          date_format="%Y-%m-%d %H:%M:%S")
print("WROTE", out, df.shape, os.path.getsize(out) / 1e6, "MB", flush=True)
print("time", df.timestamp.min(), "..", df.timestamp.max(), "monotonic", df.timestamp.is_monotonic_increasing, flush=True)
print("fault_d7", df.fault_d7.value_counts().to_dict(), "unreliable rows", int((df.label_reliable == 0).sum()), flush=True)
