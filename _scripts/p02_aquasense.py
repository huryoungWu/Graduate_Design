import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

SRC = r"D:\GraduateDesign\Dataset"
OUT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"
os.makedirs(os.path.join(OUT, "02_AquaSense"), exist_ok=True)

df = pd.read_csv(os.path.join(SRC, "AquaSense-ICS", "fused_data.csv"), low_memory=False)
df = df[df["origin"] == "real"].copy()
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
df = df.sort_values("timestamp").reset_index(drop=True)

want = ["timestamp", "temperature", "humidity", "gas_resistance", "tds_value",
        "quality", "quality_final", "wq_score", "wq_class",
        "attack_id", "attack_type", "attack_channel",
        "fault_type", "fault_channel"]
cols = [c for c in want if c in df.columns]
df = df[cols].copy()
df["attack_label"] = (pd.to_numeric(df["attack_id"], errors="coerce").fillna(0) > 0).astype(int)

# add the legacy 4-class quality label from Real_data.csv
real = pd.read_csv(os.path.join(SRC, "AquaSense-ICS", "Real_data.csv"), usecols=["timestamp", "quality"])
real["timestamp"] = pd.to_datetime(real["timestamp"], errors="raise")
real = real.rename(columns={"quality": "quality_legacy"}).drop_duplicates("timestamp")
df = df.merge(real, on="timestamp", how="left", validate="1:1")

out = os.path.join(OUT, "02_AquaSense", "aquasense_real_labeled.csv")
df.to_csv(out, index=False, encoding="utf-8-sig", float_format="%.10g",
          date_format="%Y-%m-%d %H:%M:%S")
print("WROTE", out, df.shape, os.path.getsize(out) / 1e6, "MB", flush=True)
print("time", df.timestamp.min(), "..", df.timestamp.max(), "monotonic", df.timestamp.is_monotonic_increasing, flush=True)
print("attack_label", df.attack_label.value_counts().to_dict(), flush=True)
print("attack types", df.attack_type.value_counts().to_dict(), flush=True)

# attack segments
lab = df["attack_label"].values
edges = np.flatnonzero(np.diff(np.r_[0, lab, 0]))
segs = []
for i in range(0, len(edges), 2):
    a, b = edges[i], edges[i + 1]
    seg = df.iloc[a:b]
    segs.append({
        "start": seg.timestamp.iloc[0],
        "end": seg.timestamp.iloc[-1],
        "duration_s": b - a,
        "attack_id": seg.attack_id.iloc[0],
        "attack_type": seg.attack_type.iloc[0],
        "attack_channel": seg.attack_channel.iloc[0],
    })
segdf = pd.DataFrame(segs)
segout = os.path.join(OUT, "02_AquaSense", "aquasense_attack_segments.csv")
segdf.to_csv(segout, index=False, encoding="utf-8-sig")
print("WROTE", segout, segdf.shape, flush=True)
print(segdf.to_string(index=False), flush=True)
