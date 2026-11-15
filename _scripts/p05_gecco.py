import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

SRC = r"D:\GraduateDesign\Dataset"
OUT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"
os.makedirs(os.path.join(OUT, "05_GECCO"), exist_ok=True)

df = pd.read_csv(os.path.join(SRC, "GECCO", "5_gecco2019_all_water_quality.csv"))
df.columns = [str(c).strip() for c in df.columns]
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
df = df.rename(columns={"Time": "timestamp"})
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

vars6 = ["Tp", "pH", "Cond", "Turb", "SAC", "PFM"]
for c in vars6:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[df[c] == 0, c] = np.nan
df["event"] = df["Event"].astype(int)
df = df.drop(columns=["Event"])

t0 = pd.Timestamp("2017-10-01 00:00:00")
t1 = pd.Timestamp("2017-11-08 00:30:00")
df["split"] = np.where(df["timestamp"] < t0, "train",
                np.where(df["timestamp"] < t1, "valid", "test"))
df = df.sort_values("timestamp").reset_index(drop=True)

cols = ["timestamp"] + vars6 + ["event", "split"]
df = df[cols]
out = os.path.join(OUT, "05_GECCO", "GECCO_all_1min.csv")
df.to_csv(out, index=False, encoding="utf-8-sig", float_format="%.10g",
          date_format="%Y-%m-%d %H:%M:%S")
print("WROTE", out, df.shape, round(os.path.getsize(out) / 1e6, 2), "MB", flush=True)
print("time", df.timestamp.min(), "..", df.timestamp.max(), "monotonic", df.timestamp.is_monotonic_increasing, flush=True)
print("split", df.split.value_counts().to_dict(), "events", int(df.event.sum()), flush=True)
print("missing", df[vars6].isna().mean().round(5).to_dict(), flush=True)

# event intervals
v = df["event"].values
edges = np.flatnonzero(np.diff(np.r_[0, v, 0]))
rows = []
for i in range(0, len(edges), 2):
    a, b = edges[i], edges[i + 1]
    seg = df.iloc[a:b]
    rows.append({
        "start": seg.timestamp.iloc[0],
        "end": seg.timestamp.iloc[-1],
        "duration_min": b - a,
        "split": seg.split.iloc[0],
    })
ev = pd.DataFrame(rows)
evout = os.path.join(OUT, "05_GECCO", "GECCO_events.csv")
ev.to_csv(evout, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
print("WROTE", evout, ev.shape, "event counts by split", ev.split.value_counts().to_dict(), flush=True)
print("event duration min/median/max", ev.duration_min.min(), ev.duration_min.median(), ev.duration_min.max(), flush=True)
