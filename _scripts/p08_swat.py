import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

SRC = r"D:\GraduateDesign\Dataset"
OUT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"
os.makedirs(os.path.join(OUT, "08_SWaT"), exist_ok=True)

def load(path):
    d = pd.read_excel(path, header=1)
    d.columns = [str(c).strip() for c in d.columns]
    d["Timestamp"] = pd.to_datetime(d["Timestamp"].astype(str).str.strip(),
                                    dayfirst=True, format="mixed", errors="raise")
    return d

print("reading normal...", flush=True)
n = load(os.path.join(SRC, "SWaT", "SWaT_Dataset_Normal_v1.xlsx"))
n = n.drop(columns=["Normal/Attack"])
n["label"] = 0
n["period"] = "normal"

print("reading attack...", flush=True)
a = load(os.path.join(SRC, "SWaT", "SWaT_Dataset_Attack_v0.xlsx"))
raw = a["Normal/Attack"].astype(str).str.strip()
a["label"] = raw.str.lower().str.replace(" ", "", regex=False).eq("attack").astype(int)
a["period"] = "attack"
a = a.drop(columns=["Normal/Attack"])
print("attack label raw", raw.value_counts().to_dict(), flush=True)

sensor_cols = [c for c in n.columns if c not in ("Timestamp", "label", "period")]
assert set(sensor_cols) == set(a.columns) - {"Timestamp", "label", "period"}
a = a[["Timestamp"] + sensor_cols + ["label", "period"]]
n = n[["Timestamp"] + sensor_cols + ["label", "period"]]

df = pd.concat([n, a], ignore_index=True).sort_values("Timestamp").reset_index(drop=True)
drop_const = [c for c in sensor_cols if df[c].nunique(dropna=False) <= 1]
print("dropping constant cols:", drop_const, flush=True)
df = df.drop(columns=drop_const)
df = df.rename(columns={"Timestamp": "timestamp"})

out = os.path.join(OUT, "08_SWaT", "SWaT_all_1s.csv")
df.to_csv(out, index=False, encoding="utf-8-sig", float_format="%.10g",
          date_format="%Y-%m-%d %H:%M:%S")
print("WROTE", out, df.shape, round(os.path.getsize(out) / 1e6, 1), "MB", flush=True)
print("time", df.timestamp.min(), "..", df.timestamp.max(), "monotonic", df.timestamp.is_monotonic_increasing,
      "dup", int(df.timestamp.duplicated().sum()), flush=True)
print("label", df.label.value_counts().to_dict(), "period", df.period.value_counts().to_dict(), flush=True)
print("missing", round(float(df.drop(columns=["timestamp", "label", "period"]).isna().mean().mean()), 6), flush=True)

# attack segments
v = df["label"].values
edges = np.flatnonzero(np.diff(np.r_[0, v, 0]))
rows = []
for i in range(0, len(edges), 2):
    s, e = edges[i], edges[i + 1]
    rows.append({"start": df.timestamp.iloc[s], "end": df.timestamp.iloc[e - 1],
                 "duration_s": e - s})
seg = pd.DataFrame(rows)
segout = os.path.join(OUT, "08_SWaT", "SWaT_attack_segments.csv")
seg.to_csv(segout, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
print("WROTE", segout, seg.shape, flush=True)
print(seg.to_string(index=False), flush=True)

with open(os.path.join(OUT, "08_SWaT", "SWaT_dropped_constant_columns.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(drop_const))
