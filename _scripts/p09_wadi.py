import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

SRC = r"D:\GraduateDesign\Dataset"
OUT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"
os.makedirs(os.path.join(OUT, "09_WADI"), exist_ok=True)
BASE = os.path.join(SRC, "WADI", "WADI.A1_9 Oct 2017")

def parse_wadi_time(df):
    # WADI Time is MM:SS.t and resets every hour; dates are contiguous blocks.
    # Reconstruct by taking the first Time of each date block as the start offset,
    # then adding one second per row within that block.
    date = (df["Date"].astype(str).str.strip()
            .str.replace(" ", " ", regex=False).str.strip())
    p = date.str.extract(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$")
    month = p[0].astype(int)
    day = p[1].astype(int)
    year = p[2].astype(int)
    year = year.where(year >= 100, year + 2000)
    base_date = pd.to_datetime(dict(year=year, month=month, day=day))

    tm = df["Time"].astype(str).str.strip()
    q = tm.str.extract(r"^(\d{1,3}):(\d{2})(?:\.(\d+))?$")
    mins = q[0].astype(float)
    secs = q[1].astype(float)
    frac = q[2].fillna("0").astype(str)
    frac_sec = frac.astype(float) / (10 ** frac.str.len())
    offset = mins * 60 + secs + frac_sec

    block = (date != date.shift()).cumsum()
    ts = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for _, idx in df.groupby(block, sort=False).groups.items():
        ii = np.asarray(idx, dtype=int)
        start = base_date[ii[0]] + pd.to_timedelta(offset.iloc[ii[0]], unit="s")
        ts.iloc[ii] = start + pd.to_timedelta(np.arange(len(ii)), unit="s")
    return ts

def clean_cols(d):
    d.columns = [str(c).strip() for c in d.columns]
    return d

print("reading normal...", flush=True)
n = clean_cols(pd.read_csv(os.path.join(BASE, "WADI_14days_new.csv"), low_memory=False))
n_before = len(n)
n = n.dropna(subset=["Date", "Time"]).reset_index(drop=True)
print("normal dropped missing-timestamp rows:", n_before - len(n), flush=True)
n["Timestamp"] = parse_wadi_time(n)
n["label"] = 0
n["period"] = "normal"
n_sensors = [c for c in n.columns if c not in ("Row", "Date", "Time", "Timestamp", "label", "period")]
n = n[["Timestamp"] + n_sensors + ["label", "period"]]
print("normal", n.shape, "time", n.Timestamp.min(), "..", n.Timestamp.max(), "dup", int(n.Timestamp.duplicated().sum()), flush=True)

print("reading attack...", flush=True)
a = clean_cols(pd.read_csv(os.path.join(BASE, "WADI_attackdataLABLE.csv"), skiprows=1, low_memory=False))
a_before = len(a)
a = a.dropna(subset=["Date", "Time"]).reset_index(drop=True)
print("attack dropped missing-timestamp rows:", a_before - len(a), flush=True)
a["Timestamp"] = parse_wadi_time(a)
label_col = [c for c in a.columns if "Attack LABLE" in c][0]
a["label"] = (pd.to_numeric(a[label_col], errors="coerce") == -1).astype(int)
a["period"] = "attack"
a_sensors = [c for c in a.columns if c not in ("Row", "Date", "Time", "Timestamp", "label", "period", label_col)]
print("attack", a.shape, "label", a["label"].value_counts().to_dict(), "time", a.Timestamp.min(), "..", a.Timestamp.max(), "dup", int(a.Timestamp.duplicated().sum()), flush=True)
assert set(n_sensors) == set(a_sensors), (len(n_sensors), len(a_sensors), list(set(n_sensors) ^ set(a_sensors))[:10])
sensors = n_sensors
a = a[["Timestamp"] + sensors + ["label", "period"]]

# attack segments before combining
v = a["label"].values
edges = np.flatnonzero(np.diff(np.r_[0, v, 0]))
rows = []
for i in range(0, len(edges), 2):
    s, e = edges[i], edges[i + 1]
    rows.append({"start": a.Timestamp.iloc[s], "end": a.Timestamp.iloc[e - 1], "duration_s": e - s})
seg = pd.DataFrame(rows)
segout = os.path.join(OUT, "09_WADI", "WADI_attack_segments.csv")
seg.to_csv(segout, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
print("WROTE", segout, seg.shape, flush=True)
print(seg.to_string(index=False), flush=True)

df = pd.concat([n, a], ignore_index=True).sort_values("Timestamp").reset_index(drop=True)
drop_const = [c for c in sensors if df[c].nunique(dropna=False) <= 1]
print("dropping constant cols", len(drop_const), drop_const, flush=True)
df = df.drop(columns=drop_const)
df = df.rename(columns={"Timestamp": "timestamp"})

out = os.path.join(OUT, "09_WADI", "WADI_all_1s.csv")
df.to_csv(out, index=False, encoding="utf-8-sig", float_format="%.10g",
          date_format="%Y-%m-%d %H:%M:%S")
print("WROTE", out, df.shape, round(os.path.getsize(out) / 1e6, 1), "MB", flush=True)
print("time", df.timestamp.min(), "..", df.timestamp.max(), "monotonic", df.timestamp.is_monotonic_increasing,
      "dup", int(df.timestamp.duplicated().sum()), flush=True)
print("label", df.label.value_counts().to_dict(), "period", df.period.value_counts().to_dict(), flush=True)
print("missing", round(float(df.drop(columns=["timestamp", "label", "period"]).isna().mean().mean()), 6), flush=True)

with open(os.path.join(OUT, "09_WADI", "WADI_dropped_constant_columns.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(drop_const))
