import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

SRC = r"D:\GraduateDesign\Dataset"
OUT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"
os.makedirs(os.path.join(OUT, "04_BattLeDIM"), exist_ok=True)

def load(year, kind, prefix):
    f = os.path.join(SRC, "BattLeDIM", f"{year}_SCADA_{kind}.csv")
    d = pd.read_csv(f, sep=";", decimal=",")
    d.columns = [str(c).strip() for c in d.columns]
    d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="raise")
    d = d.rename(columns={c: f"{prefix}_{c}" for c in d.columns if c != "Timestamp"})
    return d

def load_leaks(year):
    f = os.path.join(SRC, "BattLeDIM", f"{year}_Leakages.csv")
    d = pd.read_csv(f, sep=";", decimal=",")
    d.columns = [str(c).strip() for c in d.columns]
    d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="raise")
    d = d.rename(columns={c: f"leak_{c}" for c in d.columns if c != "Timestamp"})
    return d

def leak_events(df, year):
    leak_cols = [c for c in df.columns if c.startswith("leak_")
                 and c not in ("leak_onset_count", "leak_onset")]
    rows = []
    for c in leak_cols:
        v = (df[c].fillna(0) > 0).values
        edges = np.flatnonzero(np.diff(np.r_[0, v.astype(int), 0]))
        for i in range(0, len(edges), 2):
            a, b = edges[i], edges[i + 1]
            seg = df.iloc[a:b]
            peak_idx = seg[c].idxmax()
            rows.append({
                "year": year,
                "link": c.replace("leak_", ""),
                "start": seg["Timestamp"].iloc[0],
                "end": seg["Timestamp"].iloc[-1],
                "duration_h": round((b - a) * 5 / 60, 2),
                "max_leak_m3h": float(seg[c].max()),
                "mean_leak_m3h": round(float(seg[c].mean()), 4),
                "peak_time": df.loc[peak_idx, "Timestamp"],
                "onset_observed": int(a > 0),
            })
    return pd.DataFrame(rows)

for year in (2018, 2019):
    parts = [
        load(year, "Flows", "flow"),
        load(year, "Pressures", "pressure"),
        load(year, "Levels", "level"),
        load(year, "Demands", "demand"),
        load_leaks(year),
    ]
    df = parts[0]
    for p in parts[1:]:
        df = pd.merge(df, p, on="Timestamp", how="outer", validate="1:1")
    df = df.sort_values("Timestamp").reset_index(drop=True)
    leak_cols = [c for c in df.columns if c.startswith("leak_")]
    df["total_leak_m3h"] = df[leak_cols].fillna(0).sum(axis=1)
    df["any_leak"] = (df["total_leak_m3h"] > 0).astype(int)

    # onset = first 5-min step where a link's leak flow becomes > 0
    onset_count = np.zeros(len(df), dtype=int)
    for c in leak_cols:
        v = df[c].fillna(0)
        onset = ((v > 0) & (v.shift(1).fillna(0) <= 0)).astype(int)
        if len(onset):
            onset.iloc[0] = 0  # pre-existing leak at the very start: onset not observed
        onset_count += onset.to_numpy()
    df["leak_onset_count"] = onset_count
    df["leak_onset"] = (onset_count > 0).astype(int)

    out = os.path.join(OUT, "04_BattLeDIM", f"BattLeDIM_{year}_5min.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig", float_format="%.10g",
              date_format="%Y-%m-%d %H:%M:%S")
    print("WROTE", out, df.shape, round(os.path.getsize(out) / 1e6, 1), "MB", flush=True)
    print(" time", df.Timestamp.min(), "..", df.Timestamp.max(), "dup", int(df.Timestamp.duplicated().sum()), flush=True)
    print(" any_leak", df.any_leak.value_counts().to_dict(), "missing", round(float(df.isna().mean().mean()), 5), flush=True)

    ev = leak_events(df, year)
    evout = os.path.join(OUT, "04_BattLeDIM", f"BattLeDIM_{year}_leak_events.csv")
    ev.to_csv(evout, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    print("WROTE", evout, ev.shape, flush=True)
    print(ev.to_string(index=False), flush=True)
