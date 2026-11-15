import os, glob, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

OUT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"
lines = []

def check(path, label=None, usecols=None, nrows=None):
    rel = os.path.relpath(path, OUT)
    d = pd.read_csv(path, usecols=usecols, nrows=nrows, low_memory=False)
    info = [f"FILE: {rel}", f"  shape: {d.shape}", f"  size_mb: {round(os.path.getsize(path)/1e6,2)}"]
    if "timestamp" in d.columns:
        t = pd.to_datetime(d["timestamp"], errors="coerce")
        info.append(f"  time: {t.min()} .. {t.max()}  monotonic={t.is_monotonic_increasing}  dup={int(t.duplicated().sum())}  bad={int(t.isna().sum())}")
    info.append(f"  missing_overall: {round(float(d.isna().mean().mean()),6)}")
    for c in ["label", "leak_label", "event", "any_leak", "attack_label", "period", "split", "subset"]:
        if c in d.columns:
            vc = d[c].value_counts(dropna=False).head(8).to_dict()
            info.append(f"  {c}: {vc}")
    lines.extend(info)
    print("\n".join(info), flush=True)

for rel in [
    "01_2026_multisource/2026_multisource_hourly.csv",
    "02_AquaSense/aquasense_real_labeled.csv",
    "03_BATADAL/BATADAL_all.csv",
    "04_BattLeDIM/BattLeDIM_2018_5min.csv",
    "04_BattLeDIM/BattLeDIM_2019_5min.csv",
    "05_GECCO/GECCO_all_1min.csv",
    "06_LeakageRepairs/LeakageRepairs_flows_5min.csv",
    "07_LeakDB/Hanoi500_Scenario-010.csv",
    "07_LeakDB/Net1_1000_Scenario-001.csv",
    "10_WDSEventDB/WDSEventDB_all_events.csv",
]:
    p = os.path.join(OUT, rel)
    if os.path.exists(p):
        check(p)

# big files: read only key columns
for rel in ["08_SWaT/SWaT_all_1s.csv", "09_WADI/WADI_all_1s.csv"]:
    p = os.path.join(OUT, rel)
    if os.path.exists(p):
        check(p, usecols=["timestamp", "label", "period"])

with open(os.path.join(OUT, "VALIDATION.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("WROTE VALIDATION.txt", flush=True)
