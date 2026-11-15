import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

SRC = r"D:\GraduateDesign\Dataset"
OUT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"
os.makedirs(os.path.join(OUT, "06_LeakageRepairs"), exist_ok=True)

series = []
summary = []
for i in range(1, 7):
    area = f"Area_{i}"
    f = os.path.join(SRC, "Leakage-Repairs-Dataset", "data", "raw_flow_data", f"{area}.csv")
    d = pd.read_csv(f)
    d["result_time"] = pd.to_datetime(d["result_time"], errors="raise")
    d["Flow"] = pd.to_numeric(d["Flow"], errors="coerce")
    d = d.sort_values("result_time")

    n_raw = len(d)
    d.loc[d["Flow"] <= 0, "Flow"] = np.nan
    pos = d["Flow"].dropna()
    med = float(pos.median())
    q999 = float(pos.quantile(0.999))
    cap = max(q999 * 3.0, med * 10.0)
    n_cap = int((d["Flow"] > cap).sum())
    d.loc[d["Flow"] > cap, "Flow"] = np.nan

    s = (d.set_index("result_time")["Flow"]
           .resample("5min").mean()
           .rename(f"{area}_flow"))
    series.append(s)
    summary.append({
        "area": area, "raw_rows": n_raw, "median_flow": round(med, 4),
        "q999": round(q999, 4), "cap": round(cap, 4), "capped_rows": n_cap,
        "reporting_rows_5min": int(s.notna().sum()),
        "missing_5min": int(s.isna().sum()),
        "start": s.index.min(), "end": s.index.max(),
    })
    print(area, summary[-1], flush=True)

flow = pd.concat(series, axis=1, join="outer").sort_index()
flow.index.name = "timestamp"
flow["n_areas_reporting"] = flow.notna().sum(axis=1).astype(int)
flow = flow.reset_index()

out = os.path.join(OUT, "06_LeakageRepairs", "LeakageRepairs_flows_5min.csv")
flow.to_csv(out, index=False, encoding="utf-8-sig", float_format="%.10g",
            date_format="%Y-%m-%d %H:%M:%S")
print("WROTE", out, flow.shape, round(os.path.getsize(out) / 1e6, 2), "MB", flush=True)
print("time", flow.timestamp.min(), "..", flow.timestamp.max(), "monotonic", flow.timestamp.is_monotonic_increasing, flush=True)
print("missing per flow col", flow.filter(like="_flow").isna().mean().round(4).to_dict(), flush=True)

rep = pd.read_csv(os.path.join(SRC, "Leakage-Repairs-Dataset", "data", "processed_reports_data", "leak_reports.csv"))
rep["timestamp"] = pd.to_datetime(rep["timestamp"], errors="raise")
rep["severity"] = rep["severity"].fillna("unknown")
rep = rep.sort_values("timestamp").reset_index(drop=True)
repout = os.path.join(OUT, "06_LeakageRepairs", "LeakageRepairs_reports.csv")
rep.to_csv(repout, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
print("WROTE", repout, rep.shape, flush=True)
print("reports by area", rep.area.value_counts().to_dict(), flush=True)

sumdf = pd.DataFrame(summary)
sumout = os.path.join(OUT, "06_LeakageRepairs", "LeakageRepairs_area_summary.csv")
sumdf.to_csv(sumout, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
print("WROTE", sumout, sumdf.shape, flush=True)
