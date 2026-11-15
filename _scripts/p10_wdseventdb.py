import os, glob, warnings
warnings.filterwarnings("ignore")
import pandas as pd

SRC = r"D:\GraduateDesign\Dataset"
OUT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"
os.makedirs(os.path.join(OUT, "10_WDSEventDB"), exist_ok=True)

mapping = {
    "Clean": "clean",
    "Cyberattack": "cyberattack",
    "Leakage": "leakage",
    "Sensor Failure": "sensor_failure",
}
frames = []
summary = []
sensor_cols = None
for folder, etype in mapping.items():
    for f in sorted(glob.glob(os.path.join(SRC, "WDSEventDB", folder, "*.xlsx"))):
        d = pd.read_excel(f)
        d.columns = [str(c).strip() for c in d.columns]
        label_col = [c for c in d.columns if c.lower() == "labels"][0]
        sensors = [c for c in d.columns if c != label_col]
        if sensor_cols is None:
            sensor_cols = sensors
        assert sensors == sensor_cols, (f, sensors)
        d = d.rename(columns={label_col: "label"})
        d["label"] = pd.to_numeric(d["label"], errors="coerce").fillna(0).astype(int)
        d.insert(0, "event_type", etype)
        d.insert(1, "source_file", os.path.basename(f))
        d.insert(2, "segment_id", f"{etype}:{os.path.splitext(os.path.basename(f))[0]}")
        d.insert(3, "row_index", range(len(d)))
        d.insert(4, "relative_time_s", range(len(d)))
        frames.append(d[["event_type", "source_file", "segment_id", "row_index", "relative_time_s"] + sensors + ["label"]])
        summary.append({"event_type": etype, "source_file": os.path.basename(f), "rows": len(d),
                        "positive": int(d.label.sum()), "positive_pct": round(float(d.label.mean()) * 100, 2)})

all_df = pd.concat(frames, ignore_index=True)
out = os.path.join(OUT, "10_WDSEventDB", "WDSEventDB_all_events.csv")
all_df.to_csv(out, index=False, encoding="utf-8-sig", float_format="%.10g")
print("WROTE", out, all_df.shape, round(os.path.getsize(out) / 1e6, 2), "MB", flush=True)
print("event types", all_df.event_type.value_counts().to_dict(), flush=True)
print("labels", all_df.label.value_counts().to_dict(), flush=True)

sumdf = pd.DataFrame(summary)
sumout = os.path.join(OUT, "10_WDSEventDB", "WDSEventDB_file_summary.csv")
sumdf.to_csv(sumout, index=False, encoding="utf-8-sig")
print("WROTE", sumout, sumdf.shape, flush=True)
print(sumdf.to_string(index=False), flush=True)
