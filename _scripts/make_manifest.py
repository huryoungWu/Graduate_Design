import os, glob, csv

OUT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"
rows = []
for path in sorted(glob.glob(os.path.join(OUT, "**", "*.csv"), recursive=True)):
    rel = os.path.relpath(path, OUT)
    if rel.startswith("_scripts") or rel == "MANIFEST.csv":
        continue
    with open(path, "rb") as f:
        first = f.readline().decode("utf-8-sig", "replace").rstrip("\r\n")
        n = 1
        first_data = None
        last_data = None
        for line in f:
            if first_data is None:
                first_data = line.decode("utf-8-sig", "replace").rstrip("\r\n")
            last_data = line.decode("utf-8-sig", "replace").rstrip("\r\n")
            n += 1
    cols = next(csv.reader([first]))
    nrows = max(n - 1, 0)
    ts_min = ts_max = ""
    # first column is timestamp for most files; check by header name
    if cols and cols[0] in ("timestamp", "Timestamp") and first_data and last_data:
        ts_min = first_data.split(",", 1)[0]
        ts_max = last_data.split(",", 1)[0]
    rows.append({
        "file": rel,
        "rows": nrows,
        "columns": len(cols),
        "size_mb": round(os.path.getsize(path) / 1e6, 2),
        "timestamp_min": ts_min,
        "timestamp_max": ts_max,
        "first_columns": ";".join(cols[:5]),
    })

import pandas as pd
df = pd.DataFrame(rows)
out = os.path.join(OUT, "MANIFEST.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")
print("WROTE", out, df.shape, flush=True)
print(df.to_string(index=False, max_colwidth=60), flush=True)
