import os, io, zipfile, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

SRC = r"D:\GraduateDesign\Dataset"
OUT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"
os.makedirs(os.path.join(OUT, "07_LeakDB"), exist_ok=True)

SOURCES = [
    ("Hanoi_CMH.zip", "Hanoi_CMH", "Hanoi500"),
    ("Hanoi_CMH_1000scenarios.zip", "Hanoi_CMH", "Hanoi1000"),
    ("Net1_CMH.zip", "Net1_CMH", "Net1_1000"),
]

def read_member(zf, name):
    with zf.open(name) as f:
        return pd.read_csv(f)

def read_labels(zf, root):
    d = read_member(zf, f"{root}/Labels.csv")
    d.columns = [str(c).strip() for c in d.columns]
    key = "Timestamp" if "Timestamp" in d.columns else "Index"
    d["_key"] = pd.to_datetime(d[key], errors="coerce") if key == "Timestamp" else pd.to_numeric(d[key])
    return d[["_key", "Label"]].rename(columns={"_key": key})

def select_scenarios(zf, root, n_leak=10):
    d = read_member(zf, f"{root}/Labels.csv")
    d.columns = ["Scenario", "HasLeak"]
    d["Scenario"] = pd.to_numeric(d["Scenario"])
    free = d.loc[d.HasLeak == 0, "Scenario"].tolist()
    leaks = d.loc[d.HasLeak == 1, "Scenario"].tolist()[:n_leak]
    fracs = []
    for s in leaks:
        lab = read_labels(zf, f"{root}/Scenario-{int(s)}")
        fracs.append((s, float(lab["Label"].mean())))
    fracs.sort(key=lambda x: x[1])
    chosen = []
    if free:
        chosen.append((int(free[0]), 0.0))
    if fracs:
        chosen.append((int(fracs[0][0]), fracs[0][1]))
        chosen.append((int(fracs[-1][0]), fracs[-1][1]))
    out, seen = [], set()
    for s, f in chosen:
        if s not in seen:
            out.append((s, f)); seen.add(s)
    return out

def read_scenario(zf, root, scen):
    base = f"{root}/Scenario-{scen}"
    names = zf.namelist()
    frames = []
    key_used = None
    for sub, prefix in [("Demands", "demand"), ("Flows", "flow"), ("Pressures", "pressure"), ("Leaks", "leak")]:
        files = [n for n in names
                 if n.startswith(f"{base}/{sub}/") and n.endswith(".csv") and not n.endswith("_info.csv")]
        for n in files:
            d = read_member(zf, n)
            d.columns = [str(c).strip() for c in d.columns]
            key = "Timestamp" if "Timestamp" in d.columns else "Index"
            if key_used is None:
                key_used = key
            if key != key_used:
                raise ValueError(f"key mismatch {n}")
            if key == "Timestamp":
                d[key] = pd.to_datetime(d[key], errors="raise")
            val = [c for c in d.columns if c != key][0]
            stem = os.path.basename(n)[:-4]
            if sub == "Leaks":
                stem = stem.replace("_demand", "")
            newname = f"{prefix}_{stem}"
            d = d[[key, val]].rename(columns={val: newname}).set_index(key)
            frames.append(d)
    wide = pd.concat(frames, axis=1)
    lab = read_labels(zf, base).set_index(key_used)
    wide = wide.join(lab, how="left")
    wide = wide.sort_index()

    if key_used == "Index":
        idx = pd.to_numeric(pd.Series(wide.index), errors="coerce").to_numpy().astype(int)
        ts = pd.Timestamp("2017-01-01 00:00:00") + pd.to_timedelta((idx - 1) * 30, unit="min")
        wide.insert(0, "timestamp", ts)
    else:
        wide.insert(0, "timestamp", pd.to_datetime(wide.index, errors="raise"))
    wide = wide.reset_index(drop=True)
    return wide

index_rows = []
selected_rows = []
for archive, network, tag in SOURCES:
    zpath = os.path.join(SRC, "LeakDB", archive)
    with zipfile.ZipFile(zpath) as zf:
        labels_all = read_member(zf, f"{network}/Labels.csv")
        labels_all.columns = ["Scenario", "HasLeak"]
        for _, r in labels_all.iterrows():
            index_rows.append({"network": network, "archive": archive, "scenario": int(r.Scenario), "has_leak": int(r.HasLeak)})

        chosen = select_scenarios(zf, network)
        print(f"{tag}: selected {chosen}", flush=True)
        for scen, frac in chosen:
            df = read_scenario(zf, network, scen)
            has_leak = int(df["Label"].fillna(0).max() > 0)
            outname = f"{tag}_Scenario-{scen:03d}.csv"
            out = os.path.join(OUT, "07_LeakDB", outname)
            df.to_csv(out, index=False, encoding="utf-8-sig", float_format="%.10g",
                      date_format="%Y-%m-%d %H:%M:%S")
            selected_rows.append({
                "network": network, "archive": archive, "scenario": scen,
                "has_leak": has_leak, "leak_fraction": round(float(df["Label"].mean()), 6),
                "rows": len(df), "columns": df.shape[1], "output_file": outname,
            })
            print("WROTE", out, df.shape, round(os.path.getsize(out) / 1e6, 2), "MB",
                  "time", df.timestamp.min(), "..", df.timestamp.max(),
                  "leak_frac", round(float(df["Label"].mean()), 4), flush=True)

idx = pd.DataFrame(index_rows)
idxout = os.path.join(OUT, "07_LeakDB", "LeakDB_all_scenarios_index.csv")
idx.to_csv(idxout, index=False, encoding="utf-8-sig")
print("WROTE", idxout, idx.shape, flush=True)

sel = pd.DataFrame(selected_rows)
selout = os.path.join(OUT, "07_LeakDB", "LeakDB_selected_scenarios.csv")
sel.to_csv(selout, index=False, encoding="utf-8-sig")
print("WROTE", selout, sel.shape, flush=True)
print(sel.to_string(index=False), flush=True)
