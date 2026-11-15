import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd

SRC = r"D:\GraduateDesign\Dataset"
OUT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"
os.makedirs(os.path.join(OUT, "03_BATADAL"), exist_ok=True)

def load_xlsx(path):
    d = pd.read_excel(path)
    d.columns = [str(c).strip() for c in d.columns]
    d["timestamp"] = pd.to_datetime(d["DATETIME"].astype(str).str.strip(), dayfirst=True, errors="raise")
    d = d.drop(columns=["DATETIME"])
    return d

d3 = load_xlsx(os.path.join(SRC, "BATADAL", "BATADAL_dataset03.xlsx"))
d3 = d3.drop(columns=["ATT_FLAG"])
d3["label"] = 0
d3["subset"] = "train_normal_2014_2015"

d4 = load_xlsx(os.path.join(SRC, "BATADAL", "BATADAL_dataset04.xlsx"))
d4["label"] = (pd.to_numeric(d4["ATT_FLAG"], errors="coerce").fillna(0) == 1).astype(int)
d4 = d4.drop(columns=["ATT_FLAG"])
d4["subset"] = "test_labeled_2016"

d5 = pd.read_csv(os.path.join(SRC, "BATADAL", "BATADAL_test_dataset.zip"))
# pandas can read the zip directly; fall back to zipfile if needed
d5.columns = [str(c).strip() for c in d5.columns]
d5["timestamp"] = pd.to_datetime(d5["DATETIME"].astype(str).str.strip(), dayfirst=True, errors="raise")
d5 = d5.drop(columns=["DATETIME"])
d5["label"] = pd.NA
d5["subset"] = "test_unlabeled_2017"

features = [c for c in d3.columns if c not in ("timestamp", "label", "subset")]
for d in (d4, d5):
    for c in features:
        if c not in d.columns:
            raise KeyError(f"missing {c} in {d.shape}")
    d5[c] = pd.to_numeric(d5[c], errors="coerce")

all_df = pd.concat([d3[["timestamp"] + features + ["label", "subset"]],
                    d4[["timestamp"] + features + ["label", "subset"]],
                    d5[["timestamp"] + features + ["label", "subset"]]],
                   ignore_index=True)
all_df = all_df.sort_values("timestamp").reset_index(drop=True)

out = os.path.join(OUT, "03_BATADAL", "BATADAL_all.csv")
all_df.to_csv(out, index=False, encoding="utf-8-sig", float_format="%.10g",
              date_format="%Y-%m-%d %H:%M:%S")
print("WROTE", out, all_df.shape, os.path.getsize(out) / 1e6, "MB", flush=True)
print("time", all_df.timestamp.min(), "..", all_df.timestamp.max(), "monotonic", all_df.timestamp.is_monotonic_increasing, flush=True)
print("subset counts", all_df.subset.value_counts().to_dict(), flush=True)
print("labels", all_df.label.value_counts(dropna=False).to_dict(), flush=True)
print("dup timestamps", int(all_df.timestamp.duplicated().sum()), flush=True)
