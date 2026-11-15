import sys, os, json, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

path = sys.argv[1]
nrows = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "all" else None
max_cols_show = 15

print("=" * 100)
print("FILE:", path)
print("SIZE_MB:", round(os.path.getsize(path) / 1e6, 2))

# encoding fallback
for enc in ["utf-8", "latin-1"]:
    try:
        df = pd.read_csv(path, nrows=nrows, low_memory=False, encoding=enc)
        print("ENCODING:", enc)
        break
    except UnicodeDecodeError:
        continue

print("SHAPE(sampled):", df.shape)
print("NCOLS:", df.shape[1])
print("COLUMNS:", list(df.columns)[:80])
if df.shape[1] > 80:
    print("... total cols:", df.shape[1])

# time columns
time_cols = []
for c in df.columns:
    lc = str(c).lower().strip()
    if any(k in lc for k in ["time", "date", "timestamp", "datetime"]):
        time_cols.append(c)
print("TIME_COLS:", time_cols)
for c in time_cols[:3]:
    s = df[c]
    print(f"  [{c}] dtype={s.dtype} first={s.iloc[0]!r} last={s.iloc[-1]!r} n_null={s.isna().sum()}")
    if "time" != str(c).lower().strip():
        try:
            t = pd.to_datetime(s, errors="coerce", format="mixed")
        except TypeError:
            t = pd.to_datetime(s, errors="coerce")
        ok = t.notna().sum()
        print(f"    parsed_ok={ok}/{len(t)}", end="")
        if ok > 2:
            tt = t.dropna().sort_values()
            diffs = tt.diff().dropna()
            if len(diffs):
                print(f" range={tt.min()} .. {tt.max()} mode_diff={diffs.mode().iloc[0]} "
                      f"median_diff={diffs.median()} n_unique_diff={diffs.nunique()}")
        else:
            print()

# labels
label_kw = ["event", "label", "attack", "fault", "anomaly", "leak", "target", "class", "status", "quality"]
label_cols = [c for c in df.columns if any(k in str(c).lower() for k in label_kw)]
print("LABEL_CANDIDATES:", label_cols)
for c in label_cols[:12]:
    vc = df[c].value_counts(dropna=False).head(8)
    print(f"  [{c}] nunique={df[c].nunique(dropna=True)} null={df[c].isna().sum()} top={dict(vc)}")

# missing summary
miss = df.isna().mean()
print("MISSING: overall=%.4f  cols>0=%d  cols>50%%=%d" % (miss.mean(), (miss > 0).sum(), (miss > 0.5).sum()))
worst = miss.sort_values(ascending=False).head(8)
print("  worst:", {k: round(v, 4) for k, v in worst.items() if v > 0})

# dtypes summary
print("DTYPES:", df.dtypes.value_counts().to_dict())

# numeric summary
num = df.select_dtypes(include=[np.number])
print("NUMERIC_COLS:", num.shape[1])
if num.shape[1]:
    desc = num.describe().T[["mean", "std", "min", "max"]]
    print(desc.head(20).round(4).to_string())

# constant / id-like columns
const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
print("CONSTANT_COLS(%d):" % len(const), const[:30])

# duplicate timestamps if clear time col
tc = None
for c in df.columns:
    if str(c).lower().strip() in ("timestamp", "time", "datetime", "date") or str(c).lower().strip().endswith("time"):
        tc = c
        break
if tc:
    try:
        t = pd.to_datetime(df[tc], errors="coerce", format="mixed")
        print("DUP_TIMESTAMPS(sampled):", int(t.duplicated().sum()))
    except Exception:
        pass

print("HEAD:")
print(df.head(3).to_string(max_cols=20))
print()
