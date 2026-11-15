# -*- coding: utf-8 -*-
"""Attack label statistics for every labeled HAI file."""
import os, json
import numpy as np
import polars as pl

ROOT = r'D:\法律\archive'
OUT = r'C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset\_hai_analysis'

def segments(arr):
    """Return list of (start_idx, end_idx_inclusive) for runs of 1."""
    if len(arr) == 0:
        return []
    d = np.diff(arr.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts, ends))

def stats(abs_path_or_df, ts_col, lab_col, rel, sep=','):
    df = pl.read_csv(abs_path_or_df, separator=sep, infer_schema_length=None)
    lab = df[lab_col].to_numpy().astype(np.int8)
    vals = sorted(set(lab.tolist()))
    segs = segments(lab)
    ts = df[ts_col].to_list()
    recs = []
    for s, e in segs:
        recs.append({'start': ts[s], 'end': ts[e], 'seconds': int(e - s + 1)})
    durs = [r['seconds'] for r in recs]
    return {
        'file': rel, 'rows': df.height, 'label_col': lab_col, 'label_values': vals,
        'pos': int(lab.sum()), 'pct': round(100.0 * lab.sum() / len(lab), 4),
        'segments': len(segs),
        'dur_total_s': int(sum(durs)),
        'dur_min_s': int(min(durs)) if durs else 0,
        'dur_max_s': int(max(durs)) if durs else 0,
        'dur_mean_s': round(float(np.mean(durs)), 1) if durs else 0,
        'events': recs,
    }

targets = []
# 20.07 / 21.03 / 22.04: label column inside the file
for folder, sep, labcol in [('hai-20.07', ';', 'attack'), ('hai-21.03', ',', 'attack'), ('hai-22.04', ',', 'Attack')]:
    for f in sorted(os.listdir(os.path.join(ROOT, folder))):
        if f.endswith('.csv'):
            p = os.path.join(ROOT, folder, f)
            # detect time column name from header
            with open(p, encoding='utf-8-sig') as fh:
                h = fh.readline().strip().split(sep)
            targets.append((p, h[0], labcol, f'{folder}/{f}', sep))
# 23.05 label files
for folder in ['hai-23.05', 'haiend-23.05']:
    for f in sorted(os.listdir(os.path.join(ROOT, folder))):
        if f.startswith('label') and f.endswith('.csv'):
            p = os.path.join(ROOT, folder, f)
            targets.append((p, 'timestamp', 'label', f'{folder}/{f}', ','))

out = []
for p, ts, lab, rel, sep in targets:
    try:
        r = stats(p, ts, lab, rel, sep)
    except Exception as e:
        r = {'file': rel, 'fatal': repr(e)}
    out.append(r)
    print(json.dumps({k: v for k, v in r.items() if k != 'events'}, ensure_ascii=False))
with open(os.path.join(OUT, 'labels_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
# print event tables for the test files
for r in out:
    if 'events' in r and r['segments']:
        print('\n##', r['file'], 'segments', r['segments'])
        for k, ev in enumerate(r['events'], 1):
            print(f"  {k:2d} {ev['start']} -> {ev['end']}  {ev['seconds']}s")
