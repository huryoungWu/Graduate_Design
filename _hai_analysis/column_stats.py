# -*- coding: utf-8 -*-
"""Per-column statistics for representative HAI files."""
import os, json
import polars as pl

ROOT = r'D:\法律\archive'
OUT = r'C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset\_hai_analysis'
FILES = [
    ('hai-20.07/train1.csv', ';'),
    ('hai-21.03/train1.csv', ','),
    ('hai-22.04/train1.csv', ','),
    ('hai-23.05/hai-train1.csv', ','),
    ('hai-23.05/hai-test1.csv', ','),
    ('haiend-23.05/end-train1.csv', ','),
]

for rel, sep in FILES:
    path = os.path.join(ROOT, rel.replace('/', os.sep))
    df = pl.read_csv(path, separator=sep, infer_schema_length=None)
    n = df.height
    rows = []
    for c in df.columns:
        s = df[c]
        dtype = str(s.dtype)
        nulls = int(s.null_count())
        try:
            nu = int(s.n_unique())
        except Exception:
            nu = None
        rec = {'col': c, 'dtype': dtype, 'nulls': nulls, 'null_pct': round(100*nulls/n, 4), 'n_unique': nu}
        if s.dtype.is_numeric():
            rec['min'] = s.min()
            rec['max'] = s.max()
            rec['mean'] = round(float(s.mean()), 6) if s.mean() is not None else None
            rec['std'] = round(float(s.std()), 6) if s.std() is not None else None
            rec['zeros'] = int((s == 0).sum())
        rows.append(rec)
    out = os.path.join(OUT, 'cols_' + rel.replace('/', '_').replace('.csv', '') + '.csv')
    pl.DataFrame(rows).write_csv(out)
    const = [r['col'] for r in rows if r['n_unique'] == 1]
    binary = [r['col'] for r in rows if r['n_unique'] == 2]
    numeric = [r for r in rows if 'min' in r]
    miss = sorted(rows, key=lambda r: -r['nulls'])[:5]
    print(f'=== {rel}: rows={n} cols={len(df.columns)} numeric={len(numeric)} const={len(const)} binary={len(binary)}')
    print('  missing top:', [(r['col'], r['null_pct']) for r in miss if r['nulls']])
    print('  constant:', const[:20])
    print('  binary:', binary[:30])
