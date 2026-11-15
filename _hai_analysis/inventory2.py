# -*- coding: utf-8 -*-
"""Accurate missing-value scan: full-schema inference, no silent parse errors."""
import os, json, sys
import polars as pl

ROOT = r'D:\法律\archive'
OUT = r'C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset\_hai_analysis'

def sniff(path):
    with open(path, 'rb') as fh:
        header = fh.readline().decode('utf-8-sig', errors='replace').strip()
    sep = ';' if header.count(';') > header.count(',') else ','
    return sep, [c.strip() for c in header.split(sep)]

def main():
    paths = []
    for dp, dn, fn in os.walk(ROOT):
        for f in fn:
            if f.lower().endswith('.csv'):
                paths.append(os.path.join(dp, f))
    paths.sort()
    out = []
    for p in paths:
        rel = os.path.relpath(p, ROOT)
        sep, cols = sniff(p)
        try:
            df = pl.read_csv(p, separator=sep, infer_schema_length=None)
            n = df.height
            nc = df.select(pl.all().null_count()).row(0)
            nulls = {c: int(v) for c, v in zip(df.columns, nc) if v}
            total = int(sum(nulls.values()))
            rec = {'file': rel, 'nrows': n, 'ncols': df.width, 'null_total': total,
                   'cells': n * df.width,
                   'null_pct': round(100.0 * total / (n * df.width), 4) if n else None,
                   'null_col_count': len(nulls),
                   'null_cols': dict(sorted(nulls.items(), key=lambda kv: -kv[1])[:15]),
                   'dtypes': {k: str(v) for k, v in df.schema.items()}}
            print(json.dumps(rec, ensure_ascii=False))
        except Exception as e:
            rec = {'file': rel, 'fatal': repr(e)}
            print(json.dumps(rec, ensure_ascii=False))
        sys.stdout.flush()
        out.append(rec)
    with open(os.path.join(OUT, 'inventory2.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

if __name__ == '__main__':
    main()
