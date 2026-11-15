# -*- coding: utf-8 -*-
"""Inventory + quality scan of the HAI dataset archive (read-only)."""
import os, json, math, sys
import polars as pl

ROOT = r'D:\法律\archive'
OUT = r'C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset\_hai_analysis'

def sniff(path):
    with open(path, 'rb') as fh:
        header = fh.readline().decode('utf-8-sig', errors='replace').strip()
    sep = ';' if header.count(';') > header.count(',') else ','
    cols = [c.strip() for c in header.split(sep)]
    return sep, cols

def ts_expr(col):
    return pl.col(col).cast(pl.String).str.to_datetime(format='%Y-%m-%d %H:%M:%S', strict=False)

def label_stats(sr):
    sr = sr.cast(pl.Int64, strict=False).fill_null(0)
    pos = int(sr.sum())
    n = len(sr)
    arr = sr.to_numpy()
    seg = int(((arr[1:] == 1) & (arr[:-1] == 0)).sum()) + (1 if n and arr[0] == 1 else 0)
    return pos, n, seg

def analyze(path):
    rel = os.path.relpath(path, ROOT)
    sep, cols = sniff(path)
    lf = pl.scan_csv(path, separator=sep, infer_schema_length=5000, ignore_errors=True)
    schema = lf.collect_schema()
    ncols = len(cols)
    base = lf.select([
        pl.len().alias('n'),
        pl.col(cols[0]).cast(pl.String).first().alias('first_raw'),
        pl.col(cols[0]).cast(pl.String).last().alias('last_raw'),
    ]).collect()
    n = int(base['n'][0])
    first_raw = base['first_raw'][0]
    last_raw = base['last_raw'][0]

    # timestamps + gaps
    info = {'file': rel, 'sep': sep, 'ncols': ncols, 'nrows': n,
            'first': first_raw, 'last': last_raw}
    try:
        tdf = lf.select(ts_expr(cols[0]).alias('t')).collect()
        ts = tdf['t']
        nn = ts.drop_nulls()
        info['ts_valid'] = len(nn)
        if len(nn):
            info['tmin'] = str(nn.min())
            info['tmax'] = str(nn.max())
            d = nn.diff().drop_nulls().dt.total_milliseconds()
            d = d.filter(d > 0)
            if len(d):
                # sample only if huge
                ds = d if len(d) <= 2_000_000 else d.gather_every(len(d)//2_000_000 + 1)
                info['dt_median_s'] = float(ds.median()) / 1000.0
                info['dt_max_s'] = float(d.max()) / 1000.0
                info['gaps_gt_2x'] = int((d > 2 * ds.median()).sum())
        info['dups'] = int((ts.is_duplicated()).sum())
    except Exception as e:
        info['ts_error'] = repr(e)

    # nulls across all columns
    try:
        nc = lf.select(pl.all().null_count()).collect().row(0)
        info['null_total'] = int(sum(v for v in nc if v is not None))
        info['null_cells'] = int(sum(v for v in nc if v is not None))
        info['cells'] = n * ncols
        # per-column nulls > 0
        nz = {c: int(v) for c, v in zip(cols, nc) if v}
        info['null_cols'] = nz if len(nz) <= 20 else dict(list(nz.items())[:20])
        info['null_col_count'] = len(nz)
    except Exception as e:
        info['null_error'] = repr(e)

    # label columns inside the file
    labcols = [c for c in cols if ('attack' in c.lower() or c.lower() in ('label', 'is_attack'))]
    info['label_cols'] = {}
    for c in labcols:
        try:
            sr = lf.select(pl.col(c)).collect().to_series()
            pos, nn_, seg = label_stats(sr)
            info['label_cols'][c] = {'pos': pos, 'rows': nn_, 'segments': seg,
                                     'pct': round(100.0*pos/nn_, 4) if nn_ else None}
        except Exception as e:
            info['label_cols'][c] = {'error': repr(e)}

    # schema hint
    info['dtypes'] = {k: str(v) for k, v in schema.items() if len(schema) <= 10}
    return info

def main():
    paths = []
    for dp, dn, fn in os.walk(ROOT):
        for f in fn:
            if f.lower().endswith('.csv'):
                paths.append(os.path.join(dp, f))
    paths.sort()
    out = []
    for p in paths:
        try:
            r = analyze(p)
        except Exception as e:
            r = {'file': os.path.relpath(p, ROOT), 'fatal': repr(e)}
        out.append(r)
        print(json.dumps(r, ensure_ascii=False))
        sys.stdout.flush()
    with open(os.path.join(OUT, 'inventory.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

if __name__ == '__main__':
    main()
