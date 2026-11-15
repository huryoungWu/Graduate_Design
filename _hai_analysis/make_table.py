# -*- coding: utf-8 -*-
import json, os

base = r'C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset\_hai_analysis'

def load(name):
    with open(os.path.join(base, name), encoding='utf-8') as f:
        return json.load(f)

inv = {r['file'].replace('\\', '/'): r for r in load('inventory2.json')}
lab = {r['file']: r for r in load('labels_summary.json')}
inv1 = {r['file'].replace('\\', '/'): r for r in load('inventory.json')}

lines = ['| 文件 | 行数 | 标签列数(含标签) | 起 | 止 | 时长(h) | 攻击段 | 攻击点 | 正例% |',
         '|---|---:|---:|---|---|---:|---:|---:|---:|']
for f in sorted(inv):
    r = inv[f]
    if 'fatal' in r:
        continue
    r1 = inv1.get(f, {})
    l = lab.get(f, {})
    ncols = r['ncols']
    if os.path.basename(f).startswith('label'):
        tags = '-'
    elif 'hai-20.07' in f or 'hai-21.03' in f:
        tags = f'{ncols-5} + 4'
    elif 'hai-22.04' in f:
        tags = f'{ncols-2} + 1'
    else:
        tags = str(ncols - 1)
    rows = r['nrows']
    hours = round(rows / 3600.0, 2)
    seg = l.get('segments', '-')
    pos = l.get('pos', '-')
    pct = l.get('pct', '-')
    pos_s = f'{pos:,}' if isinstance(pos, int) else pos
    lines.append(f"| {f} | {rows:,} | {tags} | {r1.get('tmin','?')} | {r1.get('tmax','?')} | {hours} | {seg} | {pos_s} | {pct} |")

with open(os.path.join(base, 'table.md'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('written', len(lines))
