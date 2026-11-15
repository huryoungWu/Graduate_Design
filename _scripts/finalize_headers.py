import os, glob

OUT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"

changed = []
for path in sorted(glob.glob(os.path.join(OUT, "**", "*.csv"), recursive=True)):
    if "_scripts" in path:
        continue
    with open(path, "rb") as f:
        first = f.readline()
        rest = f.read()
    new = first
    if b"Timestamp" in new:
        new = new.replace(b"Timestamp", b"timestamp", 1)
    if b",Label" in new:
        new = new.replace(b",Label", b",leak_label", 1)
    if new != first:
        tmp = path + ".tmp"
        with open(tmp, "wb") as g:
            g.write(new)
            g.write(rest)
        os.replace(tmp, path)
        changed.append((os.path.relpath(path, OUT), first.decode("utf-8-sig", "replace").strip()[:80],
                        new.decode("utf-8-sig", "replace").strip()[:80]))

print("header changes:", len(changed), flush=True)
for c in changed:
    print(c, flush=True)
