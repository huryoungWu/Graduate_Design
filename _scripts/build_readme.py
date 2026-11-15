import os, pandas as pd

OUT = r"C:\Users\ASUS\Documents\GraduateDesign_Processed_Dataset"
TEMPLATE = r"C:\Users\ASUS\AppData\Local\Temp\dataset_analysis\README_template.md"

text = open(TEMPLATE, encoding="utf-8").read()
mf = pd.read_csv(os.path.join(OUT, "MANIFEST.csv"))

def dim(prefix):
    m = mf[mf["file"].str.replace("\\", "/").str.endswith(prefix)]
    if m.empty:
        return "见 MANIFEST.csv"
    r = m.iloc[0]
    return f"{int(r['rows']):,} × {int(r['columns'])}"

text = text.replace("{{SWAT_DIM}}", dim("08_SWaT/SWaT_all_1s.csv"))
text = text.replace("{{WADI_DIM}}", dim("09_WADI/WADI_all_1s.csv"))

with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
    f.write(text)
print("WROTE README.md", len(text), "chars", flush=True)
print("SWaT", dim("08_SWaT/SWaT_all_1s.csv"), "WADI", dim("09_WADI/WADI_all_1s.csv"), flush=True)
