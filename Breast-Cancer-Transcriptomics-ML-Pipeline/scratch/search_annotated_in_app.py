with open("app.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if "annotated_global_biomarkers" in line or "annotated_biomarkers" in line:
        print(f"Line {idx+1}: {line.strip()}")
