with open("README.md", "r", encoding="utf-8") as f:
    text = f.read()

# Replace lingering occurrences
text = text.replace("257 biomarkers", "267 biomarkers")
text = text.replace("54,677", "54,613")
text = text.replace("257 genes", "267 genes")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(text)

print(" Lingering counts updated!")
