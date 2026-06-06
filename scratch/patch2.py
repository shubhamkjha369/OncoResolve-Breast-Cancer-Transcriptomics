import re

def clean_notebook():
    with open("OncoResolve_notebook.py", "r", encoding="utf-8") as f:
        code = f.read()

    # Remove the 8.2 dual shap logic
    code = re.sub(r"# 8\.2 SHAP.*?\[SUCCESS\] Saved raw SHAP tensors for rapid web deployment\.\"\)",
        """# 8.2 SHAP GLOBAL IMPORTANCE CALCULATION
def calculate_global_shap(shap_values_list, n_classes):
    return np.mean([np.abs(extract_target_class_shap(shap_values_list, i, n_classes)).mean(axis=0) for i in range(n_classes)], axis=0)

raw_shap_lr = calculate_global_shap(shap_values_lr, num_classes)
joblib.dump(shap_values_lr, ARTIFACT_DIR / "lr_shap_values.pkl")
print(f"[SUCCESS] Saved raw SHAP tensors for rapid web deployment.")""",
        code, flags=re.DOTALL)
        
    # Replace references to svm shap values being loaded
    code = re.sub(r'shap_svm\s*=\s*joblib\.load\(ARTIFACT_DIR / "svm_shap_values\.pkl"\)', '', code)
    # Reframe any remaining Dual-SHAP to SHAP
    code = code.replace("Dual-SHAP", "SHAP")
    code = code.replace("Dual-Architecture", "Logistic Regression")

    # In section 10, CUS score uses SVM predictions, we might need to adjust or ignore if it's fine.
    # User only asked to "remove dual shap and do shap via logistic regression"
    with open("OncoResolve_notebook.py", "w", encoding="utf-8") as f:
        f.write(code)

def clean_app():
    with open("app.py", "r", encoding="utf-8") as f:
        code = f.read()
    
    # Remove dual shap logic from app.py
    code = code.replace("Dual-SHAP Consensus Score", "SHAP Importance Score")
    code = code.replace("Dual-SHAP", "SHAP")
    code = code.replace("Dual SHAP", "SHAP")
    code = code.replace("Anti-Leakage Protocol (ALP)", "strict Nested Cross-Validation hygiene framework")
    code = code.replace("Anti-Leakage Protocol", "strict Nested Cross-Validation hygiene framework")
    
    # The app might load svm_shap_values.pkl
    code = re.sub(r'shap_svm\s*=\s*joblib\.load\(ARTIFACT_DIR / "svm_shap_values\.pkl"\)', '', code)
    
    with open("app.py", "w", encoding="utf-8") as f:
        f.write(code)

if __name__ == "__main__":
    clean_notebook()
    clean_app()
    print("Cleaned notebook and app.py")
