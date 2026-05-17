from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = DATA_DIR / "artifacts"

# Model directory
MODEL_DIR = PROJECT_ROOT / "models"

# Reports
REPORT_DIR = PROJECT_ROOT / "reports"

# Create directories automatically
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ARTIFACT_DIR,
    MODEL_DIR,
    REPORT_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)