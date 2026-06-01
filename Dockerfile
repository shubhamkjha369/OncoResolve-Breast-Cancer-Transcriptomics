# ==============================================================================
# OncoResolve v2.0.1 — Multi-stage Docker Build
# Primary dataset: TCGA-BRCA Pan-Can Atlas 2018 (RNA-seq)
# External cohorts: METABRIC, SCAN-B (mount at runtime — too large to bundle)
# ==============================================================================

# ==============================================================================
# STAGE 1: Build — install all Python dependencies
# ==============================================================================
FROM python:3.13-slim AS builder

WORKDIR /workspace

# System build deps (required by lightgbm, umap-learn, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ==============================================================================
# STAGE 2: Production Runtime
# ==============================================================================
FROM python:3.13-slim AS runner

LABEL org.opencontainers.image.title="OncoResolve" \
    org.opencontainers.image.description="Breast Cancer Transcriptomics ML Pipeline — TCGA-BRCA RNA-seq" \
    org.opencontainers.image.version="2.0.1" \
    org.opencontainers.image.source="https://github.com/shubhamkjha369/OncoResolve-Breast-Cancer-Transcriptomics"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Streamlit config
COPY .streamlit/ ./.streamlit/

# Source module
COPY src/ ./src/

# Core application files
COPY app.py automl_page.py pipeline_engine.py ./

# Pre-computed ML artifacts (models, SHAP values, parquets, PNGs)
COPY data/artifacts/ ./data/artifacts/

# Pre-trained model checkpoints
COPY models/ ./models/

# External cohort download script (data files are mounted at runtime)
COPY data/external_cohort/download_external_cohorts.py ./data/external_cohort/

# NOTE: Large data files are NOT bundled in the image.
# Mount them at runtime with:
#   -v /path/to/data/raw:/app/data/raw
#   -v /path/to/data/external_cohort:/app/data/external_cohort
# The app gracefully handles missing data files with informative warnings.

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "app.py"]