# ==============================================================================
# STAGE 1: Compilation Build Stage
# ==============================================================================
FROM python:3.13-slim AS builder

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies to local location context
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ==============================================================================
# STAGE 2: Optimized Production Runtime
# ==============================================================================
FROM python:3.13-slim AS runner

WORKDIR /app

# Copy cached package environment over from builder stage
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy configuration and all core application files
COPY .streamlit/ ./.streamlit/
COPY data/processed/ ./data/processed/
COPY data/artifacts/ ./data/artifacts/
COPY src/ ./src/
COPY app.py automl_page.py pipeline_engine.py ./

# Expose Streamlit defaults
EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Start server
CMD ["streamlit", "run", "app.py"]