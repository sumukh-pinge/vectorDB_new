# CUDA-ready, conda+mamba base
FROM condaforge/mambaforge:23.11.0-0

# Faster builds + cleaner logs
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache

WORKDIR /workspace

# Create the env exactly as your local one (name = vector_db)
COPY environment.yml .
RUN mamba env create -f environment.yml && \
    conda clean -afy

# Make that env the default for all subsequent RUN/CMD
SHELL ["bash", "-lc"]
ENV CONDA_DEFAULT_ENV=vector_db
ENV PATH=/opt/conda/envs/vector_db/bin:$PATH

# Optional: create model cache dir
RUN mkdir -p /opt/hf_cache && chmod -R 777 /opt/hf_cache

# Copy your app code
COPY app/ ./app/

# Default working dir inside the container
WORKDIR /workspace/app

# No ENTRYPOINT so Jobs can pass any command; the env is already on PATH.
# To sanity-check at build time you can uncomment:
RUN python -c "import torch, faiss, transformers; print('Torch:', torch.__version__); print('FAISS:', faiss.__version__)"
