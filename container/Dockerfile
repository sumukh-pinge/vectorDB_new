FROM condaforge/mambaforge:23.11.0-0

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache

WORKDIR /workspace

COPY environment.yml /workspace/environment.yml
RUN mamba env create -f /workspace/environment.yml && conda clean -afy

SHELL ["bash", "-lc"]
ENV CONDA_DEFAULT_ENV=vector_db
ENV PATH=/opt/conda/envs/vector_db/bin:$PATH

RUN mkdir -p /opt/hf_cache && chmod -R 777 /opt/hf_cache

COPY entrypoint.sh /usr/local/bin/nrp-launch
RUN chmod +x /usr/local/bin/nrp-launch

WORKDIR /workspace/src
CMD ["/usr/local/bin/nrp-launch", "bash", "-lc", "python app/run_full_suite.py --help"]
