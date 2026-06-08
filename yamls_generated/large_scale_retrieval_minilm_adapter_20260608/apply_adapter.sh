#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f nq/vdb-ls-nq-minilm-adapter.yaml
kubectl apply -f dpr-w100/vdb-ls-dpr-w100-minilm-adapter.yaml
kubectl apply -f miracl-en/vdb-ls-miracl-en-minilm-adapter.yaml
