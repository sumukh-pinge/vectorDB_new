#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f nq/vdb-ls-nq-minilm-n1024-prep.yaml
kubectl apply -f dpr-w100/vdb-ls-dpr-w100-minilm-n4096-prep.yaml
kubectl apply -f miracl-en/vdb-ls-miracl-en-minilm-n8192-prep.yaml
