#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f dpr-w100/vdb-ls-dpr-w100-minilm-n4096-index.yaml
kubectl apply -f miracl-en/vdb-ls-miracl-en-minilm-n8192-index.yaml
