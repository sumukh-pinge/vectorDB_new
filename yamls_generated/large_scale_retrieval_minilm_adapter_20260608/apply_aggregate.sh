#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f nq/vdb-ls-nq-minilm-aggregate.yaml
kubectl apply -f dpr-w100/vdb-ls-dpr-w100-minilm-aggregate.yaml
kubectl apply -f miracl-en/vdb-ls-miracl-en-minilm-aggregate.yaml
