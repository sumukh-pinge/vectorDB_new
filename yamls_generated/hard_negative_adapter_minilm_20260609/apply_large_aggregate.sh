#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f dpr-w100/vdb-hn-dpr-w100-minilm-aggregate.yaml
kubectl apply -f miracl-en/vdb-hn-miracl-en-minilm-aggregate.yaml
