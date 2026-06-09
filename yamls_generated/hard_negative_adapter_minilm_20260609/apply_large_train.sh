#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f dpr-w100/vdb-hn-dpr-w100-minilm-merge.yaml
kubectl apply -f dpr-w100/vdb-hn-dpr-w100-minilm-train.yaml
kubectl apply -f miracl-en/vdb-hn-miracl-en-minilm-merge.yaml
kubectl apply -f miracl-en/vdb-hn-miracl-en-minilm-train.yaml
