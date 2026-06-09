#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f nq/vdb-hn-nq-minilm-merge.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-train.yaml
