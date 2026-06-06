#!/usr/bin/env bash
set -euo pipefail
kubectl delete -f vdb-large-miracl-prep.yaml || true
kubectl delete -f vdb-large-miracl-minilm-embed.yaml || true
kubectl delete -f vdb-large-miracl-mpnet-embed.yaml || true
kubectl delete -f vdb-large-dprw100-prep.yaml || true
kubectl delete -f vdb-large-dprw100-minilm-embed.yaml || true
kubectl delete -f vdb-large-dprw100-mpnet-embed.yaml || true
