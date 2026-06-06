#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f vdb-large-miracl-prep.yaml
kubectl apply -f vdb-large-miracl-minilm-embed.yaml
kubectl apply -f vdb-large-miracl-mpnet-embed.yaml
kubectl apply -f vdb-large-dprw100-prep.yaml
kubectl apply -f vdb-large-dprw100-minilm-embed.yaml
kubectl apply -f vdb-large-dprw100-mpnet-embed.yaml
