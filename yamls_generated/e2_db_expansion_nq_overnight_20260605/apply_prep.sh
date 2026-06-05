#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f minilm/vdb-e2-minilm-nq-np64-s42-prep.yaml
kubectl apply -f minilm/vdb-e2-minilm-nq-np128-s42-prep.yaml
kubectl apply -f minilm/vdb-e2-minilm-nq-np64-s43-prep.yaml
kubectl apply -f minilm/vdb-e2-minilm-nq-np128-s43-prep.yaml
kubectl apply -f mpnet/vdb-e2-mpnet-nq-np64-s42-prep.yaml
kubectl apply -f mpnet/vdb-e2-mpnet-nq-np128-s42-prep.yaml
kubectl apply -f mpnet/vdb-e2-mpnet-nq-np64-s43-prep.yaml
kubectl apply -f mpnet/vdb-e2-mpnet-nq-np128-s43-prep.yaml
