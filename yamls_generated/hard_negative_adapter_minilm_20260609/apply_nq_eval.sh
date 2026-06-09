#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f nq/vdb-hn-nq-minilm-e1-np64-ivf.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e1-np64-dts111.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e1-np64-dts242.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e1-np128-ivf.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e1-np128-dts111.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e1-np128-dts242.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e2-np64-ivf.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e2-np64-dts111.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e2-np64-dts242.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e2-np128-ivf.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e2-np128-dts111.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e2-np128-dts242.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e3-np64-ivf.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e3-np64-dts111.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e3-np64-dts242.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e3-np128-ivf.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e3-np128-dts111.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e3-np128-dts242.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e5-np64-ivf.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e5-np64-dts111.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e5-np64-dts242.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e5-np128-ivf.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e5-np128-dts111.yaml
kubectl apply -f nq/vdb-hn-nq-minilm-e5-np128-dts242.yaml
