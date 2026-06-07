#!/usr/bin/env bash
set -euo pipefail
kubectl delete -f dpr-w100/vdb-ls-dpr-w100-minilm-n4096-index.yaml || true
kubectl delete -f dpr-w100/vdb-ls-dpr-w100-minilm-np128-ivf.yaml || true
kubectl delete -f dpr-w100/vdb-ls-dpr-w100-minilm-np128-dts111.yaml || true
kubectl delete -f dpr-w100/vdb-ls-dpr-w100-minilm-np128-dts242.yaml || true
kubectl delete -f dpr-w100/vdb-ls-dpr-w100-minilm-np256-ivf.yaml || true
kubectl delete -f dpr-w100/vdb-ls-dpr-w100-minilm-np256-dts111.yaml || true
kubectl delete -f dpr-w100/vdb-ls-dpr-w100-minilm-np256-dts242.yaml || true
kubectl delete -f dpr-w100/vdb-ls-dpr-w100-minilm-np512-ivf.yaml || true
kubectl delete -f dpr-w100/vdb-ls-dpr-w100-minilm-np512-dts111.yaml || true
kubectl delete -f dpr-w100/vdb-ls-dpr-w100-minilm-np512-dts242.yaml || true
kubectl delete -f dpr-w100/vdb-ls-dpr-w100-minilm-aggregate.yaml || true
kubectl delete -f miracl-en/vdb-ls-miracl-en-minilm-n8192-index.yaml || true
kubectl delete -f miracl-en/vdb-ls-miracl-en-minilm-np128-ivf.yaml || true
kubectl delete -f miracl-en/vdb-ls-miracl-en-minilm-np128-dts111.yaml || true
kubectl delete -f miracl-en/vdb-ls-miracl-en-minilm-np128-dts242.yaml || true
kubectl delete -f miracl-en/vdb-ls-miracl-en-minilm-np256-ivf.yaml || true
kubectl delete -f miracl-en/vdb-ls-miracl-en-minilm-np256-dts111.yaml || true
kubectl delete -f miracl-en/vdb-ls-miracl-en-minilm-np256-dts242.yaml || true
kubectl delete -f miracl-en/vdb-ls-miracl-en-minilm-np512-ivf.yaml || true
kubectl delete -f miracl-en/vdb-ls-miracl-en-minilm-np512-dts111.yaml || true
kubectl delete -f miracl-en/vdb-ls-miracl-en-minilm-np512-dts242.yaml || true
kubectl delete -f miracl-en/vdb-ls-miracl-en-minilm-np1024-ivf.yaml || true
kubectl delete -f miracl-en/vdb-ls-miracl-en-minilm-np1024-dts111.yaml || true
kubectl delete -f miracl-en/vdb-ls-miracl-en-minilm-np1024-dts242.yaml || true
kubectl delete -f miracl-en/vdb-ls-miracl-en-minilm-aggregate.yaml || true
