#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f dpr-w100/vdb-ls-dpr-w100-minilm-np128-ivf.yaml
kubectl apply -f dpr-w100/vdb-ls-dpr-w100-minilm-np128-dts111.yaml
kubectl apply -f dpr-w100/vdb-ls-dpr-w100-minilm-np128-dts242.yaml
kubectl apply -f dpr-w100/vdb-ls-dpr-w100-minilm-np256-ivf.yaml
kubectl apply -f dpr-w100/vdb-ls-dpr-w100-minilm-np256-dts111.yaml
kubectl apply -f dpr-w100/vdb-ls-dpr-w100-minilm-np256-dts242.yaml
kubectl apply -f dpr-w100/vdb-ls-dpr-w100-minilm-np512-ivf.yaml
kubectl apply -f dpr-w100/vdb-ls-dpr-w100-minilm-np512-dts111.yaml
kubectl apply -f dpr-w100/vdb-ls-dpr-w100-minilm-np512-dts242.yaml
kubectl apply -f miracl-en/vdb-ls-miracl-en-minilm-np128-ivf.yaml
kubectl apply -f miracl-en/vdb-ls-miracl-en-minilm-np128-dts111.yaml
kubectl apply -f miracl-en/vdb-ls-miracl-en-minilm-np128-dts242.yaml
kubectl apply -f miracl-en/vdb-ls-miracl-en-minilm-np256-ivf.yaml
kubectl apply -f miracl-en/vdb-ls-miracl-en-minilm-np256-dts111.yaml
kubectl apply -f miracl-en/vdb-ls-miracl-en-minilm-np256-dts242.yaml
kubectl apply -f miracl-en/vdb-ls-miracl-en-minilm-np512-ivf.yaml
kubectl apply -f miracl-en/vdb-ls-miracl-en-minilm-np512-dts111.yaml
kubectl apply -f miracl-en/vdb-ls-miracl-en-minilm-np512-dts242.yaml
kubectl apply -f miracl-en/vdb-ls-miracl-en-minilm-np1024-ivf.yaml
kubectl apply -f miracl-en/vdb-ls-miracl-en-minilm-np1024-dts111.yaml
kubectl apply -f miracl-en/vdb-ls-miracl-en-minilm-np1024-dts242.yaml
