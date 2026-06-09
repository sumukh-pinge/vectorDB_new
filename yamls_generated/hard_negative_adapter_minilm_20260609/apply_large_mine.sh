#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f dpr-w100/vdb-hn-dpr-w100-minilm-mine-s0.yaml
kubectl apply -f dpr-w100/vdb-hn-dpr-w100-minilm-mine-s1.yaml
kubectl apply -f dpr-w100/vdb-hn-dpr-w100-minilm-mine-s2.yaml
kubectl apply -f dpr-w100/vdb-hn-dpr-w100-minilm-mine-s3.yaml
kubectl apply -f dpr-w100/vdb-hn-dpr-w100-minilm-mine-s4.yaml
kubectl apply -f dpr-w100/vdb-hn-dpr-w100-minilm-mine-s5.yaml
kubectl apply -f dpr-w100/vdb-hn-dpr-w100-minilm-mine-s6.yaml
kubectl apply -f dpr-w100/vdb-hn-dpr-w100-minilm-mine-s7.yaml
kubectl apply -f miracl-en/vdb-hn-miracl-en-minilm-mine-s0.yaml
kubectl apply -f miracl-en/vdb-hn-miracl-en-minilm-mine-s1.yaml
kubectl apply -f miracl-en/vdb-hn-miracl-en-minilm-mine-s2.yaml
kubectl apply -f miracl-en/vdb-hn-miracl-en-minilm-mine-s3.yaml
kubectl apply -f miracl-en/vdb-hn-miracl-en-minilm-mine-s4.yaml
kubectl apply -f miracl-en/vdb-hn-miracl-en-minilm-mine-s5.yaml
kubectl apply -f miracl-en/vdb-hn-miracl-en-minilm-mine-s6.yaml
kubectl apply -f miracl-en/vdb-hn-miracl-en-minilm-mine-s7.yaml
