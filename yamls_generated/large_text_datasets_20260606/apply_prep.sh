#!/usr/bin/env bash
set -euo pipefail
kubectl apply -f vdb-large-miracl-prep.yaml
kubectl apply -f vdb-large-dprw100-prep.yaml
