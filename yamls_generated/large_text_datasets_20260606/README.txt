Large text dataset prep jobs.

Namespace: designlab
PVC: sumukh-4tb
Data root: /mnt/work/VectorDB_MICRO/datasets/semantic
Intermediate root: /mnt/work/VectorDB_MICRO/intermediate_data
Logs: /mnt/work/VectorDB_MICRO/large_text_prep_logs_20260606

Deploy:
  cd yamls_generated/large_text_datasets_20260606
  ./apply_all.sh

Monitor:
  kubectl get jobs -n designlab -l app=vectordb-large-text-prep
  kubectl get pods -n designlab -l app=vectordb-large-text-prep
  kubectl logs -n designlab job/vdb-large-miracl-prep --tail=100
