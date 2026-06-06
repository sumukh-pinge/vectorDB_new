#!/usr/bin/env python3
from pathlib import Path


NAMESPACE = "designlab"
IMAGE = "spinge/vectordb-new:20260304"
PVC = "sumukh-4tb"
MOUNT = "/mnt/work"

GIT_REPO = "https://github.com/sumukh-pinge/vectorDB_new.git"
GIT_REF = "main"

DATA_ROOT = f"{MOUNT}/VectorDB_MICRO/datasets/semantic"
INTERMEDIATE_ROOT = f"{MOUNT}/VectorDB_MICRO/intermediate_data"
LOG_ROOT = f"{MOUNT}/VectorDB_MICRO/large_text_prep_logs_20260606"

OUT_DIR = Path("yamls_generated/large_text_datasets_20260606")

DATASETS = {
    "miracl-en": {
        "short": "miracl",
        "expected_docs": 32893221,
    },
    "dpr-w100": {
        "short": "dprw100",
        "expected_docs": 21015324,
    },
}

ENCODERS = {
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "corpus_batch_size": 512,
        "query_batch_size": 512,
        "request_memory": "24Gi",
        "limit_memory": "32Gi",
    },
    "mpnet": {
        "name": "all-mpnet-base-v2",
        "corpus_batch_size": 256,
        "query_batch_size": 512,
        "request_memory": "48Gi",
        "limit_memory": "64Gi",
    },
}


def yaml_args(args):
    return "\n".join(f'          - "{arg}"' for arg in args)


def common_env():
    return f"""          - name: GIT_REPO
            value: "{GIT_REPO}"
          - name: GIT_REF
            value: "{GIT_REF}"
          - name: HF_HOME
            value: "{MOUNT}/hf_cache"
          - name: HF_DATASETS_CACHE
            value: "{MOUNT}/hf_cache/datasets"
          - name: TRANSFORMERS_CACHE
            value: "{MOUNT}/hf_cache"
          - name: IR_DATASETS_HOME
            value: "{MOUNT}/hf_cache/ir_datasets"
          - name: PIP_CACHE_DIR
            value: "{MOUNT}/hf_cache/pip"
          - name: DATA_ROOT
            value: "{DATA_ROOT}"
          - name: INTERMEDIATE_ROOT
            value: "{INTERMEDIATE_ROOT}"
          - name: PYTHONUNBUFFERED
            value: "1\""""


def affinity():
    return """      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: kubernetes.io/hostname
                    operator: NotIn
                    values:
                      - k8s-gpu-2.ucr.edu
"""


def job_doc(name, labels, args, gpu, cpu, request_memory, limit_memory):
    label_lines = "\n".join(f'    {key}: "{value}"' for key, value in labels.items())
    pod_label_lines = "\n".join(f'        {key}: "{value}"' for key, value in labels.items())
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {NAMESPACE}
  labels:
{label_lines}
spec:
  backoffLimit: 1
  template:
    metadata:
      labels:
{pod_label_lines}
    spec:
      restartPolicy: Never
{affinity()}      containers:
      - name: run
        image: {IMAGE}
        imagePullPolicy: IfNotPresent
        command: ["/usr/local/bin/nrp-launch"]
        args:
{yaml_args(args)}
        env:
{common_env()}
        resources:
          requests:
            nvidia.com/gpu: {gpu}
            cpu: "{cpu}"
            memory: "{request_memory}"
          limits:
            nvidia.com/gpu: {gpu}
            cpu: "{cpu}"
            memory: "{limit_memory}"
        volumeMounts:
          - name: work
            mountPath: {MOUNT}
      volumes:
        - name: work
          persistentVolumeClaim:
            claimName: {PVC}
"""


def shell_args(job_name, inner_command):
    log_file = f"{LOG_ROOT}/{job_name}.log"
    command = (
        "set -euo pipefail; "
        f"mkdir -p {LOG_ROOT}; "
        f"echo '[job] {job_name}' | tee -a {log_file}; "
        f"echo '[start]' $(date -u +%Y-%m-%dT%H:%M:%SZ) | tee -a {log_file}; "
        f"({inner_command}) 2>&1 | tee -a {log_file}; "
        f"echo '[done]' $(date -u +%Y-%m-%dT%H:%M:%SZ) | tee -a {log_file}"
    )
    return ["bash", "-lc", command]


def prep_job(dataset, cfg):
    name = f"vdb-large-{cfg['short']}-prep"
    labels = {
        "app": "vectordb-large-text-prep",
        "dataset": dataset,
        "stage": "prep",
        "encoder": "none",
    }
    inner = f"python prepare_large_text_dataset.py --dataset {dataset} --data_root {DATA_ROOT}"
    return name, job_doc(name, labels, shell_args(name, inner), 0, 4, "12Gi", "16Gi")


def embed_job(dataset, ds_cfg, enc_short, enc_cfg):
    name = f"vdb-large-{ds_cfg['short']}-{enc_short}-embed"
    labels = {
        "app": "vectordb-large-text-prep",
        "dataset": dataset,
        "stage": "embed",
        "encoder": enc_short,
    }
    inner = (
        f"python embed_large_text_dataset.py "
        f"--dataset {dataset} "
        f"--encoder {enc_cfg['name']} "
        f"--data_root {DATA_ROOT} "
        f"--intermediate_root {INTERMEDIATE_ROOT} "
        f"--expected_docs {ds_cfg['expected_docs']} "
        f"--corpus_batch_size {enc_cfg['corpus_batch_size']} "
        f"--query_batch_size {enc_cfg['query_batch_size']} "
        f"--query_splits dev,train "
        f"--wait_timeout_sec 172800 "
        f"--wait_poll_sec 60"
    )
    return name, job_doc(
        name,
        labels,
        shell_args(name, inner),
        1,
        4,
        enc_cfg["request_memory"],
        enc_cfg["limit_memory"],
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_prep = ["#!/usr/bin/env bash", "set -euo pipefail"]
    apply_embed = ["#!/usr/bin/env bash", "set -euo pipefail"]
    apply_all = ["#!/usr/bin/env bash", "set -euo pipefail"]
    delete_all = ["#!/usr/bin/env bash", "set -euo pipefail"]
    docs = []

    for dataset, ds_cfg in DATASETS.items():
        name, doc = prep_job(dataset, ds_cfg)
        path = OUT_DIR / f"{name}.yaml"
        path.write_text(doc)
        docs.append(doc)
        apply_prep.append(f"kubectl apply -f {path.name}")
        apply_all.append(f"kubectl apply -f {path.name}")
        delete_all.append(f"kubectl delete -f {path.name} || true")

        for enc_short, enc_cfg in ENCODERS.items():
            name, doc = embed_job(dataset, ds_cfg, enc_short, enc_cfg)
            path = OUT_DIR / f"{name}.yaml"
            path.write_text(doc)
            docs.append(doc)
            apply_embed.append(f"kubectl apply -f {path.name}")
            apply_all.append(f"kubectl apply -f {path.name}")
            delete_all.append(f"kubectl delete -f {path.name} || true")

    (OUT_DIR / "ALL.yaml").write_text("---\n".join(docs))
    for script_name, lines in [
        ("apply_prep.sh", apply_prep),
        ("apply_embed.sh", apply_embed),
        ("apply_all.sh", apply_all),
        ("delete_all.sh", delete_all),
    ]:
        path = OUT_DIR / script_name
        path.write_text("\n".join(lines) + "\n")
        path.chmod(0o755)

    (OUT_DIR / "README.txt").write_text(
        f"""Large text dataset prep jobs.

Namespace: {NAMESPACE}
PVC: {PVC}
Data root: {DATA_ROOT}
Intermediate root: {INTERMEDIATE_ROOT}
Logs: {LOG_ROOT}

Deploy:
  cd {OUT_DIR}
  ./apply_all.sh

Monitor:
  kubectl get jobs -n {NAMESPACE} -l app=vectordb-large-text-prep
  kubectl get pods -n {NAMESPACE} -l app=vectordb-large-text-prep
  kubectl logs -n {NAMESPACE} job/vdb-large-miracl-prep --tail=100
"""
    )
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
