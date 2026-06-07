#!/usr/bin/env python3
from pathlib import Path
import os


NAMESPACE = "designlab"
IMAGE = "spinge/vectordb-new:20260304"
PVC = "sumukh-4tb"
MOUNT = "/mnt/work"
GIT_REPO = "https://github.com/sumukh-pinge/vectorDB_new.git"
GIT_REF = "main"

ENCODER = "all-MiniLM-L6-v2"
ENCODER_SHORT = "minilm"
ENCODER_TAG = ENCODER.replace("/", "_")
RUN_TAG = "large_scale_minilm_20260607"
DATA_ROOT = f"{MOUNT}/VectorDB_MICRO/datasets/semantic"
INTERMEDIATE_ROOT = f"{MOUNT}/VectorDB_MICRO/intermediate_data"
INDEX_ROOT = f"{MOUNT}/VectorDB_MICRO/large_scale_indices/{RUN_TAG}"
RUN_ROOT = f"{MOUNT}/VectorDB_MICRO/large_scale_runs/{RUN_TAG}"

DATASETS = {
    "dpr-w100": {
        "nlist": 4096,
        "nprobes": [128, 256, 512],
        "index_mem": ("384Gi", "448Gi"),
        "eval_mem": ("384Gi", "448Gi"),
        "agg_mem": ("8Gi", "16Gi"),
    },
    "miracl-en": {
        "nlist": 8192,
        "nprobes": [128, 256, 512, 1024],
        "index_mem": ("512Gi", "640Gi"),
        "eval_mem": ("512Gi", "640Gi"),
        "agg_mem": ("8Gi", "16Gi"),
    },
}
MODES = ["ivf", "dts111", "dts242"]


def safe(name: str) -> str:
    return name.lower().replace("_", "-").replace("/", "-")


def env_yaml():
    pairs = {
        "GIT_REPO": GIT_REPO,
        "GIT_REF": GIT_REF,
        "HF_HOME": f"{MOUNT}/hf_cache",
        "TRANSFORMERS_CACHE": f"{MOUNT}/hf_cache",
        "DATA_ROOT": DATA_ROOT,
        "INTERMEDIATE_ROOT": INTERMEDIATE_ROOT,
        "INDEX_ROOT": INDEX_ROOT,
        "RUN_ROOT": RUN_ROOT,
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    return "\n".join([f'          - name: {k}\n            value: "{v}"' for k, v in pairs.items()])


def command_args(args):
    return "\n".join([f'          - "{a}"' for a in args])


def wait_init(name: str, path: str):
    return f"""
      initContainers:
      - name: wait-for-{name}
        image: busybox:1.36
        command: ["sh", "-c", "until [ -f {path} ]; do echo waiting for {path}; sleep 60; done"]
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "500m"
            memory: "512Mi"
        volumeMounts:
          - name: work
            mountPath: {MOUNT}
"""


def job_doc(name, labels, args, request_memory, limit_memory, cpu="4", init_path=None, ttl=86400):
    init = wait_init("marker", init_path) if init_path else ""
    label_lines = "\n".join([f'    {k}: "{v}"' for k, v in labels.items()])
    pod_label_lines = "\n".join([f'        {k}: "{v}"' for k, v in labels.items()])
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {NAMESPACE}
  labels:
{label_lines}
spec:
  backoffLimit: 3
  ttlSecondsAfterFinished: {ttl}
  template:
    metadata:
      labels:
{pod_label_lines}
    spec:
      restartPolicy: Never
{init}      containers:
      - name: run
        image: {IMAGE}
        imagePullPolicy: IfNotPresent
        command: ["/usr/local/bin/nrp-launch"]
        args:
{command_args(args)}
        env:
{env_yaml()}
        resources:
          requests:
            nvidia.com/gpu: 0
            cpu: "{cpu}"
            memory: "{request_memory}"
          limits:
            nvidia.com/gpu: 0
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


def index_marker(dataset, nlist):
    return f"{INDEX_ROOT}/{dataset}/{ENCODER_TAG}/nlist{nlist}/INDEX_READY.json"


def summary_marker(dataset, nlist, nprobe, mode):
    return f"{RUN_ROOT}/{dataset}/{ENCODER_TAG}/nlist{nlist}/np{nprobe}/{mode}/summary.json"


def main():
    outroot = Path("yamls_generated/large_scale_retrieval_minilm_20260607")
    outroot.mkdir(parents=True, exist_ok=True)
    docs = []
    apply_index = ["#!/usr/bin/env bash", "set -euo pipefail"]
    apply_eval = ["#!/usr/bin/env bash", "set -euo pipefail"]
    apply_agg = ["#!/usr/bin/env bash", "set -euo pipefail"]
    delete_all = ["#!/usr/bin/env bash", "set -euo pipefail"]

    for dataset, cfg in DATASETS.items():
        ds_dir = outroot / safe(dataset)
        ds_dir.mkdir(parents=True, exist_ok=True)
        nlist = cfg["nlist"]

        idx_name = f"vdb-ls-{safe(dataset)}-{ENCODER_SHORT}-n{nlist}-index"[:63]
        idx_args = [
            "python", "run_large_scale_retrieval.py",
            "--stage", "build-index",
            "--dataset", dataset,
            "--encoder", ENCODER,
            "--nlist", str(nlist),
            "--faiss_threads", "4",
        ]
        doc = job_doc(
            idx_name,
            {"app": "vectordb-large-scale", "dataset": dataset, "encoder": ENCODER_SHORT, "stage": "index"},
            idx_args,
            *cfg["index_mem"],
        )
        path = ds_dir / f"{idx_name}.yaml"
        path.write_text(doc)
        docs.append(doc)
        apply_index.append(f"kubectl apply -f {safe(dataset)}/{path.name}")
        delete_all.append(f"kubectl delete -f {safe(dataset)}/{path.name} || true")

        for nprobe in cfg["nprobes"]:
            for mode in MODES:
                name = f"vdb-ls-{safe(dataset)}-{ENCODER_SHORT}-np{nprobe}-{mode}"[:63]
                args = [
                    "python", "run_large_scale_retrieval.py",
                    "--stage", "eval",
                    "--dataset", dataset,
                    "--encoder", ENCODER,
                    "--mode", mode,
                    "--nlist", str(nlist),
                    "--nprobe", str(nprobe),
                    "--k2", "1000",
                    "--kfinal", "100",
                    "--faiss_threads", "4",
                    "--score_chunk_size", "250000",
                ]
                doc = job_doc(
                    name,
                    {
                        "app": "vectordb-large-scale",
                        "dataset": dataset,
                        "encoder": ENCODER_SHORT,
                        "stage": "eval",
                        "mode": mode,
                        "nprobe": f"np{nprobe}",
                    },
                    args,
                    *cfg["eval_mem"],
                    init_path=index_marker(dataset, nlist),
                )
                path = ds_dir / f"{name}.yaml"
                path.write_text(doc)
                docs.append(doc)
                apply_eval.append(f"kubectl apply -f {safe(dataset)}/{path.name}")
                delete_all.append(f"kubectl delete -f {safe(dataset)}/{path.name} || true")

        agg_name = f"vdb-ls-{safe(dataset)}-{ENCODER_SHORT}-aggregate"[:63]
        wait_path = summary_marker(dataset, nlist, cfg["nprobes"][-1], "dts242")
        agg_args = [
            "python", "aggregate_large_scale_retrieval.py",
            "--dataset", dataset,
            "--encoder", ENCODER,
            "--nlist", str(nlist),
            "--nprobes", ",".join(str(x) for x in cfg["nprobes"]),
            "--modes", ",".join(MODES),
        ]
        doc = job_doc(
            agg_name,
            {"app": "vectordb-large-scale", "dataset": dataset, "encoder": ENCODER_SHORT, "stage": "aggregate"},
            agg_args,
            *cfg["agg_mem"],
            cpu="2",
            init_path=wait_path,
        )
        path = ds_dir / f"{agg_name}.yaml"
        path.write_text(doc)
        docs.append(doc)
        apply_agg.append(f"kubectl apply -f {safe(dataset)}/{path.name}")
        delete_all.append(f"kubectl delete -f {safe(dataset)}/{path.name} || true")

    (outroot / "ALL.yaml").write_text("---\n".join(docs))
    for name, lines in {
        "apply_index.sh": apply_index,
        "apply_eval.sh": apply_eval,
        "apply_aggregate.sh": apply_agg,
        "delete_all.sh": delete_all,
    }.items():
        path = outroot / name
        path.write_text("\n".join(lines) + "\n")
        os.chmod(path, 0o755)
    (outroot / "README.txt").write_text(
        "MiniLM large-scale retrieval jobs. Run apply_index.sh first; eval and aggregate jobs can be applied immediately after and will wait on markers.\n"
    )
    print(f"wrote {len(docs)} yamls to {outroot.resolve()}")


if __name__ == "__main__":
    main()
