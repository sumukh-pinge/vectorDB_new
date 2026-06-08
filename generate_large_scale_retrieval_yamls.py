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
RUN_TAG = "large_scale_minilm_adapter_20260608"
DATA_ROOT = f"{MOUNT}/VectorDB_MICRO/datasets/semantic"
INTERMEDIATE_ROOT = f"{MOUNT}/VectorDB_MICRO/intermediate_data"
INDEX_ROOT = f"{MOUNT}/VectorDB_MICRO/large_scale_indices/{RUN_TAG}"
RUN_ROOT = f"{MOUNT}/VectorDB_MICRO/large_scale_runs/{RUN_TAG}"

DATASETS = {
    "nq": {
        "nlist": 1024,
        "nprobes": [64, 128],
        "qrels_split": "test",
        "adapter_train_split": "test",
        "adapter_source_slug": "adapter_nl1024_b4_nq_all-MiniLM-L6-v2",
        "index_mem": ("56Gi", "64Gi"),
        "eval_mem": ("56Gi", "64Gi"),
        "adapter_mem": ("16Gi", "32Gi"),
        "agg_mem": ("4Gi", "8Gi"),
    },
    "dpr-w100": {
        "nlist": 4096,
        "nprobes": [128, 256, 512],
        "qrels_split": "dev",
        "adapter_train_split": "train",
        "adapter_source_slug": "",
        "index_mem": ("384Gi", "448Gi"),
        "eval_mem": ("384Gi", "448Gi"),
        "adapter_mem": ("32Gi", "64Gi"),
        "agg_mem": ("8Gi", "16Gi"),
    },
    "miracl-en": {
        "nlist": 8192,
        "nprobes": [128, 256, 512, 1024],
        "qrels_split": "dev",
        "adapter_train_split": "train",
        "adapter_source_slug": "",
        "index_mem": ("512Gi", "640Gi"),
        "eval_mem": ("512Gi", "640Gi"),
        "adapter_mem": ("32Gi", "64Gi"),
        "agg_mem": ("8Gi", "16Gi"),
    },
}
MODES = ["ivf", "dts111", "dts242"]
ADAPTERS = ["off", "on"]


def safe(name: str) -> str:
    return name.lower().replace("_", "-").replace("/", "-")


def adapter_slug(dataset: str, nlist: int) -> str:
    return f"large_scale_adapter_{dataset}_{ENCODER_TAG}_nlist{nlist}"


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
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    return "\n".join([f'          - name: {k}\n            value: "{v}"' for k, v in pairs.items()])


def command_args(args):
    return "\n".join([f'          - "{a}"' for a in args])


def wait_init(paths):
    if not paths:
        return ""
    checks = " && ".join([f"[ -f {path} ]" for path in paths])
    names = ",".join(paths)
    return f"""
      initContainers:
      - name: wait-for-markers
        image: busybox:1.36
        command: ["sh", "-c", "until {checks}; do echo waiting for {names}; sleep 60; done"]
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


def job_doc(name, labels, args, request_memory, limit_memory, cpu="1", gpu="0", init_paths=None, ttl=86400):
    init = wait_init(init_paths or [])
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


def index_marker(dataset, nlist):
    return f"{INDEX_ROOT}/{dataset}/{ENCODER_TAG}/nlist{nlist}/INDEX_READY.json"


def adapter_marker(dataset, nlist):
    slug = adapter_slug(dataset, nlist)
    return f"{INDEX_ROOT}/{dataset}/{ENCODER_TAG}/nlist{nlist}/ADAPTER_READY_{slug}.json"


def summary_marker(dataset, nlist, nprobe, mode, adapter):
    return f"{RUN_ROOT}/{dataset}/{ENCODER_TAG}/nlist{nlist}/np{nprobe}/{mode}/adapter_{adapter}/summary.json"


def main():
    outroot = Path("yamls_generated/large_scale_retrieval_minilm_adapter_20260608")
    outroot.mkdir(parents=True, exist_ok=True)
    docs = []
    scripts = {
        "apply_index.sh": ["#!/usr/bin/env bash", "set -euo pipefail"],
        "apply_adapter.sh": ["#!/usr/bin/env bash", "set -euo pipefail"],
        "apply_eval_no_adapter.sh": ["#!/usr/bin/env bash", "set -euo pipefail"],
        "apply_eval_adapter.sh": ["#!/usr/bin/env bash", "set -euo pipefail"],
        "apply_eval.sh": ["#!/usr/bin/env bash", "set -euo pipefail"],
        "apply_aggregate.sh": ["#!/usr/bin/env bash", "set -euo pipefail"],
        "delete_all.sh": ["#!/usr/bin/env bash", "set -euo pipefail"],
    }

    for dataset, cfg in DATASETS.items():
        ds_safe = safe(dataset)
        ds_dir = outroot / ds_safe
        ds_dir.mkdir(parents=True, exist_ok=True)
        nlist = cfg["nlist"]
        slug = adapter_slug(dataset, nlist)

        idx_name = f"vdb-ls-{ds_safe}-{ENCODER_SHORT}-n{nlist}-index"[:63]
        idx_args = [
            "python", "run_large_scale_retrieval.py",
            "--stage", "build-index",
            "--dataset", dataset,
            "--encoder", ENCODER,
            "--qrels_split", cfg["qrels_split"],
            "--nlist", str(nlist),
            "--faiss_threads", "4",
            "--train_size", "400000",
            "--train_blocks", "128",
        ]
        idx_doc = job_doc(
            idx_name,
            {"app": "vectordb-large-scale", "dataset": dataset, "encoder": ENCODER_SHORT, "stage": "index"},
            idx_args,
            *cfg["index_mem"],
            cpu="4",
        )
        idx_path = ds_dir / f"{idx_name}.yaml"
        idx_path.write_text(idx_doc)
        docs.append(idx_doc)
        scripts["apply_index.sh"].append(f"kubectl apply -f {ds_safe}/{idx_path.name}")
        scripts["delete_all.sh"].append(f"kubectl delete -f {ds_safe}/{idx_path.name} || true")

        adapt_name = f"vdb-ls-{ds_safe}-{ENCODER_SHORT}-adapter"[:63]
        adapt_args = [
            "python", "run_large_scale_retrieval.py",
            "--stage", "train-adapter",
            "--dataset", dataset,
            "--encoder", ENCODER,
            "--qrels_split", cfg["qrels_split"],
            "--adapter_train_split", cfg["adapter_train_split"],
            "--nlist", str(nlist),
            "--adapter_slug", slug,
            "--adapter_epochs", "1",
            "--adapter_max_pairs", "100000",
            "--adapter_negatives", "4096",
            "--adapter_qbatch", "256",
            "--adapter_device", "cpu",
        ]
        if cfg["adapter_source_slug"]:
            adapt_args.extend(["--adapter_source_slug", cfg["adapter_source_slug"]])
        adapt_doc = job_doc(
            adapt_name,
            {"app": "vectordb-large-scale", "dataset": dataset, "encoder": ENCODER_SHORT, "stage": "adapter"},
            adapt_args,
            *cfg["adapter_mem"],
            cpu="2",
            init_paths=[index_marker(dataset, nlist)] if not cfg["adapter_source_slug"] else [],
        )
        adapt_path = ds_dir / f"{adapt_name}.yaml"
        adapt_path.write_text(adapt_doc)
        docs.append(adapt_doc)
        scripts["apply_adapter.sh"].append(f"kubectl apply -f {ds_safe}/{adapt_path.name}")
        scripts["delete_all.sh"].append(f"kubectl delete -f {ds_safe}/{adapt_path.name} || true")

        for nprobe in cfg["nprobes"]:
            for mode in MODES:
                for adapter in ADAPTERS:
                    name = f"vdb-ls-{ds_safe}-{ENCODER_SHORT}-np{nprobe}-{mode}-a{adapter}"[:63]
                    args = [
                        "python", "run_large_scale_retrieval.py",
                        "--stage", "eval",
                        "--dataset", dataset,
                        "--encoder", ENCODER,
                        "--qrels_split", cfg["qrels_split"],
                        "--mode", mode,
                        "--adapter", adapter,
                        "--adapter_slug", slug,
                        "--nlist", str(nlist),
                        "--nprobe", str(nprobe),
                        "--k2", "1000",
                        "--kfinal", "100",
                        "--faiss_threads", "1",
                        "--score_chunk_size", "250000",
                    ]
                    init_paths = [index_marker(dataset, nlist)]
                    if adapter == "on":
                        init_paths.append(adapter_marker(dataset, nlist))
                    doc = job_doc(
                        name,
                        {
                            "app": "vectordb-large-scale",
                            "dataset": dataset,
                            "encoder": ENCODER_SHORT,
                            "stage": "eval",
                            "mode": mode,
                            "adapter": adapter,
                            "nprobe": f"np{nprobe}",
                        },
                        args,
                        *cfg["eval_mem"],
                        cpu="1",
                        init_paths=init_paths,
                    )
                    path = ds_dir / f"{name}.yaml"
                    path.write_text(doc)
                    docs.append(doc)
                    target_script = "apply_eval_adapter.sh" if adapter == "on" else "apply_eval_no_adapter.sh"
                    scripts[target_script].append(f"kubectl apply -f {ds_safe}/{path.name}")
                    scripts["apply_eval.sh"].append(f"kubectl apply -f {ds_safe}/{path.name}")
                    scripts["delete_all.sh"].append(f"kubectl delete -f {ds_safe}/{path.name} || true")

        agg_name = f"vdb-ls-{ds_safe}-{ENCODER_SHORT}-aggregate"[:63]
        wait_path = summary_marker(dataset, nlist, cfg["nprobes"][-1], "dts242", "on")
        agg_args = [
            "python", "aggregate_large_scale_retrieval.py",
            "--dataset", dataset,
            "--encoder", ENCODER,
            "--nlist", str(nlist),
            "--nprobes", ",".join(str(x) for x in cfg["nprobes"]),
            "--modes", ",".join(MODES),
            "--adapters", ",".join(ADAPTERS),
        ]
        agg_doc = job_doc(
            agg_name,
            {"app": "vectordb-large-scale", "dataset": dataset, "encoder": ENCODER_SHORT, "stage": "aggregate"},
            agg_args,
            *cfg["agg_mem"],
            cpu="1",
            init_paths=[wait_path],
        )
        agg_path = ds_dir / f"{agg_name}.yaml"
        agg_path.write_text(agg_doc)
        docs.append(agg_doc)
        scripts["apply_aggregate.sh"].append(f"kubectl apply -f {ds_safe}/{agg_path.name}")
        scripts["delete_all.sh"].append(f"kubectl delete -f {ds_safe}/{agg_path.name} || true")

    (outroot / "ALL.yaml").write_text("---\n".join(docs))
    for name, lines in scripts.items():
        path = outroot / name
        path.write_text("\n".join(lines) + "\n")
        os.chmod(path, 0o755)
    (outroot / "README.txt").write_text(
        "MiniLM adapter/no-adapter retrieval jobs.\n"
        "Order: apply_index.sh, apply_adapter.sh, apply_eval.sh, apply_aggregate.sh.\n"
        "Eval jobs wait on INDEX_READY and adapter-on jobs also wait on ADAPTER_READY.\n"
        "Do not apply until code is pushed to the GIT_REF used by the YAMLs.\n"
    )
    print(f"wrote {len(docs)} yamls to {outroot.resolve()}")


if __name__ == "__main__":
    main()
