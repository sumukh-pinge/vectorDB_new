#!/usr/bin/env python3
from pathlib import Path
import os
import shutil


NAMESPACE = "designlab"
IMAGE = "spinge/vectordb-new:20260304"
PVC = "sumukh-4tb"
MOUNT = "/mnt/work"
GIT_REPO = "https://github.com/sumukh-pinge/vectorDB_new.git"
GIT_REF = "main"
ENCODER = "all-MiniLM-L6-v2"
ENCODER_SHORT = "minilm"
ENCODER_TAG = ENCODER.replace("/", "_")
INDEX_ROOT = f"{MOUNT}/VectorDB_MICRO/large_scale_indices/large_scale_minilm_adapter_20260608"
RUN_ROOT = f"{MOUNT}/VectorDB_MICRO/large_scale_runs/large_scale_hardneg_minilm_20260609"
DATA_ROOT = f"{MOUNT}/VectorDB_MICRO/datasets/semantic"
INTERMEDIATE_ROOT = f"{MOUNT}/VectorDB_MICRO/intermediate_data"

DATASETS = {
    "nq": {
        "safe": "nq",
        "nlist": 1024,
        "qrels_split": "test",
        "train_split": "test",
        "nprobes": [64, 128],
        "mine_nprobe": 64,
        "subset": 5000,
        "shards": 1,
        "mem": ("56Gi", "64Gi"),
    },
    "dpr-w100": {
        "safe": "dpr-w100",
        "nlist": 4096,
        "qrels_split": "dev",
        "train_split": "train",
        "nprobes": [128, 256, 512],
        "mine_nprobe": 128,
        "subset": 10000,
        "shards": 8,
        "mem": ("56Gi", "64Gi"),
    },
    "miracl-en": {
        "safe": "miracl-en",
        "nlist": 8192,
        "qrels_split": "dev",
        "train_split": "train",
        "nprobes": [128, 256, 512, 1024],
        "mine_nprobe": 128,
        "subset": 10000,
        "shards": 8,
        "mem": ("56Gi", "64Gi"),
    },
}
MODES = ["ivf", "dts111", "dts242"]
EPOCHS = [1, 2, 3, 5]
HARD_MS = "1,1,1"
HARD_MS_TAG = HARD_MS.replace(",", "_")
ALPHA_NOTE = "2,2,2"


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
            cpu: "100m"
            memory: "128Mi"
        volumeMounts:
          - name: work
            mountPath: {MOUNT}
"""


def job_doc(name, labels, args, request_memory, limit_memory, cpu="1", init_paths=None, ttl=86400):
    label_lines = "\n".join([f'    {k}: "{v}"' for k, v in labels.items()])
    pod_label_lines = "\n".join([f'        {k}: "{v}"' for k, v in labels.items()])
    init = wait_init(init_paths or [])
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {NAMESPACE}
  labels:
{label_lines}
spec:
  backoffLimit: 2
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


def cache_slug(dataset, cfg):
    return (
        f"{dataset}_{ENCODER_TAG}_nlist{cfg['nlist']}"
        f"_split{cfg['train_split']}_np{cfg['mine_nprobe']}_ms{HARD_MS_TAG}"
        f"_seed123_subset{cfg['subset']}_k10"
    )


def cache_dir(dataset, cfg):
    return f"{INDEX_ROOT}/{dataset}/{ENCODER_TAG}/nlist{cfg['nlist']}/hard_negative_cache/{cache_slug(dataset, cfg)}"


def shard_path(dataset, cfg, shard):
    return f"{cache_dir(dataset, cfg)}/shard_{shard:04d}_of_{cfg['shards']:04d}.json"


def merged_path(dataset, cfg):
    return f"{cache_dir(dataset, cfg)}/merged.json"


def adapter_slug(dataset, cfg):
    return f"hardneg_{dataset}_{ENCODER_TAG}_nlist{cfg['nlist']}_ms{HARD_MS_TAG}_subset{cfg['subset']}"


def adapter_ready(dataset, cfg, slug):
    return f"{INDEX_ROOT}/{dataset}/{ENCODER_TAG}/nlist{cfg['nlist']}/ADAPTER_READY_{slug}.json"


def base_args(stage, dataset, cfg):
    return [
        "python", "run_large_scale_retrieval.py",
        "--stage", stage,
        "--dataset", dataset,
        "--encoder", ENCODER,
        "--qrels_split", cfg["qrels_split"],
        "--adapter_train_split", cfg["train_split"],
        "--nlist", str(cfg["nlist"]),
        "--nprobe", str(cfg["mine_nprobe"]),
        "--k2", "1000",
        "--kfinal", "100",
        "--faiss_threads", "1",
        "--score_chunk_size", "250000",
        "--hard_ms", HARD_MS,
        "--hard_max_queries", str(cfg["subset"]),
        "--hard_num_shards", str(cfg["shards"]),
        "--adapter_max_pairs", "100000",
        "--adapter_slug", adapter_slug(dataset, cfg),
    ]


def main():
    outroot = Path("yamls_generated/hard_negative_adapter_minilm_20260609")
    if outroot.exists():
        shutil.rmtree(outroot)
    outroot.mkdir(parents=True)
    docs = []
    scripts = {
        "apply_nq_mine.sh": ["#!/usr/bin/env bash", "set -euo pipefail"],
        "apply_nq_train.sh": ["#!/usr/bin/env bash", "set -euo pipefail"],
        "apply_nq_eval.sh": ["#!/usr/bin/env bash", "set -euo pipefail"],
        "apply_nq_aggregate.sh": ["#!/usr/bin/env bash", "set -euo pipefail"],
        "apply_large_mine.sh": ["#!/usr/bin/env bash", "set -euo pipefail"],
        "apply_large_train.sh": ["#!/usr/bin/env bash", "set -euo pipefail"],
        "apply_large_eval.sh": ["#!/usr/bin/env bash", "set -euo pipefail"],
        "apply_large_aggregate.sh": ["#!/usr/bin/env bash", "set -euo pipefail"],
        "delete_all.sh": ["#!/usr/bin/env bash", "set -euo pipefail"],
    }

    for dataset, cfg in DATASETS.items():
        ds_dir = outroot / cfg["safe"]
        ds_dir.mkdir(parents=True)
        req, lim = cfg["mem"]
        labels = {"app": "vectordb-hardneg", "dataset": cfg["safe"], "encoder": ENCODER_SHORT}
        target_prefix = "nq" if dataset == "nq" else "large"

        for shard in range(cfg["shards"]):
            name = f"vdb-hn-{cfg['safe']}-{ENCODER_SHORT}-mine-s{shard}"[:63]
            args = base_args("mine-hard-negatives", dataset, cfg) + ["--hard_shard_id", str(shard)]
            path = ds_dir / f"{name}.yaml"
            doc = job_doc(name, {**labels, "stage": "mine", "shard": str(shard)}, args, req, lim, cpu="1")
            path.write_text(doc)
            docs.append(doc)
            scripts[f"apply_{target_prefix}_mine.sh"].append(f"kubectl apply -f {cfg['safe']}/{path.name}")
            scripts["delete_all.sh"].append(f"kubectl delete -f {cfg['safe']}/{path.name} || true")

        merge_name = f"vdb-hn-{cfg['safe']}-{ENCODER_SHORT}-merge"[:63]
        merge_args = base_args("merge-hard-negatives", dataset, cfg)
        merge_path = ds_dir / f"{merge_name}.yaml"
        merge_doc = job_doc(
            merge_name,
            {**labels, "stage": "merge"},
            merge_args,
            "4Gi",
            "4Gi",
            cpu="1",
            init_paths=[shard_path(dataset, cfg, shard) for shard in range(cfg["shards"])],
        )
        merge_path.write_text(merge_doc)
        docs.append(merge_doc)
        scripts[f"apply_{target_prefix}_train.sh"].append(f"kubectl apply -f {cfg['safe']}/{merge_path.name}")
        scripts["delete_all.sh"].append(f"kubectl delete -f {cfg['safe']}/{merge_path.name} || true")

        train_name = f"vdb-hn-{cfg['safe']}-{ENCODER_SHORT}-train"[:63]
        train_args = base_args("train-from-neg-cache", dataset, cfg) + [
            "--adapter_epochs", "5",
            "--adapter_checkpoint_epochs", "1,2,3,5",
            "--adapter_qbatch", "256",
            "--adapter_device", "cpu",
            "--hard_train_negatives", "50000",
        ]
        train_path = ds_dir / f"{train_name}.yaml"
        train_doc = job_doc(
            train_name,
            {**labels, "stage": "train"},
            train_args,
            req,
            lim,
            cpu="1",
            init_paths=[merged_path(dataset, cfg)],
        )
        train_path.write_text(train_doc)
        docs.append(train_doc)
        scripts[f"apply_{target_prefix}_train.sh"].append(f"kubectl apply -f {cfg['safe']}/{train_path.name}")
        scripts["delete_all.sh"].append(f"kubectl delete -f {cfg['safe']}/{train_path.name} || true")

        for epoch in EPOCHS:
            epoch_slug = f"{adapter_slug(dataset, cfg)}_epoch{epoch}"
            for nprobe in cfg["nprobes"]:
                for mode in MODES:
                    name = f"vdb-hn-{cfg['safe']}-{ENCODER_SHORT}-e{epoch}-np{nprobe}-{mode}"[:63]
                    args = [
                        "python", "run_large_scale_retrieval.py",
                        "--stage", "eval",
                        "--dataset", dataset,
                        "--encoder", ENCODER,
                        "--qrels_split", cfg["qrels_split"],
                        "--mode", mode,
                        "--adapter", "on",
                        "--adapter_slug", epoch_slug,
                        "--result_adapter_tag", f"epoch{epoch}",
                        "--nlist", str(cfg["nlist"]),
                        "--nprobe", str(nprobe),
                        "--k2", "1000",
                        "--kfinal", "100",
                        "--faiss_threads", "1",
                        "--score_chunk_size", "250000",
                    ]
                    path = ds_dir / f"{name}.yaml"
                    doc = job_doc(
                        name,
                        {**labels, "stage": "eval", "epoch": str(epoch), "mode": mode, "nprobe": f"np{nprobe}"},
                        args,
                        req,
                        lim,
                        cpu="1",
                        init_paths=[adapter_ready(dataset, cfg, epoch_slug)],
                    )
                    path.write_text(doc)
                    docs.append(doc)
                    scripts[f"apply_{target_prefix}_eval.sh"].append(f"kubectl apply -f {cfg['safe']}/{path.name}")
                    scripts["delete_all.sh"].append(f"kubectl delete -f {cfg['safe']}/{path.name} || true")

        agg_name = f"vdb-hn-{cfg['safe']}-{ENCODER_SHORT}-aggregate"[:63]
        agg_args = [
            "python", "aggregate_large_scale_retrieval.py",
            "--dataset", dataset,
            "--encoder", ENCODER,
            "--nlist", str(cfg["nlist"]),
            "--nprobes", ",".join(str(x) for x in cfg["nprobes"]),
            "--modes", ",".join(MODES),
            "--adapters", ",".join(f"epoch{x}" for x in EPOCHS),
            "--wait_poll_sec", "60",
        ]
        agg_path = ds_dir / f"{agg_name}.yaml"
        agg_doc = job_doc(agg_name, {**labels, "stage": "aggregate"}, agg_args, "4Gi", "4Gi", cpu="1")
        agg_path.write_text(agg_doc)
        docs.append(agg_doc)
        scripts[f"apply_{target_prefix}_aggregate.sh"].append(f"kubectl apply -f {cfg['safe']}/{agg_path.name}")
        scripts["delete_all.sh"].append(f"kubectl delete -f {cfg['safe']}/{agg_path.name} || true")

    (outroot / "ALL.yaml").write_text("---\n".join(docs))
    for name, lines in scripts.items():
        path = outroot / name
        path.write_text("\n".join(lines) + "\n")
        os.chmod(path, 0o755)
    (outroot / "README.txt").write_text(
        "Hard-negative MiniLM adapter workflow.\n"
        "NQ order: apply_nq_mine.sh, apply_nq_train.sh, apply_nq_eval.sh, then apply_nq_aggregate.sh after evals finish.\n"
        "Large order after NQ validation: apply_large_mine.sh, apply_large_train.sh, apply_large_eval.sh, then apply_large_aggregate.sh.\n"
        f"Mining config: ms={HARD_MS}, alphas={ALPHA_NOTE}, K2=1000, Kfinal=100.\n"
        "Do not apply large jobs until NQ confirms the cache/train/eval path.\n"
    )
    print(f"wrote {len(docs)} docs to {outroot.resolve()}")


if __name__ == "__main__":
    main()
