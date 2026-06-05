#!/usr/bin/env python3
import os
from pathlib import Path


NAMESPACE = "designlab"
IMAGE = "spinge/vectordb-new:20260304"
PVC = "sumukh-4tb"
MOUNT = "/mnt/work"

GIT_REPO = "https://github.com/sumukh-pinge/vectorDB_new.git"
GIT_REF = os.getenv("GIT_REF", "main")

DATASET = "nq"
SEEDS = [42, 43]
NPROBES = [64, 128]
EVAL_SETTINGS = [
    ("base", "base_none"),
    ("expnone", "expanded_none"),
    ("frozen", "expanded_frozen"),
    ("refresh", "expanded_refreshed"),
]

ENCODERS = {
    "minilm": {
        "name": "all-MiniLM-L6-v2",
        "request_memory": "56Gi",
        "limit_memory": "64Gi",
    },
    "mpnet": {
        "name": "all-mpnet-base-v2",
        "request_memory": "112Gi",
        "limit_memory": "128Gi",
    },
}

RUN_LABEL = "e2_db_expansion_nq_overnight_20260605"
RUN_ROOT = f"{MOUNT}/VectorDB_MICRO/DBAM/db_expansion_runs_20260605"
DATA_ROOT = f"{MOUNT}/VectorDB_MICRO/datasets/semantic"
INTERMEDIATE_ROOT = f"{MOUNT}/VectorDB_MICRO/intermediate_data"

COMMON_ARGS = {
    "dataset": DATASET,
    "run_label": RUN_LABEL,
    "data_root": DATA_ROOT,
    "intermediate_root": INTERMEDIATE_ROOT,
    "run_root": RUN_ROOT,
    "holdout_doc_frac": "0.20",
    "base_train_frac": "0.80",
    "new_eval_frac": "0.50",
    "bits_sq": "4",
    "nlist": "1024",
    "k2": "1000",
    "kfinals": "10,25,50,100",
    "alphas": "2,2,2",
    "ms_values": "2,4,2;1,1,1",
    "adapter_mining_stages": "dual,dual,dual",
    "adapter_mining_ms": "2,4,2",
    "adapter_neg_kfinal": "10",
    "adapt_epochs": "5",
    "adapt_lr": "5e-4",
    "adapt_subset": "50000",
    "adapt_qbatch": "64",
    "tau": "1.0",
    "beta": "6.0",
    "gamma": "1.0",
    "grad_clip": "1.0",
    "wait_timeout_sec": "43200",
    "wait_poll_sec": "60",
}


def encoder_tag(name):
    return name.replace("/", "_")


def arg_list(stage, enc_cfg, seed, nprobe, eval_setting=None, aggregate_expected_tables=None):
    args = [
        "python",
        "run_db_expansion_experiment.py",
        "--stage",
        stage,
        "--encoder",
        enc_cfg["name"],
        "--seed",
        str(seed),
        "--eval_nprobe",
        str(nprobe),
        "--adapter_mining_nprobe",
        str(nprobe),
    ]
    if eval_setting is not None:
        args.extend(["--eval_setting", eval_setting])
    if aggregate_expected_tables is not None:
        args.extend(["--aggregate_expected_tables", str(aggregate_expected_tables)])
    for key, value in COMMON_ARGS.items():
        args.extend([f"--{key}", value])
    return args


def yaml_args(args):
    return "\n".join(f'          - "{arg}"' for arg in args)


def job_doc(name, labels, args, enc_cfg):
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
      containers:
      - name: run
        image: {IMAGE}
        imagePullPolicy: IfNotPresent
        command: ["/usr/local/bin/nrp-launch"]
        args:
{yaml_args(args)}
        env:
          - name: GIT_REPO
            value: "{GIT_REPO}"
          - name: GIT_REF
            value: "{GIT_REF}"
          - name: HF_HOME
            value: "{MOUNT}/hf_cache"
          - name: TRANSFORMERS_CACHE
            value: "{MOUNT}/hf_cache"
          - name: DATA_ROOT
            value: "{DATA_ROOT}"
          - name: INTERMEDIATE_ROOT
            value: "{INTERMEDIATE_ROOT}"
          - name: RUN_ROOT
            value: "{RUN_ROOT}"
          - name: PYTHONUNBUFFERED
            value: "1"
        resources:
          requests:
            nvidia.com/gpu: 0
            cpu: "4"
            memory: "{enc_cfg["request_memory"]}"
          limits:
            nvidia.com/gpu: 0
            cpu: "4"
            memory: "{enc_cfg["limit_memory"]}"
        volumeMounts:
          - name: work
            mountPath: {MOUNT}
      volumes:
        - name: work
          persistentVolumeClaim:
            claimName: {PVC}
"""


def labels(enc_short, seed, nprobe, stage, setting="none"):
    return {
        "app": "vectordb-e2-dbexp",
        "dataset": DATASET,
        "encoder": enc_short,
        "seed": f"s{seed}",
        "nprobe": f"np{nprobe}",
        "stage": stage,
        "setting": setting,
    }


def main():
    outroot = Path("yamls_generated/e2_db_expansion_nq_overnight_20260605")
    outroot.mkdir(parents=True, exist_ok=True)

    apply_prep = ["#!/usr/bin/env bash", "set -euo pipefail"]
    apply_eval = ["#!/usr/bin/env bash", "set -euo pipefail"]
    apply_all = ["#!/usr/bin/env bash", "set -euo pipefail"]
    delete_all = ["#!/usr/bin/env bash", "set -euo pipefail"]

    total = 0
    for enc_short, enc_cfg in ENCODERS.items():
        enc_out = outroot / enc_short
        enc_out.mkdir(parents=True, exist_ok=True)
        docs = []

        for seed in SEEDS:
            for nprobe in NPROBES:
                prep_name = f"vdb-e2-{enc_short}-nq-np{nprobe}-s{seed}-prep"
                prep_doc = job_doc(
                    prep_name,
                    labels(enc_short, seed, nprobe, "prep"),
                    arg_list("prep", enc_cfg, seed, nprobe),
                    enc_cfg,
                )
                prep_path = enc_out / f"{prep_name}.yaml"
                prep_path.write_text(prep_doc)
                docs.append(prep_doc)
                apply_prep.append(f"kubectl apply -f {enc_short}/{prep_name}.yaml")
                apply_all.append(f"kubectl apply -f {enc_short}/{prep_name}.yaml")
                delete_all.append(f"kubectl delete -f {enc_short}/{prep_name}.yaml || true")
                total += 1

                for setting_short, eval_setting in EVAL_SETTINGS:
                    eval_name = f"vdb-e2-{enc_short}-nq-np{nprobe}-s{seed}-{setting_short}"
                    eval_doc = job_doc(
                        eval_name,
                        labels(enc_short, seed, nprobe, "eval", setting_short),
                        arg_list("eval", enc_cfg, seed, nprobe, eval_setting=eval_setting),
                        enc_cfg,
                    )
                    eval_path = enc_out / f"{eval_name}.yaml"
                    eval_path.write_text(eval_doc)
                    docs.append(eval_doc)
                    apply_eval.append(f"kubectl apply -f {enc_short}/{eval_name}.yaml")
                    apply_all.append(f"kubectl apply -f {enc_short}/{eval_name}.yaml")
                    delete_all.append(f"kubectl delete -f {enc_short}/{eval_name}.yaml || true")
                    total += 1

                summary_name = f"vdb-e2-{enc_short}-nq-np{nprobe}-s{seed}-summary"
                summary_doc = job_doc(
                    summary_name,
                    labels(enc_short, seed, nprobe, "summarize"),
                    arg_list("summarize", enc_cfg, seed, nprobe),
                    enc_cfg,
                )
                summary_path = enc_out / f"{summary_name}.yaml"
                summary_path.write_text(summary_doc)
                docs.append(summary_doc)
                apply_eval.append(f"kubectl apply -f {enc_short}/{summary_name}.yaml")
                apply_all.append(f"kubectl apply -f {enc_short}/{summary_name}.yaml")
                delete_all.append(f"kubectl delete -f {enc_short}/{summary_name}.yaml || true")
                total += 1

        aggregate_expected = len(SEEDS) * len(NPROBES)
        aggregate_name = f"vdb-e2-{enc_short}-nq-aggregate"
        aggregate_doc = job_doc(
            aggregate_name,
            {
                "app": "vectordb-e2-dbexp",
                "dataset": DATASET,
                "encoder": enc_short,
                "seed": "all",
                "nprobe": "all",
                "stage": "aggregate",
                "setting": "all",
            },
            arg_list("aggregate", enc_cfg, SEEDS[0], NPROBES[0], aggregate_expected_tables=aggregate_expected),
            enc_cfg,
        )
        aggregate_path = enc_out / f"{aggregate_name}.yaml"
        aggregate_path.write_text(aggregate_doc)
        docs.append(aggregate_doc)
        apply_eval.append(f"kubectl apply -f {enc_short}/{aggregate_name}.yaml")
        apply_all.append(f"kubectl apply -f {enc_short}/{aggregate_name}.yaml")
        delete_all.append(f"kubectl delete -f {enc_short}/{aggregate_name}.yaml || true")
        total += 1

        (enc_out / "ALL.yaml").write_text("---\n".join(docs))

    (outroot / "apply_prep.sh").write_text("\n".join(apply_prep) + "\n")
    os.chmod(outroot / "apply_prep.sh", 0o755)
    (outroot / "apply_eval_and_summary.sh").write_text("\n".join(apply_eval) + "\n")
    os.chmod(outroot / "apply_eval_and_summary.sh", 0o755)
    (outroot / "apply_all_overnight.sh").write_text("\n".join(apply_all) + "\n")
    os.chmod(outroot / "apply_all_overnight.sh", 0o755)
    (outroot / "delete_all.sh").write_text("\n".join(delete_all) + "\n")
    os.chmod(outroot / "delete_all.sh", 0o755)

    (outroot / "RESULTS_ROOT.txt").write_text(RUN_ROOT + "\n")
    (outroot / "COPY_HINTS.txt").write_text(
        f"All results land under:\n{RUN_ROOT}\n\n"
        "Primary per-run final tables:\n"
        f"{RUN_ROOT}/nq/<encoder_tag>/results/db_expansion/{RUN_LABEL}/np*/**/final_rebuttal_table.csv\n\n"
        "Aggregate tables:\n"
        f"{RUN_ROOT}/nq/<encoder_tag>/results/db_expansion/{RUN_LABEL}/aggregate/\n\n"
        f"Generated job count: {total}\n"
        f"GIT_REF used in YAMLs: {GIT_REF}\n"
    )

    print(f"wrote {total} jobs under {outroot.resolve()}")
    print(f"results root: {RUN_ROOT}")
    print(f"git ref in YAMLs: {GIT_REF}")


if __name__ == "__main__":
    main()
