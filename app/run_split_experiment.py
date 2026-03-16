#!/usr/bin/env python3
import os
import json
import time
import argparse
import fcntl
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm

from utilis_dbam_v3_new import (
    build_pipeline_baseline,
    run_and_evaluate,
    load_or_build_pipeline_for_adapter,
    train_W_param,
)

SCENARIOS = {
    "100_100": ("full_full", None),
    "90_10":   ("disjoint", 0.90),
    "80_20":   ("disjoint", 0.80),
    "70_30":   ("disjoint", 0.70),
    "60_40":   ("disjoint", 0.60),
    "50_50":   ("disjoint", 0.50),
    "40_60":   ("disjoint", 0.40),
    "30_70":   ("disjoint", 0.30),
    "20_80":   ("disjoint", 0.20),
    "10_90":   ("disjoint", 0.10),
    "0_100":   ("zero_full", None),
}


def as_tuple_int(s: str):
    vals = tuple(int(x.strip()) for x in s.split(","))
    if len(vals) != 3:
        raise ValueError(f"Expected 3 comma-separated ints, got: {s}")
    return vals


def as_tuple_str(s: str):
    vals = tuple(x.strip() for x in s.split(","))
    if len(vals) != 3:
        raise ValueError(f"Expected 3 comma-separated strings, got: {s}")
    return vals


def parse_int_list(s: str):
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer")
    return vals


def l2norm(x, eps=1e-12):
    n = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
    return (x / n).astype("float32")


@contextmanager
def file_lock(lock_path: str):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def resolve_dataset_dir(data_root: str, dataset: str) -> Path:
    ds_dir = Path(data_root) / dataset
    if not (ds_dir / "corpus.jsonl").exists() and (ds_dir / dataset / "corpus.jsonl").exists():
        ds_dir = ds_dir / dataset
    return ds_dir


def choose_qrels_file(ds_dir: Path):
    qrels_dir = ds_dir / "qrels"
    for split in ["test", "dev", "train"]:
        cand = qrels_dir / f"{split}.tsv"
        if cand.exists():
            return split, cand
    raise FileNotFoundError(f"No qrels split found under {qrels_dir}")


def load_or_build_passage_embeddings(
    corpus_file: str,
    embed_path: str,
    encoder: SentenceTransformer,
    batch_size: int = 512,
):
    if os.path.exists(embed_path):
        passage_ids = []
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                passage_ids.append(str(json.loads(line)["_id"]))
        embeddings = np.load(embed_path).astype("float32")
        return passage_ids, embeddings

    lock_path = embed_path + ".lock"
    with file_lock(lock_path):
        if os.path.exists(embed_path):
            passage_ids = []
            with open(corpus_file, "r", encoding="utf-8") as f:
                for line in f:
                    passage_ids.append(str(json.loads(line)["_id"]))
            embeddings = np.load(embed_path).astype("float32")
            return passage_ids, embeddings

        passage_ids = []
        chunks = []
        texts = []

        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Encoding passages -> {os.path.basename(embed_path)}"):
                obj = json.loads(line)
                passage_ids.append(str(obj["_id"]))
                texts.append((obj.get("title", "") + " " + obj.get("text", "")).strip())

                if len(texts) >= batch_size:
                    emb = encoder.encode(
                        texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=batch_size,
                    )
                    chunks.append(emb.astype("float32"))
                    texts = []

        if texts:
            emb = encoder.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )
            chunks.append(emb.astype("float32"))

        embeddings = np.vstack(chunks).astype("float32")
        np.save(embed_path, embeddings)
        return passage_ids, embeddings


def load_or_build_query_embeddings(
    queries_file: str,
    qrels_file: str,
    qembed_path: str,
    encoder: SentenceTransformer,
    batch_size: int = 128,
):
    qrels = pd.read_csv(
        qrels_file,
        sep="\t",
        header=None,
        names=["query_id", "corpus_id", "score"],
        dtype=str,
    )
    pos_qrels = qrels[qrels["score"] != "0"].reset_index(drop=True)
    keep_qids = set(pos_qrels["query_id"].astype(str).unique().tolist())

    queries_ds = [json.loads(line) for line in open(queries_file, "r", encoding="utf-8")]
    queries_sample = [q for q in queries_ds if str(q["_id"]) in keep_qids]
    query_ids = [str(q["_id"]) for q in queries_sample]
    query_texts = [q["text"] for q in queries_sample]

    if os.path.exists(qembed_path):
        queries_emb = np.load(qembed_path).astype("float32")
    else:
        lock_path = qembed_path + ".lock"
        with file_lock(lock_path):
            if os.path.exists(qembed_path):
                queries_emb = np.load(qembed_path).astype("float32")
            else:
                queries_emb = encoder.encode(
                    query_texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                ).astype("float32")
                np.save(qembed_path, queries_emb)

    query_to_gt = {}
    for _, r in pos_qrels.iterrows():
        qid = str(r["query_id"])
        pid = str(r["corpus_id"])
        if qid in keep_qids:
            query_to_gt.setdefault(qid, []).append(pid)

    return query_ids, queries_emb, query_to_gt, pos_qrels


def subset_queries(query_ids_all, queries_emb_all, query_to_gt_all, keep_set):
    idx = [i for i, qid in enumerate(query_ids_all) if qid in keep_set]
    qids = [query_ids_all[i] for i in idx]
    qemb = queries_emb_all[idx]
    q2gt = {qid: query_to_gt_all[qid] for qid in qids}
    return idx, qids, qemb, q2gt


def metric_row(df):
    row = df.iloc[0].to_dict()
    keep = [
        "H@10", "H@25", "H@50", "H@100",
        "R@10", "R@25", "R@50", "R@100",
        "MRR@10", "MRR@25", "MRR@50", "MRR@100",
    ]
    return {k: row.get(k) for k in keep}


def empty_metric_row():
    return {
        "H@10": None, "H@25": None, "H@50": None, "H@100": None,
        "R@10": None, "R@25": None, "R@50": None, "R@100": None,
        "MRR@10": None, "MRR@25": None, "MRR@50": None, "MRR@100": None,
    }


def make_disjoint_split(query_ids_all, train_frac, seed=42, allow_zero_train=False):
    ordered = np.array(sorted(query_ids_all), dtype=object)
    perm = np.random.default_rng(seed).permutation(len(ordered))
    ordered = ordered[perm].tolist()

    n_total = len(ordered)
    n_train = int(round(train_frac * n_total))

    if allow_zero_train:
        n_train = max(0, min(n_total, n_train))
    else:
        n_train = max(1, min(n_total - 1, n_train))

    train_set = set(ordered[:n_train])
    eval_set = set(ordered[n_train:])
    return train_set, eval_set


def seed_everything(seed: int, device: str):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def run_split_case(split_tag, mode, train_frac, seed, data, cfg, job_out_dir):
    seed_everything(seed, cfg["device"])

    query_ids_sample = data["query_ids"]
    queries_emb_sample = data["queries_emb"]
    query_to_gt = data["query_to_gt"]
    embeddings = data["embeddings"]
    passage_ids_sample = data["passage_ids"]

    all_qids_set = set(query_ids_sample)

    if mode == "disjoint":
        train_set, eval_set = make_disjoint_split(
            query_ids_sample,
            train_frac=train_frac,
            seed=seed,
            allow_zero_train=False,
        )
    elif mode == "zero_full":
        train_set = set()
        eval_set = all_qids_set
    elif mode == "full_full":
        train_set = all_qids_set
        eval_set = all_qids_set
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    _, train_qids, train_qemb, train_q2gt = subset_queries(
        query_ids_sample, queries_emb_sample, query_to_gt, train_set
    )
    _, eval_qids, eval_qemb, eval_q2gt = subset_queries(
        query_ids_sample, queries_emb_sample, query_to_gt, eval_set
    )

    print("=" * 100, flush=True)
    print(f"Split: {split_tag}", flush=True)
    print(f"Mode : {mode}", flush=True)
    print(f"Seed : {seed}", flush=True)
    print(f"train queries: {len(train_qids)}", flush=True)
    print(f"eval  queries: {len(eval_qids)}", flush=True)

    scenario_results_dir = os.path.join(job_out_dir, f"seed_{seed}")
    os.makedirs(scenario_results_dir, exist_ok=True)

    adapter_trial_split = (
        f"adapter_{cfg['dataset']}_{cfg['encoder_tag']}_{split_tag}_seed{seed}"
        f"_mineDDD_m{cfg['adapter_mining_ms_tag']}_np{cfg['adapter_mining_nprobe']}"
    )
    w_path_split = os.path.join(cfg["intermediate_dir"], f"W_{adapter_trial_split}.npy")
    adapter_ckpt_split = os.path.join(cfg["intermediate_dir"], f"W_{adapter_trial_split}_adapter.pt")

    pipeline_eval_no = build_pipeline_baseline(
        slug=f"base_eval_{split_tag}_seed{seed}",
        embeddings=embeddings,
        queries_emb_sample=eval_qemb,
        passage_ids_sample=passage_ids_sample,
        bits_sq=cfg["bits_sq"],
        nlist=cfg["nlist"],
    )

    cfg_ivf_no = {
        "experiment_name": f"{split_tag}_seed{seed}_np{cfg['eval_nprobe']}_ivf_ivf_ivf_no_adapter",
        "nprobe_sweep_values": [cfg["eval_nprobe"]],
        "stage_methods": {"s1": "ivf", "s2": "ivf", "s3": "ivf"},
        "alphas": cfg["alphas_infer"],
        "ms": cfg["ms_ivf"],
        "k2_fixed": cfg["k2_fixed"],
        "k_final_values": cfg["k_finals"],
    }
    df_ivf_no = run_and_evaluate(
        cfg_ivf_no, scenario_results_dir, pipeline_eval_no, eval_qids, eval_q2gt
    )

    cfg_dual_no = {
        "experiment_name": (
            f"{split_tag}_seed{seed}_np{cfg['eval_nprobe']}_"
            f"dual_dual_dual_no_adapter_m{cfg['ms_dual_tag']}"
        ),
        "nprobe_sweep_values": [cfg["eval_nprobe"]],
        "stage_methods": {"s1": "dual", "s2": "dual", "s3": "dual"},
        "alphas": cfg["alphas_infer"],
        "ms": cfg["ms_dual"],
        "k2_fixed": cfg["k2_fixed"],
        "k_final_values": cfg["k_finals"],
    }
    df_dual_no = run_and_evaluate(
        cfg_dual_no, scenario_results_dir, pipeline_eval_no, eval_qids, eval_q2gt
    )

    df_ivf_a = None
    df_dual_a = None

    if len(train_qids) > 0:
        pipeline_train = build_pipeline_baseline(
            slug=f"base_train_{split_tag}_seed{seed}",
            embeddings=embeddings,
            queries_emb_sample=train_qemb,
            passage_ids_sample=passage_ids_sample,
            bits_sq=cfg["bits_sq"],
            nlist=cfg["nlist"],
        )

        if not os.path.exists(adapter_ckpt_split):
            print(f"⚙️ training adapter: {adapter_ckpt_split}", flush=True)
            train_W_param(
                emb_np=embeddings,
                que_np=train_qemb,
                d_out=embeddings.shape[1],
                save_path=w_path_split,
                passage_ids_sample=passage_ids_sample,
                query_ids_sample=train_qids,
                query_to_gt=train_q2gt,
                device=cfg["device"],
                epochs=cfg["adapt_epochs"],
                lr=cfg["adapt_lr"],
                subset=cfg["adapt_subset"],
                q_batch=cfg["adapt_qbatch"],
                tau=cfg["tau"],
                beta=cfg["beta"],
                gamma=cfg["gamma"],
                grad_clip=cfg["grad_clip"],
                pipeline_data=pipeline_train,
                ALPHAS_INFER=cfg["alphas_infer"],
                MS_INFER=cfg["adapter_mining_ms"],
                STAGES_INFER=cfg["adapter_mining_stages"],
                SELECT_NPROBE=cfg["adapter_mining_nprobe"],
                K_final=cfg["adapter_neg_kfinal"],
            )
        else:
            print(f"Reusing adapter checkpoint: {adapter_ckpt_split}", flush=True)

        pipeline_eval_adapt, _ = load_or_build_pipeline_for_adapter(
            trial_slug=adapter_trial_split,
            embeddings=embeddings,
            queries_emb_sample=eval_qemb,
            passage_ids_sample=passage_ids_sample,
            nlist=cfg["nlist"],
            bits_sq=cfg["bits_sq"],
            intermediate_dir=cfg["intermediate_dir"],
            device=cfg["device"],
            fallback_to_baseline=False,
        )

        cfg_ivf_a = {
            "experiment_name": f"{split_tag}_seed{seed}_np{cfg['eval_nprobe']}_ivf_ivf_ivf_with_adapter",
            "nprobe_sweep_values": [cfg["eval_nprobe"]],
            "stage_methods": {"s1": "ivf", "s2": "ivf", "s3": "ivf"},
            "alphas": cfg["alphas_infer"],
            "ms": cfg["ms_ivf"],
            "k2_fixed": cfg["k2_fixed"],
            "k_final_values": cfg["k_finals"],
        }
        df_ivf_a = run_and_evaluate(
            cfg_ivf_a, scenario_results_dir, pipeline_eval_adapt, eval_qids, eval_q2gt
        )

        cfg_dual_a = {
            "experiment_name": (
                f"{split_tag}_seed{seed}_np{cfg['eval_nprobe']}_"
                f"dual_dual_dual_with_adapter_m{cfg['ms_dual_tag']}"
            ),
            "nprobe_sweep_values": [cfg["eval_nprobe"]],
            "stage_methods": {"s1": "dual", "s2": "dual", "s3": "dual"},
            "alphas": cfg["alphas_infer"],
            "ms": cfg["ms_dual"],
            "k2_fixed": cfg["k2_fixed"],
            "k_final_values": cfg["k_finals"],
        }
        df_dual_a = run_and_evaluate(
            cfg_dual_a, scenario_results_dir, pipeline_eval_adapt, eval_qids, eval_q2gt
        )
    else:
        print("No train queries -> adapter training skipped.", flush=True)

    rows = [
        {
            "dataset": cfg["dataset"],
            "encoder": cfg["encoder_name"],
            "encoder_tag": cfg["encoder_tag"],
            "split": split_tag,
            "seed": seed,
            "eval_nprobe": cfg["eval_nprobe"],
            "mode": "ivf_ivf_ivf",
            "adapter": "off",
            "ms": ",".join(map(str, cfg["ms_ivf"])),
            **metric_row(df_ivf_no),
        },
        {
            "dataset": cfg["dataset"],
            "encoder": cfg["encoder_name"],
            "encoder_tag": cfg["encoder_tag"],
            "split": split_tag,
            "seed": seed,
            "eval_nprobe": cfg["eval_nprobe"],
            "mode": "dual_dual_dual",
            "adapter": "off",
            "ms": ",".join(map(str, cfg["ms_dual"])),
            **metric_row(df_dual_no),
        },
        {
            "dataset": cfg["dataset"],
            "encoder": cfg["encoder_name"],
            "encoder_tag": cfg["encoder_tag"],
            "split": split_tag,
            "seed": seed,
            "eval_nprobe": cfg["eval_nprobe"],
            "mode": "ivf_ivf_ivf",
            "adapter": "on",
            "ms": ",".join(map(str, cfg["ms_ivf"])),
            **(metric_row(df_ivf_a) if df_ivf_a is not None else empty_metric_row()),
        },
        {
            "dataset": cfg["dataset"],
            "encoder": cfg["encoder_name"],
            "encoder_tag": cfg["encoder_tag"],
            "split": split_tag,
            "seed": seed,
            "eval_nprobe": cfg["eval_nprobe"],
            "mode": "dual_dual_dual",
            "adapter": "on",
            "ms": ",".join(map(str, cfg["ms_dual"])),
            **(metric_row(df_dual_a) if df_dual_a is not None else empty_metric_row()),
        },
    ]

    return {
        "split_tag": split_tag,
        "seed": seed,
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser("Run one split job with 3 seeds and all 4 eval cases.")
    p.add_argument("--dataset", required=True, choices=["nq", "hotpotqa"])
    p.add_argument("--encoder", default="all-MiniLM-L6-v2")
    p.add_argument("--split", required=True, choices=list(SCENARIOS.keys()))
    p.add_argument("--seeds", default="42,43,44")

    p.add_argument("--data_root", default=os.getenv("DATA_ROOT", "/mnt/work/VectorDB_MICRO/datasets/semantic"))
    p.add_argument("--intermediate_root", default=os.getenv("INTERMEDIATE_ROOT", "/mnt/work/VectorDB_MICRO/intermediate_data"))
    p.add_argument("--run_root", default=os.getenv("RUN_ROOT", "/mnt/work/VectorDB_MICRO/DBAM/runs"))

    p.add_argument("--bits_sq", type=int, default=4)
    p.add_argument("--nlist", type=int, default=1024)
    p.add_argument("--eval_nprobe", type=int, required=True)
    p.add_argument("--k2", type=int, default=1000)
    p.add_argument("--kfinals", default="10,25,50,100")
    p.add_argument("--alphas", default="2,2,2")
    p.add_argument("--ms_ivf", default="1,1,1")
    p.add_argument("--ms_dual", default="1,1,1")

    p.add_argument("--adapter_mining_stages", default="dual,dual,dual")
    p.add_argument("--adapter_mining_ms", default="1,1,1")
    p.add_argument("--adapter_mining_nprobe", type=int, required=True)
    p.add_argument("--adapter_neg_kfinal", type=int, default=10)

    p.add_argument("--adapt_epochs", type=int, default=5)
    p.add_argument("--adapt_lr", type=float, default=5e-4)
    p.add_argument("--adapt_subset", type=int, default=50000)
    p.add_argument("--adapt_qbatch", type=int, default=64)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=6.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    seeds = parse_int_list(args.seeds)
    mode, train_frac = SCENARIOS[args.split]

    encoder_tag = args.encoder.replace("/", "_")
    ms_ivf = as_tuple_int(args.ms_ivf)
    ms_dual = as_tuple_int(args.ms_dual)
    alphas = as_tuple_int(args.alphas)
    adapter_mining_stages = as_tuple_str(args.adapter_mining_stages)
    adapter_mining_ms = as_tuple_int(args.adapter_mining_ms)

    ms_dual_tag = "_".join(map(str, ms_dual))
    adapter_mining_ms_tag = "_".join(map(str, adapter_mining_ms))

    ds_dir = resolve_dataset_dir(args.data_root, args.dataset)
    qrels_split, qrels_file = choose_qrels_file(ds_dir)

    corpus_file = str(ds_dir / "corpus.jsonl")
    queries_file = str(ds_dir / "queries.jsonl")

    inter_dir = os.path.join(args.intermediate_root, args.dataset, encoder_tag)
    job_out_dir = os.path.join(
        args.run_root,
        args.dataset,
        encoder_tag,
        "results",
        "split_runs",
        f"split_{args.split}",
        f"np{args.eval_nprobe}",
    )
    os.makedirs(inter_dir, exist_ok=True)
    os.makedirs(job_out_dir, exist_ok=True)

    embed_path = os.path.join(inter_dir, f"passage_embeddings_{args.dataset}_{encoder_tag}.npy")
    qembed_path = os.path.join(inter_dir, f"query_embeddings_{args.dataset}_{encoder_tag}_{qrels_split}.npy")

    t0 = time.time()
    encoder = SentenceTransformer(args.encoder, device=device)

    passage_ids, embeddings = load_or_build_passage_embeddings(
        corpus_file=corpus_file,
        embed_path=embed_path,
        encoder=encoder,
        batch_size=512,
    )

    query_ids, queries_emb, query_to_gt, pos_qrels = load_or_build_query_embeddings(
        queries_file=queries_file,
        qrels_file=str(qrels_file),
        qembed_path=qembed_path,
        encoder=encoder,
        batch_size=128,
    )

    embeddings = l2norm(embeddings)
    queries_emb = l2norm(queries_emb)
    t_load = time.time() - t0

    print("passages:", embeddings.shape, embeddings.dtype, flush=True)
    print("queries :", queries_emb.shape, queries_emb.dtype, flush=True)
    print("ids     :", len(passage_ids), len(query_ids), flush=True)
    print("qrels split:", qrels_split, flush=True)

    data = {
        "passage_ids": passage_ids,
        "embeddings": embeddings,
        "query_ids": query_ids,
        "queries_emb": queries_emb,
        "query_to_gt": query_to_gt,
    }

    cfg = {
        "dataset": args.dataset,
        "encoder_name": args.encoder,
        "encoder_tag": encoder_tag,
        "device": device,
        "intermediate_dir": inter_dir,
        "bits_sq": args.bits_sq,
        "nlist": args.nlist,
        "alphas_infer": alphas,
        "ms_ivf": ms_ivf,
        "ms_dual": ms_dual,
        "ms_dual_tag": ms_dual_tag,
        "eval_nprobe": args.eval_nprobe,
        "k2_fixed": args.k2,
        "k_finals": parse_int_list(args.kfinals),
        "adapter_mining_stages": adapter_mining_stages,
        "adapter_mining_ms": adapter_mining_ms,
        "adapter_mining_ms_tag": adapter_mining_ms_tag,
        "adapter_mining_nprobe": args.adapter_mining_nprobe,
        "adapter_neg_kfinal": args.adapter_neg_kfinal,
        "adapt_epochs": args.adapt_epochs,
        "adapt_lr": args.adapt_lr,
        "adapt_subset": args.adapt_subset,
        "adapt_qbatch": args.adapt_qbatch,
        "tau": args.tau,
        "beta": args.beta,
        "gamma": args.gamma,
        "grad_clip": args.grad_clip,
    }

    print("split            =", args.split, flush=True)
    print("mode/train_frac  =", mode, train_frac, flush=True)
    print("seeds            =", seeds, flush=True)
    print("eval_nprobe      =", args.eval_nprobe, flush=True)
    print("ms_dual          =", ms_dual, flush=True)
    print("adapter mining   =", adapter_mining_stages, adapter_mining_ms, args.adapter_mining_nprobe, flush=True)

    all_rows = []
    all_outputs = {}

    for seed in seeds:
        out = run_split_case(
            split_tag=args.split,
            mode=mode,
            train_frac=train_frac,
            seed=seed,
            data=data,
            cfg=cfg,
            job_out_dir=job_out_dir,
        )
        all_outputs[f"seed_{seed}"] = out
        all_rows.extend(out["rows"])

    summary_df = pd.DataFrame(all_rows)
    summary_csv = os.path.join(job_out_dir, "summary_all_seeds.csv")
    summary_df.to_csv(summary_csv, index=False)

    metric_cols = [c for c in summary_df.columns if c.startswith(("H@", "R@", "MRR@"))]
    group_cols = ["dataset", "encoder", "encoder_tag", "split", "eval_nprobe", "mode", "adapter", "ms"]
    agg_df = summary_df.groupby(group_cols, as_index=False)[metric_cols].agg(["mean", "std"])

    agg_df.columns = [
        "_".join([str(x) for x in col if x != ""]).rstrip("_") if isinstance(col, tuple) else col
        for col in agg_df.columns
    ]
    agg_csv = os.path.join(job_out_dir, "summary_mean_std.csv")
    agg_df.to_csv(agg_csv, index=False)

    short_cols = [
        "dataset", "encoder", "encoder_tag", "split", "eval_nprobe", "mode", "adapter", "ms",
        "H@25_mean", "H@25_std", "H@100_mean", "H@100_std",
    ]
    short_df = agg_df[short_cols].copy()
    short_csv = os.path.join(job_out_dir, "summary_h25_h100_mean_std.csv")
    short_df.to_csv(short_csv, index=False)

    meta = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "split": args.split,
        "mode": mode,
        "train_frac": train_frac,
        "seeds": seeds,
        "qrels_split": qrels_split,
        "paths": {
            "dataset_dir": str(ds_dir),
            "corpus_file": corpus_file,
            "queries_file": queries_file,
            "qrels_file": str(qrels_file),
            "embed_path": embed_path,
            "qembed_path": qembed_path,
            "intermediate_dir": inter_dir,
            "job_out_dir": job_out_dir,
            "summary_all_seeds": summary_csv,
            "summary_mean_std": agg_csv,
            "summary_h25_h100_mean_std": short_csv,
        },
        "counts": {
            "passages": len(passage_ids),
            "queries": len(query_ids),
            "pos_qrels": int(len(pos_qrels)),
        },
        "config": {
            "bits_sq": args.bits_sq,
            "nlist": args.nlist,
            "eval_nprobe": args.eval_nprobe,
            "k2": args.k2,
            "kfinals": parse_int_list(args.kfinals),
            "alphas": alphas,
            "ms_ivf": ms_ivf,
            "ms_dual": ms_dual,
            "adapter_mining_stages": adapter_mining_stages,
            "adapter_mining_ms": adapter_mining_ms,
            "adapter_mining_nprobe": args.adapter_mining_nprobe,
            "adapter_neg_kfinal": args.adapter_neg_kfinal,
            "adapt_epochs": args.adapt_epochs,
            "adapt_lr": args.adapt_lr,
            "adapt_subset": args.adapt_subset,
            "adapt_qbatch": args.adapt_qbatch,
            "tau": args.tau,
            "beta": args.beta,
            "gamma": args.gamma,
            "grad_clip": args.grad_clip,
        },
        "timing_sec": {
            "load_total": t_load,
        },
    }
    meta_json = os.path.join(job_out_dir, "job_meta.json")
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ saved:", summary_csv, flush=True)
    print("✅ saved:", agg_csv, flush=True)
    print("✅ saved:", short_csv, flush=True)
    print("✅ saved:", meta_json, flush=True)


if __name__ == "__main__":
    main()
