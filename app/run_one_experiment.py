#!/usr/bin/env python3
import os, argparse, json, csv
import numpy as np
import torch

from beir_jsonl_loader import load_beir_jsonl
from utilis_dbam_v3 import (
    build_pipeline_baseline,
    load_or_build_pipeline_for_adapter,
    train_W_param,
    quick_eval_for_pipeline_ivf,
    quick_eval_for_pipeline_dual,
)

def as_tuple(s: str):
    return tuple(int(x) for x in s.split(","))

def write_single_row_csv(path, rowdict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [
        "S1","S2","S3","nprobe","K2",
        "H@10","H@25","H@50","H@100",
        "R@10","R@25","R@50","R@100",
        "MRR@10","MRR@25","MRR@50","MRR@100",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({k: rowdict.get(k, None) for k in cols})
    print("Saved:", path, flush=True)

def main():
    p = argparse.ArgumentParser("Run ONE experiment: IVF or DUAL, with/without adapter.")
    p.add_argument("--dataset", required=True)                # nq, hotpotqa, ...
    p.add_argument("--encoder", required=True)                # all-MiniLM-L6-v2, all-mpnet-base-v2, ...
    p.add_argument("--mode", choices=["ivf", "dual"], required=True)
    p.add_argument("--adapter", choices=["on","off"], default="off")

    # roots (PVC-friendly)
    p.add_argument("--work_dir", default=os.getenv("WORK_DIR", "/mnt/work"))
    p.add_argument("--data_root", default=os.getenv("DATA_ROOT", "/mnt/work/VectorDB_MICRO/datasets/semantic"))
    p.add_argument("--intermediate_root", default=os.getenv("INTERMEDIATE_ROOT", "/mnt/work/VectorDB_MICRO/intermediate_data"))
    p.add_argument("--run_root", default=os.getenv("RUN_ROOT", "/mnt/work/VectorDB_MICRO/DBAM/runs"))

    # knobs
    p.add_argument("--bits_sq", type=int, default=4)
    p.add_argument("--nlist", type=int, default=1024)
    p.add_argument("--nprobe", type=int, default=32)
    p.add_argument("--k2", type=int, default=1000)
    p.add_argument("--kfinals", default="10,25,50,100")
    p.add_argument("--ms", default="2,4,2")
    p.add_argument("--alphas", default="2,2,2")

    # adapter train knobs (only used if adapter missing)
    p.add_argument("--adapt_epochs", type=int, default=5)
    p.add_argument("--adapt_lr", type=float, default=5e-4)
    p.add_argument("--adapt_subset", type=int, default=50000)
    p.add_argument("--adapt_qbatch", type=int, default=64)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=6.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--trial", default="adapter")  # trial slug base, we’ll namespace it below

    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    encoder_tag = args.encoder.replace("/", "_")
    kfinals = [int(x) for x in args.kfinals.split(",")]
    ms = as_tuple(args.ms)
    alphas = as_tuple(args.alphas)

    # Namespaced dirs
    inter_dir = os.path.join(args.intermediate_root, args.dataset, encoder_tag)
    res_dir = os.path.join(args.run_root, args.dataset, encoder_tag, "results")
    os.makedirs(inter_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    embed_path = os.path.join(inter_dir, f"passage_embeddings_{args.dataset}_{encoder_tag}.npy")
    qembed_path = os.path.join(inter_dir, f"query_embeddings_{args.dataset}_{encoder_tag}.npy")

    # Load (build caches if missing)
    embeddings, queries_emb, passage_ids, query_ids, query_to_gt, meta = load_beir_jsonl(
        data_root=args.data_root,
        dataset=args.dataset,
        encoder_name=args.encoder,
        device=device,
        embed_path=embed_path,
        query_embed_path=qembed_path,
    )

    # Build baseline pipeline
    pipe_base = build_pipeline_baseline(
        slug=f"base_{args.dataset}_{encoder_tag}",
        embeddings=embeddings,
        queries_emb_sample=queries_emb,
        passage_ids_sample=passage_ids,
        bits_sq=args.bits_sq,
        nlist=args.nlist,
    )

    # Choose pipeline (adapter or not)
    pipe = pipe_base
    trial_slug = "no_adapter"
    if args.adapter == "on":
        # encoder-aware adapter slug to prevent 384/768 collisions
        trial_slug = f"{args.trial}_{args.dataset}_{encoder_tag}"
        W_path = os.path.join(inter_dir, f"W_{trial_slug}.npy")
        ckpt_path = os.path.join(inter_dir, f"W_{trial_slug}_adapter.pt")

        if not os.path.exists(ckpt_path):
            print(f"⚙️ adapter missing -> training: {ckpt_path}", flush=True)
            train_W_param(
                emb_np=embeddings,
                que_np=queries_emb,
                d_out=embeddings.shape[1],
                save_path=W_path,
                passage_ids_sample=passage_ids,
                query_ids_sample=query_ids,
                query_to_gt=query_to_gt,
                device=device,
                epochs=args.adapt_epochs,
                lr=args.adapt_lr,
                subset=args.adapt_subset,
                q_batch=args.adapt_qbatch,
                tau=args.tau,
                beta=args.beta,
                gamma=args.gamma,
                grad_clip=args.grad_clip,
                pipeline_data=pipe_base,
                ALPHAS_INFER=alphas,
                MS_INFER=ms,              # mining config: use the same ms by default
                SELECT_NPROBE=args.nprobe,
                K_final=10,
            )

        pipe, _ = load_or_build_pipeline_for_adapter(
            trial_slug=trial_slug,
            embeddings=embeddings,
            queries_emb_sample=queries_emb,
            passage_ids_sample=passage_ids,
            nlist=args.nlist,
            bits_sq=args.bits_sq,
            intermediate_dir=inter_dir,
            device=device,
            fallback_to_baseline=False,
        )

    # Evaluate ONE mode
    exp_name = f"{args.dataset}__{encoder_tag}__{args.mode}__adapter{args.adapter}__m{ms[0]}_{ms[1]}_{ms[2]}__np{args.nprobe}"
    out_csv = os.path.join(res_dir, f"{exp_name}.csv")
    out_meta = os.path.join(res_dir, f"{exp_name}.meta.json")

    if args.mode == "ivf":
        _, row = quick_eval_for_pipeline_ivf(
            pipeline_data=pipe,
            trial_slug=trial_slug,
            SELECT_NPROBE=args.nprobe,
            ALPHAS_INFER=alphas,
            MS_INFER=ms,
            K2_FIXED=args.k2,
            K_FINALS=kfinals,
            results_dir=res_dir,
            query_ids_sample=query_ids,
            query_to_gt=query_to_gt,
        )
        row_out = dict(row)
        row_out.update({"S1":"ivf","S2":"ivf","S3":"ivf","nprobe":args.nprobe,"K2":args.k2})
        write_single_row_csv(out_csv, row_out)

    else:  # dual
        _, row = quick_eval_for_pipeline_dual(
            pipeline_data=pipe,
            trial_slug=trial_slug,
            SELECT_NPROBE=args.nprobe,
            ALPHAS_INFER=alphas,
            MS_INFER=ms,
            K2_FIXED=args.k2,
            K_FINALS=kfinals,
            results_dir=res_dir,
            query_ids_sample=query_ids,
            query_to_gt=query_to_gt,
        )
        row_out = dict(row)
        row_out.update({"S1":"dual","S2":"dual","S3":"dual","nprobe":args.nprobe,"K2":args.k2})
        write_single_row_csv(out_csv, row_out)

    # Write metadata (so jobs can be audited later)
    meta_out = {
        "exp_name": exp_name,
        "dataset": args.dataset,
        "encoder": args.encoder,
        "encoder_tag": encoder_tag,
        "mode": args.mode,
        "adapter": args.adapter,
        "trial_slug": trial_slug,
        "bits_sq": args.bits_sq,
        "nlist": args.nlist,
        "nprobe": args.nprobe,
        "k2": args.k2,
        "kfinals": kfinals,
        "ms": ms,
        "alphas": alphas,
        "paths": {"embed_path": embed_path, "qembed_path": qembed_path, "inter_dir": inter_dir, "res_dir": res_dir},
        "dataset_meta": meta,
    }
    with open(out_meta, "w") as f:
        json.dump(meta_out, f, indent=2)
    print("✅ meta:", out_meta, flush=True)
    print("✅ csv :", out_csv, flush=True)

if __name__ == "__main__":
    main()