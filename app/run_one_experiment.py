# run_one_experiment.py
#!/usr/bin/env python3
import os, argparse, json, csv, time
from datetime import datetime
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

# Always log these Ks (plus anything user passes)
METRIC_KS_CANON = [1, 5, 10, 25, 50, 100]


def as_tuple(s: str):
    return tuple(int(x) for x in s.split(","))


def parse_k_list(s: str):
    ks = [int(x) for x in s.split(",") if x.strip()]
    return sorted(set(ks))


def metric_cols_for_csv():
    cols = ["S1", "S2", "S3", "nprobe", "K2"]
    cols += [f"H@{k}" for k in METRIC_KS_CANON]
    cols += [f"R@{k}" for k in METRIC_KS_CANON]
    cols += [f"MRR@{k}" for k in METRIC_KS_CANON]
    return cols


def write_single_row_csv(path, rowdict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = metric_cols_for_csv()
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({k: rowdict.get(k, None) for k in cols})
    print("Saved:", path, flush=True)


def append_summary_csv(path, rowdict):
    """
    Append one row into a stable per-(dataset,encoder) summary CSV.
    Uses file-locking to tolerate concurrent jobs on the same PVC.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    base_cols = [
        "timestamp",
        "exp_name",
        "dataset",
        "encoder",
        "encoder_tag",
        "mode",
        "adapter",
        "trial_slug",
        "bits_sq",
        "nlist",
        "nprobe",
        "k2",
        "kfinals",
        "ms",
        "alphas",
        "qrels_split",
        "passages",
        "queries",
        "pos_qrels",
        "csv_path",
        "meta_path",
        "metrics_json",
        "timing_load_s",
        "timing_pipeline_s",
        "timing_eval_s",
    ]
    metric_cols = metric_cols_for_csv()
    cols = base_cols + metric_cols

    try:
        import fcntl
        lock_ok = True
    except Exception:
        lock_ok = False

    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        if lock_ok:
            fcntl.flock(f, fcntl.LOCK_EX)

        w = csv.DictWriter(f, fieldnames=cols)
        if not file_exists:
            w.writeheader()
        w.writerow({k: rowdict.get(k, None) for k in cols})

        if lock_ok:
            fcntl.flock(f, fcntl.LOCK_UN)

    print("✅ appended summary:", path, flush=True)


def main():
    p = argparse.ArgumentParser("Run ONE experiment: IVF or DUAL, with/without adapter.")
    p.add_argument("--dataset", required=True)                # nq, hotpotqa, msmarco, ...
    p.add_argument("--encoder", required=True)                # all-MiniLM-L6-v2, sentence-transformers/all-mpnet-base-v2, ...
    p.add_argument("--mode", choices=["ivf", "dual"], required=True)
    p.add_argument("--adapter", choices=["on", "off"], default="off")

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
    p.add_argument("--kfinals", default="10,25,50,100")   # unioned with METRIC_KS_CANON
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
    p.add_argument("--trial", default="adapter")  # slug base (namespaced below)

    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    encoder_tag = args.encoder.replace("/", "_")

    # Always evaluate canonical Ks, even if user passes a smaller list
    kfinals_user = parse_k_list(args.kfinals)
    kfinals_eval = sorted(set(kfinals_user) | set(METRIC_KS_CANON))

    ms = as_tuple(args.ms)
    alphas = as_tuple(args.alphas)

    # Namespaced dirs: separate per dataset + encoder
    inter_dir = os.path.join(args.intermediate_root, args.dataset, encoder_tag)
    res_dir = os.path.join(args.run_root, args.dataset, encoder_tag, "results")
    os.makedirs(inter_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    embed_path = os.path.join(inter_dir, f"passage_embeddings_{args.dataset}_{encoder_tag}.npy")
    qembed_path = os.path.join(inter_dir, f"query_embeddings_{args.dataset}_{encoder_tag}.npy")

    # Load (build caches if missing)
    t0 = time.time()
    embeddings, queries_emb, passage_ids, query_ids, query_to_gt, meta = load_beir_jsonl(
        data_root=args.data_root,
        dataset=args.dataset,
        encoder_name=args.encoder,
        device=device,
        embed_path=embed_path,
        query_embed_path=qembed_path,
        normalize=True,
    )
    t_load = time.time() - t0

    # Build baseline pipeline
    t1 = time.time()
    pipe_base = build_pipeline_baseline(
        slug=f"base_{args.dataset}_{encoder_tag}",
        embeddings=embeddings,
        queries_emb_sample=queries_emb,
        passage_ids_sample=passage_ids,
        bits_sq=args.bits_sq,
        nlist=args.nlist,
    )
    t_pipe = time.time() - t1

    # Choose pipeline (adapter or not)
    pipe = pipe_base
    trial_slug = "no_adapter"

    if args.adapter == "on":
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
                MS_INFER=ms,
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

    # Unique exp name (avoid overwrites)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = (
        f"{args.dataset}__{encoder_tag}__{args.mode}__adapter{args.adapter}"
        f"__m{ms[0]}_{ms[1]}_{ms[2]}__np{args.nprobe}__{ts}"
    )
    out_csv = os.path.join(res_dir, f"{exp_name}.csv")
    out_meta = os.path.join(res_dir, f"{exp_name}.meta.json")
    summary_csv = os.path.join(res_dir, "results_summary.csv")

    # Evaluate ONE mode
    t2 = time.time()
    if args.mode == "ivf":
        _, row = quick_eval_for_pipeline_ivf(
            pipeline_data=pipe,
            trial_slug=trial_slug,
            SELECT_NPROBE=args.nprobe,
            ALPHAS_INFER=alphas,
            MS_INFER=ms,
            K2_FIXED=args.k2,
            K_FINALS=kfinals_eval,
            results_dir=res_dir,
            query_ids_sample=query_ids,
            query_to_gt=query_to_gt,
        )
        row_out = dict(row)
        row_out.update({"S1": "ivf", "S2": "ivf", "S3": "ivf", "nprobe": args.nprobe, "K2": args.k2})
    else:
        _, row = quick_eval_for_pipeline_dual(
            pipeline_data=pipe,
            trial_slug=trial_slug,
            SELECT_NPROBE=args.nprobe,
            ALPHAS_INFER=alphas,
            MS_INFER=ms,
            K2_FIXED=args.k2,
            K_FINALS=kfinals_eval,
            results_dir=res_dir,
            query_ids_sample=query_ids,
            query_to_gt=query_to_gt,
        )
        row_out = dict(row)
        row_out.update({"S1": "dual", "S2": "dual", "S3": "dual", "nprobe": args.nprobe, "K2": args.k2})
    t_eval = time.time() - t2

    # Write per-run metric CSV
    write_single_row_csv(out_csv, row_out)

    # Write metadata (audit)
    meta_out = {
        "timestamp": ts,
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
        "kfinals_eval": kfinals_eval,
        "kfinals_user": kfinals_user,
        "ms": ms,
        "alphas": alphas,
        "paths": {
            "embed_path": embed_path,
            "qembed_path": qembed_path,
            "inter_dir": inter_dir,
            "res_dir": res_dir,
            "csv": out_csv,
            "summary_csv": summary_csv,
        },
        "dataset_meta": meta,
        "timings_sec": {"load": t_load, "pipeline": t_pipe, "eval": t_eval},
    }
    with open(out_meta, "w") as f:
        json.dump(meta_out, f, indent=2)

    # Append summary row
    metrics_json = {k: v for k, v in row_out.items() if isinstance(k, str) and "@" in k}
    summary_row = {
        "timestamp": ts,
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
        "kfinals": ",".join(map(str, kfinals_eval)),
        "ms": ",".join(map(str, ms)),
        "alphas": ",".join(map(str, alphas)),
        "qrels_split": meta.get("qrels_split"),
        "passages": meta.get("counts", {}).get("passages"),
        "queries": meta.get("counts", {}).get("queries"),
        "pos_qrels": meta.get("counts", {}).get("pos_qrels"),
        "csv_path": out_csv,
        "meta_path": out_meta,
        "metrics_json": json.dumps(metrics_json),
        "timing_load_s": t_load,
        "timing_pipeline_s": t_pipe,
        "timing_eval_s": t_eval,
    }
    summary_row.update(row_out)  # adds H/R/MRR keys, etc.
    append_summary_csv(summary_csv, summary_row)

    print("✅ meta:", out_meta, flush=True)
    print("✅ csv :", out_csv, flush=True)
    print("✅ summary:", summary_csv, flush=True)


if __name__ == "__main__":
    main()