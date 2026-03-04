# nq_cli.py
import os
import argparse
import csv
import numpy as np
import torch
import pandas as pd

from nq_loader import load_all
from utilis_dbam_v3 import (
    build_pipeline_baseline,
    run_and_evaluate,
    quick_eval_for_pipeline_ivf,
    quick_eval_for_pipeline_direct,
    quick_eval_for_pipeline_dual,
    load_or_build_pipeline_for_adapter,
)

# --------------------- small helpers ---------------------


def as_tuple_xyz(s):
    return tuple(int(x) for x in s.split(","))


def write_single_row_csv(path, rowdict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [
        "S1", "S2", "S3", "nprobe", "K2",
        "H@10", "H@25", "H@50", "H@100",
        "R@10", "R@25", "R@50", "R@100",
        "MRR@10", "MRR@25", "MRR@50", "MRR@100",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({k: rowdict.get(k, None) for k in cols})
    print("Saved:", path, flush=True)


def default_slug_from_args(a):
    # Mirrors your training slug; used when user didn’t pass --slug
    return (
        f"tau{a.tau}_b{a.beta}_c{a.cands}_{a.teacher}_lr{a.lr}"
        f"_e{a.epochs}_s{a.subset}"
    ).replace(".", "")


def adapter_paths(slug, inter_dir):
    base = os.path.join(inter_dir, f"W_{slug}")
    return base + ".npy", base + "_adapter.pt"


def ensure_adapter(
    a,
    device,
    inter_dir,
    embeddings,
    queries_emb,
    passage_ids,
    nlist,
    bits_sq,
    query_ids,
    query_to_gt,
    ALPHAS,
    MS_INFER,
):
    """
    Make sure adapter checkpoint exists for slug.
    If not, run a minimal training pass with your train_W_param.
    """
    from utilis_dbam_v3 import train_W_param, load_or_build_pipeline_for_adapter

    W_path, adapter_pt = adapter_paths(a.slug, inter_dir)

    if os.path.exists(adapter_pt):
        print(f"🔄 Adapter exists: {adapter_pt}", flush=True)
        return

    print(f"⚙️ No adapter for slug '{a.slug}'. Training now…", flush=True)

    # Build a fixed baseline pipeline (as your notebook did)
    pipe_fixed, _ = load_or_build_pipeline_for_adapter(
        trial_slug="baseline_fixed",
        embeddings=embeddings,
        queries_emb_sample=queries_emb,
        passage_ids_sample=passage_ids,
        nlist=nlist,
        bits_sq=bits_sq,
        intermediate_dir=inter_dir,
        device=device,
        fallback_to_baseline=True,
    )

    # Train (same defaults you used)
    W_np, logs = train_W_param(
        emb_np=embeddings,
        que_np=queries_emb,
        d_out=embeddings.shape[1],
        save_path=W_path,
        passage_ids_sample=passage_ids,
        query_ids_sample=query_ids,
        query_to_gt=query_to_gt,
        device=device,
        epochs=a.epochs,
        lr=a.lr,
        subset=a.subset,
        q_batch=a.q_batch,
        tau=a.tau,
        beta=a.beta,
        gamma=1.0,
        grad_clip=1.0,
        pipeline_data=pipe_fixed,
        ALPHAS_INFER=ALPHAS,
        MS_INFER=MS_INFER,
        SELECT_NPROBE=a.select_nprobe,
        K_final=10,
    )

    # train_W_param writes both adapter pt and identity W
    _, adapter_pt2 = adapter_paths(a.slug, inter_dir)
    if not os.path.exists(adapter_pt2):
        raise FileNotFoundError(
            f"Adapter training finished but not found: {adapter_pt2}"
        )
    print(f"✅ Adapter ready: {adapter_pt2}", flush=True)


# --------------------- CLI ---------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Run NQ/HotpotQA tests (IVF/DBAM with/without adapter)."
    )

    p.add_argument(
        "--dataset",
        choices=["beir_nq", "beir_nq_mpnet", "hotpotqa", "hotpotqa_mpnet"],
        required=True,
    )

    # paths (PVC-friendly defaults)
    p.add_argument("--work_dir", default=os.getenv("WORK_DIR", "/mnt/work"))
    p.add_argument("--data_root", default=os.getenv("DATA_ROOT", "/mnt/work/datasets"))
    p.add_argument("--run_root", default=os.getenv("RUN_ROOT", "/mnt/work/runs/nq"))
    p.add_argument(
        "--intermediate_root",
        default=None,
        help="If set, overrides run_root/intermediate",
    )

    # common knobs
    p.add_argument("--bits_sq", type=int, default=4)
    p.add_argument("--nlist", type=int, default=2048)
    p.add_argument("--select_nprobe", type=int, default=16)
    p.add_argument("--k2_fixed", type=int, default=1000)
    p.add_argument("--kfinal", type=int, nargs="+", default=[10, 25, 50, 100])

    p.add_argument("--ms_infer", default="1,8,1")  # for DBAM tests
    p.add_argument("--alphas", default="2,2,2")

    # adapter sweep bits (used for training if needed)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=6.0)
    p.add_argument("--cands", type=int, default=2048)
    p.add_argument("--teacher", default="cos")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--subset", type=int, default=50000)
    p.add_argument("--q_batch", type=int, default=64)

    # adapter slug: if not provided, auto-derived from hyperparams
    p.add_argument(
        "--slug",
        default="auto",
        help="Adapter slug. If 'auto', derived from hparams.",
    )

    # optional explicit output path
    p.add_argument("--out_csv", default=None)

    # exact operation to run
    p.add_argument(
        "--mode",
        choices=[
            # 1) IVF FP32 w/o adapter
            "ivf_fp32",
            # 2) D-BAM (DIRECT,DIRECT,DIRECT) w/o adapter
            "dbam_direct_baseline",
            # 3) D2-BAM (DUAL,DUAL,DUAL) w/o adapter
            "dbam_dual_baseline",
            # 4) IVF FP32 w/ adapter
            "ivf_fp32_adapter",
            # 5) D-BAM w/ adapter
            "dbam_direct_adapter",
            # 6) D2-BAM w/ adapter
            "dbam_dual_adapter",
            # (utility) train adapter only
            "train_adapter",
        ],
        required=True,
    )

    return p.parse_args()


# --------------------- main ---------------------


def main():
    a = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    # Directories
    os.makedirs(a.run_root, exist_ok=True)
    results_dir = os.path.join(a.run_root, "results")
    inter_dir = a.intermediate_root or os.path.join(a.run_root, "intermediate")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(inter_dir, exist_ok=True)

    # Dataset layout
    data_dir = os.path.join(a.data_root, a.dataset)
    corpus_file = os.path.join(data_dir, "corpus.jsonl")
    queries_file = os.path.join(data_dir, "queries.jsonl")
    qrels_file = os.path.join(data_dir, "qrels", "test.tsv")

    # ---------- Embed path selection ----------
    embed_path = None
    if a.dataset == "beir_nq":
        embed_path = os.path.join(data_dir, "sample_passage_embeddings_nq.npy")
    elif a.dataset == "beir_nq_mpnet":
        # mpnet precomputed embeddings for BEIR NQ
        embed_path = os.path.join(data_dir, "sample_passage_embeddings_nq.npy")
    elif a.dataset == "hotpotqa":
        embed_path = os.path.join(data_dir, "sample_passage_embeddings_hotpotqa.npy")
    elif a.dataset == "hotpotqa_mpnet":
        # mpnet precomputed embeddings for HotpotQA (once you add it)
        embed_path = os.path.join(data_dir, "sample_passage_embeddings_hotpotqa.npy")

    # ---------- Encoder selection ----------
    # Priority:
    #   1) ENCODER_NAME env (from YAML)
    #   2) dataset-based default
    encoder_name = os.getenv("ENCODER_NAME")
    if not encoder_name:
        if "mpnet" in a.dataset:
            encoder_name = "sentence-transformers/all-mpnet-base-v2"
        else:
            encoder_name = "all-MiniLM-L6-v2"

    print(f"Using encoder: {encoder_name}", flush=True)
    if embed_path:
        print(f"Using passage embedding cache (if exists): {embed_path}", flush=True)

    # Load / encode
    embeddings, queries_emb, passage_ids, query_ids, query_to_gt = load_all(
        corpus_file,
        queries_file,
        qrels_file,
        device,
        embed_path=embed_path,
        encoder_name=encoder_name,
    )

    # Common config pieces
    MS_INFER = as_tuple_xyz(a.ms_infer)  # e.g. (1,8,1)
    ALPHAS = as_tuple_xyz(a.alphas)      # e.g. (2,2,2)
    K_FINALS = a.kfinal

    # If slug "auto", derive from hparams for deterministic caching
    if a.slug == "auto":
        a.slug = default_slug_from_args(a)
    print(f"Using adapter slug: {a.slug}", flush=True)

    # ------------------ modes ------------------

    # 1) IVF FP32 w/o adapter
    if a.mode == "ivf_fp32":
        pipe = build_pipeline_baseline(
            slug="no_adapter",
            embeddings=embeddings,
            queries_emb_sample=queries_emb,
            passage_ids_sample=passage_ids,
            bits_sq=a.bits_sq,
            nlist=a.nlist,
        )
        score, row = quick_eval_for_pipeline_ivf(
            pipeline_data=pipe,
            trial_slug="no_adapter",
            SELECT_NPROBE=a.select_nprobe,
            ALPHAS_INFER=ALPHAS,
            MS_INFER=MS_INFER,
            K2_FIXED=a.k2_fixed,
            K_FINALS=K_FINALS,
            results_dir=results_dir,
            query_ids_sample=query_ids,
            query_to_gt=query_to_gt,
        )
        row_out = dict(row)
        row_out.update(
            {
                "S1": "ivf",
                "S2": "ivf",
                "S3": "ivf",
                "nprobe": a.select_nprobe,
                "K2": a.k2_fixed,
            }
        )
        out = a.out_csv or os.path.join(results_dir, "ivf_fp32_baseline.csv")
        write_single_row_csv(out, row_out)
        return

    # 2) D-BAM DIRECT w/o adapter
    if a.mode == "dbam_direct_baseline":
        pipe = build_pipeline_baseline(
            slug="no_adapter",
            embeddings=embeddings,
            queries_emb_sample=queries_emb,
            passage_ids_sample=passage_ids,
            bits_sq=a.bits_sq,
            nlist=a.nlist,
        )
        score, row = quick_eval_for_pipeline_direct(
            pipeline_data=pipe,
            trial_slug="no_adapter",
            SELECT_NPROBE=a.select_nprobe,
            ALPHAS_INFER=ALPHAS,
            MS_INFER=MS_INFER,
            K2_FIXED=a.k2_fixed,
            K_FINALS=K_FINALS,
            results_dir=results_dir,
            query_ids_sample=query_ids,
            query_to_gt=query_to_gt,
        )
        row_out = dict(row)
        row_out.update(
            {
                "S1": "direct",
                "S2": "direct",
                "S3": "direct",
                "nprobe": a.select_nprobe,
                "K2": a.k2_fixed,
            }
        )
        ms_tag = f"m{MS_INFER[0]}-{MS_INFER[1]}-{MS_INFER[2]}"
        out = (
            a.out_csv
            or os.path.join(results_dir, f"dbam_direct_baseline_{ms_tag}.csv")
        )
        write_single_row_csv(out, row_out)
        return

    # 3) D2-BAM DUAL w/o adapter
    if a.mode == "dbam_dual_baseline":
        pipe = build_pipeline_baseline(
            slug="no_adapter",
            embeddings=embeddings,
            queries_emb_sample=queries_emb,
            passage_ids_sample=passage_ids,
            bits_sq=a.bits_sq,
            nlist=a.nlist,
        )
        score, row = quick_eval_for_pipeline_dual(
            pipeline_data=pipe,
            trial_slug="no_adapter",
            SELECT_NPROBE=a.select_nprobe,
            ALPHAS_INFER=ALPHAS,
            MS_INFER=MS_INFER,
            K2_FIXED=a.k2_fixed,
            K_FINALS=K_FINALS,
            results_dir=results_dir,
            query_ids_sample=query_ids,
            query_to_gt=query_to_gt,
        )
        row_out = dict(row)
        row_out.update(
            {
                "S1": "dual",
                "S2": "dual",
                "S3": "dual",
                "nprobe": a.select_nprobe,
                "K2": a.k2_fixed,
            }
        )
        ms_tag = f"m{MS_INFER[0]}-{MS_INFER[1]}-{MS_INFER[2]}"
        out = (
            a.out_csv
            or os.path.join(results_dir, f"dbam_dual_baseline_{ms_tag}.csv")
        )
        write_single_row_csv(out, row_out)
        return

    # 4) IVF FP32 WITH adapter (auto-trains if missing)
    if a.mode == "ivf_fp32_adapter":
        ensure_adapter(
            a,
            device,
            inter_dir,
            embeddings,
            queries_emb,
            passage_ids,
            a.nlist,
            a.bits_sq,
            query_ids,
            query_to_gt,
            ALPHAS,
            MS_INFER,
        )
        pipe_adapt, _ = load_or_build_pipeline_for_adapter(
            trial_slug=a.slug,
            embeddings=embeddings,
            queries_emb_sample=queries_emb,
            passage_ids_sample=passage_ids,
            nlist=a.nlist,
            bits_sq=a.bits_sq,
            intermediate_dir=inter_dir,
            device=device,
        )
        score, row = quick_eval_for_pipeline_ivf(
            pipeline_data=pipe_adapt,
            trial_slug=a.slug,
            SELECT_NPROBE=a.select_nprobe,
            ALPHAS_INFER=ALPHAS,
            MS_INFER=MS_INFER,
            K2_FIXED=a.k2_fixed,
            K_FINALS=K_FINALS,
            results_dir=results_dir,
            query_ids_sample=query_ids,
            query_to_gt=query_to_gt,
        )
        row_out = dict(row)
        row_out.update(
            {
                "S1": "ivf",
                "S2": "ivf",
                "S3": "ivf",
                "nprobe": a.select_nprobe,
                "K2": a.k2_fixed,
            }
        )
        out = a.out_csv or os.path.join(results_dir, "ivf_fp32_with_adapter.csv")
        write_single_row_csv(out, row_out)
        return

    # 5) D-BAM DIRECT WITH adapter (auto-trains if missing)
    if a.mode == "dbam_direct_adapter":
        ensure_adapter(
            a,
            device,
            inter_dir,
            embeddings,
            queries_emb,
            passage_ids,
            a.nlist,
            a.bits_sq,
            query_ids,
            query_to_gt,
            ALPHAS,
            MS_INFER,
        )
        pipe_adapt, _ = load_or_build_pipeline_for_adapter(
            trial_slug=a.slug,
            embeddings=embeddings,
            queries_emb_sample=queries_emb,
            passage_ids_sample=passage_ids,
            nlist=a.nlist,
            bits_sq=a.bits_sq,
            intermediate_dir=inter_dir,
            device=device,
        )
        score, row = quick_eval_for_pipeline_direct(
            pipeline_data=pipe_adapt,
            trial_slug=a.slug,
            SELECT_NPROBE=a.select_nprobe,
            ALPHAS_INFER=ALPHAS,
            MS_INFER=MS_INFER,
            K2_FIXED=a.k2_fixed,
            K_FINALS=K_FINALS,
            results_dir=results_dir,
            query_ids_sample=query_ids,
            query_to_gt=query_to_gt,
        )
        row_out = dict(row)
        row_out.update(
            {
                "S1": "direct",
                "S2": "direct",
                "S3": "direct",
                "nprobe": a.select_nprobe,
                "K2": a.k2_fixed,
            }
        )
        ms_tag = f"m{MS_INFER[0]}-{MS_INFER[1]}-{MS_INFER[2]}"
        out = (
            a.out_csv
            or os.path.join(results_dir, f"dbam_direct_with_adapter_{ms_tag}.csv")
        )
        write_single_row_csv(out, row_out)
        return

    # 6) D2-BAM DUAL WITH adapter (auto-trains if missing)
    if a.mode == "dbam_dual_adapter":
        ensure_adapter(
            a,
            device,
            inter_dir,
            embeddings,
            queries_emb,
            passage_ids,
            a.nlist,
            a.bits_sq,
            query_ids,
            query_to_gt,
            ALPHAS,
            MS_INFER,
        )
        pipe_adapt, _ = load_or_build_pipeline_for_adapter(
            trial_slug=a.slug,
            embeddings=embeddings,
            queries_emb_sample=queries_emb,
            passage_ids_sample=passage_ids,
            nlist=a.nlist,
            bits_sq=a.bits_sq,
            intermediate_dir=inter_dir,
            device=device,
        )
        score, row = quick_eval_for_pipeline_dual(
            pipeline_data=pipe_adapt,
            trial_slug=a.slug,
            SELECT_NPROBE=a.select_nprobe,
            ALPHAS_INFER=ALPHAS,
            MS_INFER=MS_INFER,
            K2_FIXED=a.k2_fixed,
            K_FINALS=K_FINALS,
            results_dir=results_dir,
            query_ids_sample=query_ids,
            query_to_gt=query_to_gt,
        )
        row_out = dict(row)
        row_out.update(
            {
                "S1": "dual",
                "S2": "dual",
                "S3": "dual",
                "nprobe": a.select_nprobe,
                "K2": a.k2_fixed,
            }
        )
        ms_tag = f"m{MS_INFER[0]}-{MS_INFER[1]}-{MS_INFER[2]}"
        out = (
            a.out_csv
            or os.path.join(results_dir, f"dbam_dual_with_adapter_{ms_tag}.csv")
        )
        write_single_row_csv(out, row_out)
        return

    # utility: train adapter only (if you want to pre-bake)
    if a.mode == "train_adapter":
        if a.slug == "auto":
            a.slug = default_slug_from_args(a)
        ensure_adapter(
            a,
            device,
            inter_dir,
            embeddings,
            queries_emb,
            passage_ids,
            a.nlist,
            a.bits_sq,
            query_ids,
            query_to_gt,
            ALPHAS,
            MS_INFER,
        )
        return


if __name__ == "__main__":
    main()
