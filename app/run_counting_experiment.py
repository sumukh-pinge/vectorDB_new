#!/usr/bin/env python3
import os, argparse, json, csv, time
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from beir_jsonl_loader import load_beir_jsonl
from utilis_dbam_v3 import (
    build_pipeline_baseline,
    load_or_build_pipeline_for_adapter,
    train_W_param,
)

# Try importing from util; fallback if needed
try:
    from utilis_dbam_v3 import dbam_dual
except Exception:
    def dbam_dual(q_code, base_q, base_dual, alpha, m, g, levels):
        bg = base_q.reshape(-1, g, m)
        bd = base_dual.reshape(-1, g, m)
        qg = q_code.reshape(g, m)
        ub_orig = np.all(bg <= (qg + alpha), axis=2)
        ub_dual = np.all(bd <= ((levels - 1 - qg) + alpha), axis=2)
        return ub_orig.sum(axis=1) + ub_dual.sum(axis=1)

try:
    from utilis_dbam_v3 import dbam_direct
except Exception:
    def dbam_direct(q_code, base_q, alpha, m, g):
        bg = base_q.reshape(-1, g, m)
        qg = q_code.reshape(g, m)
        ub = np.all(bg <= (qg + alpha), axis=2)
        lb = np.any(bg >= (qg - alpha), axis=2)
        return ub.sum(axis=1) + lb.sum(axis=1)

def as_tuple(s: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))

def safe_encoder_tag(encoder: str) -> str:
    return encoder.replace("/", "_")

def ms_tag(ms: Tuple[int,int,int]) -> str:
    return f"m{ms[0]}_{ms[1]}_{ms[2]}"

def stages_tag(stages: Tuple[str,str,str]) -> str:
    return "_".join(stages)

def _hist_stats(hist: np.ndarray) -> Dict[str, Any]:
    total = int(hist.sum())
    if total == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None, "p90": None, "p99": None}
    nz = np.nonzero(hist)[0]
    vmin, vmax = int(nz[0]), int(nz[-1])
    values = np.arange(hist.size, dtype=np.float64)
    mean = float((values * hist).sum() / total)
    cdf = np.cumsum(hist)

    def q(p: float) -> int:
        thresh = int(np.ceil(p * total))
        return int(np.searchsorted(cdf, thresh, side="left"))

    return {
        "n": total,
        "min": vmin,
        "max": vmax,
        "mean": mean,
        "median": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
    }

def _update_hist(hist: np.ndarray, scores: np.ndarray):
    if scores.size == 0:
        return
    bc = np.bincount(scores.astype(np.int64), minlength=hist.size)
    hist[:bc.size] += bc

def append_csv_locked(path: str, row: Dict[str,Any], cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import fcntl
        lock_ok = True
    except Exception:
        lock_ok = False

    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        if lock_ok:
            fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, None) for k in cols})
        if lock_ok:
            fcntl.flock(f, fcntl.LOCK_UN)

def score_fn(method: str):
    if method == "dual":
        return "dual"
    if method == "direct":
        return "direct"
    raise ValueError(f"Unsupported stage method: {method} (supported: dual,direct)")

def compute_scores(method: str, q_code, base_q, base_dual, alpha, m, g, levels):
    if method == "dual":
        return dbam_dual(q_code, base_q, base_dual, alpha, m, g, levels).astype(np.int32)
    else:
        return dbam_direct(q_code, base_q, alpha, m, g).astype(np.int32)

def main():
    p = argparse.ArgumentParser("Counting / score-distribution (generic stages, YAML-controlled).")
    p.add_argument("--dataset", required=True)
    p.add_argument("--encoder", required=True)

    p.add_argument("--mode", default="counting")  # just for labeling
    p.add_argument("--adapter", choices=["on","off"], default="off")

    p.add_argument("--bits_sq", type=int, default=4)
    p.add_argument("--nlist", type=int, default=1024)
    p.add_argument("--nprobe", type=int, default=64)
    p.add_argument("--k2", type=int, default=1000)
    p.add_argument("--alphas", default="2,2,2")
    p.add_argument("--ms", required=True)              # e.g., "2,4,2" chosen by YAML
    p.add_argument("--stages", default="dual,dual,dual")  # YAML controls this too

    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max_queries", type=int, default=0)

    # roots
    p.add_argument("--data_root", default=os.getenv("DATA_ROOT", "/mnt/work/VectorDB_MICRO/datasets/semantic"))
    p.add_argument("--intermediate_root", default=os.getenv("INTERMEDIATE_ROOT", "/mnt/work/VectorDB_MICRO/intermediate_data"))
    p.add_argument("--run_root", default=os.getenv("RUN_ROOT", "/mnt/work/VectorDB_MICRO/DBAM/runs"))

    # adapter training knobs (only if missing & adapter=on)
    p.add_argument("--adapt_epochs", type=int, default=5)
    p.add_argument("--adapt_lr", type=float, default=5e-4)
    p.add_argument("--adapt_subset", type=int, default=50000)
    p.add_argument("--adapt_qbatch", type=int, default=64)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=6.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--trial", default="adapter")

    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    encoder_tag = safe_encoder_tag(args.encoder)
    alphas = as_tuple(args.alphas)
    ms = as_tuple(args.ms)
    if len(ms) != 3:
        raise ValueError("--ms must be like 2,4,2")
    stages = tuple(x.strip() for x in args.stages.split(","))
    if len(stages) != 3:
        raise ValueError("--stages must be like dual,dual,dual")

    # dirs (don’t mix)
    inter_dir = os.path.join(args.intermediate_root, args.dataset, encoder_tag)
    res_dir   = os.path.join(args.run_root, args.dataset, encoder_tag, "results")
    counting_root = os.path.join(res_dir, "counting")
    runs_root = os.path.join(counting_root, "runs")
    os.makedirs(inter_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(runs_root, exist_ok=True)

    embed_path  = os.path.join(inter_dir, f"passage_embeddings_{args.dataset}_{encoder_tag}.npy")
    qembed_path = os.path.join(inter_dir, f"query_embeddings_{args.dataset}_{encoder_tag}.npy")

    # load caches
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

    # pipeline baseline always needed (also used for mining)
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
                MS_INFER=(2,4,2),          # mining config can stay fixed
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

    # per-run folder
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{args.dataset}__{encoder_tag}__nlist{args.nlist}__np{args.nprobe}__k2{args.k2}__{stages_tag(stages)}__{args.adapter}__{ms_tag(ms)}__{ts}"
    run_dir = os.path.join(runs_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # limits
    Q = len(query_ids)
    Q_run = min(Q, args.max_queries) if args.max_queries and args.max_queries > 0 else Q

    data = pipe
    levels = int(data["levels"])
    a1,a2,a3 = alphas
    m1,m2,m3 = ms
    D = data["queries_q"].shape[1]
    assert D % m1 == 0 and D % m2 == 0 and D % m3 == 0

    g1,g2,g3 = D//m1, D//m2, D//m3

    # hist size uses max possible 2*g (works for dual and your direct score too)
    h1 = np.zeros(2*g1 + 1, dtype=np.int64)
    h2 = np.zeros(2*g2 + 1, dtype=np.int64)
    h3 = np.zeros(2*g3 + 1, dtype=np.int64)

    cent_q = data["centroids_q"]
    cent_d = data["centroids_dual"]
    base_q = data["base_q"]
    base_d = data["base_dual"]
    cluster_to_idxs = data["cluster_to_idxs"]
    passage_ids_arr = data["passage_ids_sample"]

    rows = []
    t2 = time.time()

    for i in tqdm(range(Q_run), desc=f"[{run_id}] counting"):
        qid = str(query_ids[i])
        q_code = data["queries_q"][i]

        s1 = compute_scores(stages[0], q_code, cent_q, cent_d, a1, m1, g1, levels)
        _update_hist(h1, s1)
        selected_clusters = np.argsort(-s1)[:args.nprobe]

        cand = []
        for cid in selected_clusters:
            cand.extend(cluster_to_idxs.get(int(cid), []))
        if len(cand) == 0:
            row = {"query_id": qid}
            for r in range(args.topk):
                row[f"doc{r+1}"] = ""
                row[f"score{r+1}"] = ""
            rows.append(row)
            continue

        candidate_indices = np.asarray(cand, dtype=np.int64)

        s2 = compute_scores(stages[1], q_code, base_q[candidate_indices], base_d[candidate_indices], a2, m2, g2, levels)
        _update_hist(h2, s2)
        order2 = np.argsort(-s2)
        stage2_candidates = candidate_indices[order2[:min(args.k2, len(order2))]]

        s3 = compute_scores(stages[2], q_code, base_q[stage2_candidates], base_d[stage2_candidates], a3, m3, g3, levels)
        _update_hist(h3, s3)

        order3 = np.argsort(-s3)[:args.topk]
        top_idxs = stage2_candidates[order3]
        top_scores = s3[order3]

        row = {"query_id": qid}
        for r in range(args.topk):
            if r < len(top_idxs):
                row[f"doc{r+1}"] = str(passage_ids_arr[int(top_idxs[r])])
                row[f"score{r+1}"] = int(top_scores[r])
            else:
                row[f"doc{r+1}"] = ""
                row[f"score{r+1}"] = ""
        rows.append(row)

    t_count = time.time() - t2

    # ----- write outputs in run_dir -----
    topk_csv = os.path.join(run_dir, "topk.csv")
    pd.DataFrame(rows).to_csv(topk_csv, index=False)

    def write_hist(stage: str, hist: np.ndarray) -> str:
        dfh = pd.DataFrame({"score": np.arange(len(hist), dtype=int), "count": hist})
        dfh = dfh[dfh["count"] > 0].reset_index(drop=True)
        p = os.path.join(run_dir, f"{stage}_hist.csv")
        dfh.to_csv(p, index=False)
        return p

    s1_hist = write_hist("S1", h1)
    s2_hist = write_hist("S2", h2)
    s3_hist = write_hist("S3", h3)

    # stage_stats.csv EXACTLY like you want
    config_name = f"{stages_tag(stages)}_{'with_adapter' if args.adapter=='on' else 'no_adapter'}_{ms_tag(ms)}"
    stats_df = pd.DataFrame([
        {"config": config_name, "stage": "S1_centroids", "m": m1, "g": g1, "max_possible": 2*g1, **_hist_stats(h1)},
        {"config": config_name, "stage": "S2_candidates", "m": m2, "g": g2, "max_possible": 2*g2, **_hist_stats(h2)},
        {"config": config_name, "stage": "S3_K2",        "m": m3, "g": g3, "max_possible": 2*g3, **_hist_stats(h3)},
    ])
    stage_stats_csv = os.path.join(run_dir, "stage_stats.csv")
    stats_df.to_csv(stage_stats_csv, index=False)

    meta_json = {
        "timestamp": ts,
        "run_id": run_id,
        "dataset": args.dataset,
        "encoder": args.encoder,
        "encoder_tag": encoder_tag,
        "adapter": args.adapter,
        "trial_slug": trial_slug,
        "bits_sq": args.bits_sq,
        "nlist": args.nlist,
        "nprobe": args.nprobe,
        "k2": args.k2,
        "alphas": ",".join(map(str, alphas)),
        "ms": ",".join(map(str, ms)),
        "stages": ",".join(stages),
        "topk": args.topk,
        "queries_used": Q_run,
        "paths": {
            "run_dir": run_dir,
            "topk_csv": topk_csv,
            "S1_hist": s1_hist,
            "S2_hist": s2_hist,
            "S3_hist": s3_hist,
            "stage_stats_csv": stage_stats_csv,
        },
        "dataset_meta": meta,
        "timings": {"load_s": t_load, "pipeline_s": t_pipe, "counting_s": t_count},
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta_json, f, indent=2)

    # append to counting/index.csv (append-only, safe)
    index_csv = os.path.join(counting_root, "index.csv")
    index_cols = [
        "timestamp","run_id","dataset","encoder_tag","adapter","stages","ms","nlist","nprobe","k2",
        "run_dir","stage_stats_csv","topk_csv","S1_hist","S2_hist","S3_hist"
    ]
    index_row = {
        "timestamp": ts,
        "run_id": run_id,
        "dataset": args.dataset,
        "encoder_tag": encoder_tag,
        "adapter": args.adapter,
        "stages": ",".join(stages),
        "ms": ",".join(map(str, ms)),
        "nlist": args.nlist,
        "nprobe": args.nprobe,
        "k2": args.k2,
        "run_dir": run_dir,
        "stage_stats_csv": stage_stats_csv,
        "topk_csv": topk_csv,
        "S1_hist": s1_hist,
        "S2_hist": s2_hist,
        "S3_hist": s3_hist,
    }
    append_csv_locked(index_csv, index_row, index_cols)

    print("✅ run_dir:", run_dir, flush=True)
    print("✅ stage stats:", stage_stats_csv, flush=True)
    print("✅ index:", index_csv, flush=True)

if __name__ == "__main__":
    main()
