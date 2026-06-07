#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import faiss
import numpy as np


KS = [25, 100]
LEVELS = 16


def encoder_tag(encoder: str) -> str:
    return encoder.replace("/", "_")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def write_csv(path: Path, rows, fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp.replace(path)


def read_qrels(qrels_file: Path):
    qrels = {}
    with qrels_file.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, docid, score = parts[0], parts[1], float(parts[2])
            if score <= 0:
                continue
            qrels.setdefault(qid, set()).add(docid)
    return qrels


def load_query_ids(path: Path):
    return json.loads(path.read_text())


def scan_positive_doc_rows(corpus_file: Path, wanted_docids: set[str]):
    doc_to_row = {}
    if not wanted_docids:
        return doc_to_row
    with corpus_file.open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            obj = json.loads(line)
            docid = str(obj["_id"])
            if docid in wanted_docids:
                doc_to_row[docid] = row_idx
                if len(doc_to_row) == len(wanted_docids):
                    break
    return doc_to_row


def qrels_to_row_sets(qrels, doc_to_row):
    out = {}
    missing = 0
    for qid, docids in qrels.items():
        rows = set()
        for docid in docids:
            row = doc_to_row.get(docid)
            if row is None:
                missing += 1
            else:
                rows.add(int(row))
        if rows:
            out[qid] = rows
    return out, missing


def metrics(query_ids, retrieved_rows, qrels_rows):
    denom = 0
    hit = {k: 0 for k in KS}
    recall_sum = {k: 0.0 for k in KS}
    mrr_sum = {k: 0.0 for k in KS}
    for qid, preds in zip(query_ids, retrieved_rows):
        gt = qrels_rows.get(qid)
        if not gt:
            continue
        denom += 1
        gt_len = len(gt)
        for k in KS:
            top = preds[:k]
            top_set = set(int(x) for x in top)
            inter = len(gt & top_set)
            if inter:
                hit[k] += 1
            recall_sum[k] += inter / gt_len
            rr = 0.0
            for rank, row in enumerate(top, 1):
                if int(row) in gt:
                    rr = 1.0 / rank
                    break
            mrr_sum[k] += rr
    out = {}
    for k in KS:
        out[f"H@{k}"] = hit[k] / denom if denom else 0.0
        out[f"R@{k}"] = recall_sum[k] / denom if denom else 0.0
        out[f"MRR@{k}"] = mrr_sum[k] / denom if denom else 0.0
    out["metric_queries"] = denom
    return out


def percentile(values, p):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def stat_block(values, prefix):
    if not values:
        return {
            f"{prefix}_avg": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0,
        }
    arr = np.asarray(values, dtype=np.int64)
    return {
        f"{prefix}_avg": float(arr.mean()),
        f"{prefix}_p50": percentile(values, 50),
        f"{prefix}_p95": percentile(values, 95),
        f"{prefix}_max": int(arr.max()),
    }


def quantize_chunk(x, vmin, vmax):
    rng = np.maximum(vmax - vmin, 1e-8)
    y = (x - vmin) / rng * (LEVELS - 1)
    return np.clip(np.rint(y), 0, LEVELS - 1).astype(np.uint8)


def dbam_dual_scores(q_code, base_q, alpha, m):
    dim = q_code.size
    group_count = dim // m
    bg = base_q.reshape(-1, group_count, m).astype(np.int16, copy=False)
    qg = q_code.reshape(group_count, m).astype(np.int16, copy=False)
    ub_orig = np.all(bg <= (qg + alpha), axis=2)
    ub_dual = np.all((LEVELS - 1 - bg) <= ((LEVELS - 1 - qg) + alpha), axis=2)
    return (ub_orig.sum(axis=1) + ub_dual.sum(axis=1)).astype(np.int16, copy=False)


def topk_from_scores(ids, scores, k, largest=True):
    if ids.size <= k:
        order = np.argsort(-scores if largest else scores)
        return ids[order], scores[order]
    if largest:
        part = np.argpartition(scores, -k)[-k:]
        order = np.argsort(-scores[part])
    else:
        part = np.argpartition(scores, k)[:k]
        order = np.argsort(scores[part])
    chosen = part[order]
    return ids[chosen], scores[chosen]


def chunked_l2_topk(embeddings, ids, q_vec, k, chunk_size):
    best_ids = np.empty(0, dtype=np.int64)
    best_scores = np.empty(0, dtype=np.float32)
    for start in range(0, ids.size, chunk_size):
        chunk_ids = ids[start:start + chunk_size]
        vecs = np.asarray(embeddings[chunk_ids], dtype=np.float32)
        diff = vecs - q_vec
        dist = np.einsum("ij,ij->i", diff, diff)
        cand_ids = np.concatenate([best_ids, chunk_ids])
        cand_scores = np.concatenate([best_scores, dist.astype(np.float32, copy=False)])
        best_ids, best_scores = topk_from_scores(cand_ids, cand_scores, min(k, cand_ids.size), largest=False)
    return best_ids


def chunked_dts_topk(base_q, ids, q_code, m, k, chunk_size):
    best_ids = np.empty(0, dtype=np.int64)
    best_scores = np.empty(0, dtype=np.int16)
    for start in range(0, ids.size, chunk_size):
        chunk_ids = ids[start:start + chunk_size]
        codes = np.asarray(base_q[chunk_ids], dtype=np.uint8)
        scores = dbam_dual_scores(q_code, codes, alpha=2, m=m)
        cand_ids = np.concatenate([best_ids, chunk_ids])
        cand_scores = np.concatenate([best_scores, scores])
        best_ids, best_scores = topk_from_scores(cand_ids, cand_scores, min(k, cand_ids.size), largest=True)
    return best_ids


def artifact_dir(index_root: Path, dataset: str, tag: str, nlist: int):
    return index_root / dataset / tag / f"nlist{nlist}"


def get_paths(args):
    tag = encoder_tag(args.encoder)
    dataset_dir = Path(args.data_root) / args.dataset / args.dataset
    inter_dir = Path(args.intermediate_root) / args.dataset / tag
    return {
        "tag": tag,
        "dataset_dir": dataset_dir,
        "inter_dir": inter_dir,
        "embeddings": inter_dir / f"passage_embeddings_{args.dataset}_{tag}.npy",
        "queries": inter_dir / f"query_embeddings_{args.dataset}_{tag}_{args.qrels_split}.npy",
        "query_ids": inter_dir / f"query_ids_{args.dataset}_{tag}_{args.qrels_split}.json",
        "corpus": dataset_dir / "corpus.jsonl",
        "qrels": dataset_dir / "qrels" / f"{args.qrels_split}.tsv",
        "artifacts": artifact_dir(Path(args.index_root), args.dataset, tag, args.nlist),
    }


def build_index(args):
    paths = get_paths(args)
    done = paths["artifacts"] / "INDEX_READY.json"
    if done.exists() and not args.force:
        print(f"[skip] index already ready: {done}", flush=True)
        return

    paths["artifacts"].mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(args.faiss_threads)

    embeddings = np.load(paths["embeddings"], mmap_mode="r")
    n_docs, dim = embeddings.shape
    print(f"[index] embeddings={embeddings.shape} nlist={args.nlist}", flush=True)

    rng = np.random.default_rng(args.seed)
    train_n = min(args.train_size, n_docs)
    train_ids = rng.choice(n_docs, size=train_n, replace=False)
    train = np.asarray(embeddings[train_ids], dtype=np.float32)

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, args.nlist, faiss.METRIC_L2)
    t0 = time.time()
    index.train(train)
    del train
    print(f"[index] train_s={time.time() - t0:.1f}", flush=True)

    t_add = time.time()
    for start in range(0, n_docs, args.add_batch_size):
        end = min(start + args.add_batch_size, n_docs)
        index.add(np.asarray(embeddings[start:end], dtype=np.float32))
        if end % args.progress_every < args.add_batch_size or end == n_docs:
            print(f"[index] added={end}/{n_docs}", flush=True)
    print(f"[index] add_s={time.time() - t_add:.1f}", flush=True)

    faiss.write_index(index, str(paths["artifacts"] / "ivf_flat.faiss"))

    qz = faiss.downcast_index(index.quantizer)
    centroids = np.vstack([qz.reconstruct(i) for i in range(args.nlist)]).astype(np.float32)
    np.save(paths["artifacts"] / "centroids.npy", centroids)

    invlists = index.invlists
    sizes = np.asarray([invlists.list_size(i) for i in range(args.nlist)], dtype=np.int64)
    offsets = np.zeros(args.nlist + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)
    ids_flat = np.empty(int(offsets[-1]), dtype=np.int64)
    for list_id in range(args.nlist):
        size = int(sizes[list_id])
        if size == 0:
            continue
        ptr = invlists.get_ids(list_id)
        ids_flat[offsets[list_id]:offsets[list_id + 1]] = faiss.rev_swig_ptr(ptr, size)
    np.save(paths["artifacts"] / "list_offsets.npy", offsets)
    np.save(paths["artifacts"] / "list_ids.npy", ids_flat)

    vmin = np.full(dim, np.inf, dtype=np.float32)
    vmax = np.full(dim, -np.inf, dtype=np.float32)
    for start in range(0, n_docs, args.quant_batch_size):
        chunk = np.asarray(embeddings[start:start + args.quant_batch_size], dtype=np.float32)
        vmin = np.minimum(vmin, chunk.min(axis=0))
        vmax = np.maximum(vmax, chunk.max(axis=0))
    np.save(paths["artifacts"] / "quant_vmin.npy", vmin)
    np.save(paths["artifacts"] / "quant_vmax.npy", vmax)
    centroids_q = quantize_chunk(centroids, vmin, vmax)
    np.save(paths["artifacts"] / "centroids_q.npy", centroids_q)

    codes_path = paths["artifacts"] / "base_q.uint8.npy"
    codes = np.lib.format.open_memmap(codes_path, mode="w+", dtype=np.uint8, shape=(n_docs, dim))
    for start in range(0, n_docs, args.quant_batch_size):
        end = min(start + args.quant_batch_size, n_docs)
        codes[start:end] = quantize_chunk(np.asarray(embeddings[start:end], dtype=np.float32), vmin, vmax)
        if end % args.progress_every < args.quant_batch_size or end == n_docs:
            print(f"[quant] encoded={end}/{n_docs}", flush=True)
    codes.flush()

    meta = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "encoder_tag": paths["tag"],
        "n_docs": int(n_docs),
        "dim": int(dim),
        "nlist": int(args.nlist),
        "faiss_threads": int(args.faiss_threads),
        "train_size": int(train_n),
        "artifacts": str(paths["artifacts"]),
        "timestamp": time.time(),
    }
    write_json(done, meta)
    print(f"[done] index ready: {done}", flush=True)


def selected_candidate_ids(centroids, centroids_q, offsets, list_ids, q_vec, q_code, nprobe, mode_m):
    if mode_m is None:
        diff = centroids - q_vec
        dist = np.einsum("ij,ij->i", diff, diff)
        clusters = np.argpartition(dist, nprobe)[:nprobe]
        clusters = clusters[np.argsort(dist[clusters])]
    else:
        scores = dbam_dual_scores(q_code, centroids_q, alpha=2, m=mode_m)
        clusters = np.argpartition(scores, -nprobe)[-nprobe:]
        clusters = clusters[np.argsort(-scores[clusters])]
    total = int(sum(offsets[c + 1] - offsets[c] for c in clusters))
    out = np.empty(total, dtype=np.int64)
    pos = 0
    for cluster in clusters:
        start, end = int(offsets[cluster]), int(offsets[cluster + 1])
        size = end - start
        out[pos:pos + size] = list_ids[start:end]
        pos += size
    return out


def eval_retrieval(args):
    paths = get_paths(args)
    mode_tag = args.mode
    out_dir = Path(args.run_root) / args.dataset / paths["tag"] / f"nlist{args.nlist}" / f"np{args.nprobe}" / mode_tag
    summary_json = out_dir / "summary.json"
    if summary_json.exists() and not args.force:
        print(f"[skip] summary exists: {summary_json}", flush=True)
        return

    ready = paths["artifacts"] / "INDEX_READY.json"
    if not ready.exists():
        raise FileNotFoundError(f"missing index marker: {ready}")
    faiss.omp_set_num_threads(args.faiss_threads)

    t_load = time.time()
    embeddings = np.load(paths["embeddings"], mmap_mode="r")
    queries = np.load(paths["queries"], mmap_mode="r")
    query_ids_all = load_query_ids(paths["query_ids"])
    qrels = read_qrels(paths["qrels"])
    keep = [i for i, qid in enumerate(query_ids_all) if qid in qrels]
    if args.max_queries:
        keep = keep[:args.max_queries]
    query_ids = [query_ids_all[i] for i in keep]
    queries = queries[keep]
    wanted_docids = set().union(*(qrels[qid] for qid in query_ids)) if query_ids else set()
    doc_to_row = scan_positive_doc_rows(paths["corpus"], wanted_docids)
    qrels_rows, missing_qrels = qrels_to_row_sets({qid: qrels[qid] for qid in query_ids}, doc_to_row)

    centroids = np.load(paths["artifacts"] / "centroids.npy", mmap_mode="r")
    centroids_q = np.load(paths["artifacts"] / "centroids_q.npy", mmap_mode="r")
    offsets = np.load(paths["artifacts"] / "list_offsets.npy", mmap_mode="r")
    list_ids = np.load(paths["artifacts"] / "list_ids.npy", mmap_mode="r")
    vmin = np.load(paths["artifacts"] / "quant_vmin.npy")
    vmax = np.load(paths["artifacts"] / "quant_vmax.npy")
    base_q = None if args.mode == "ivf" else np.load(paths["artifacts"] / "base_q.uint8.npy", mmap_mode="r")
    t_load = time.time() - t_load

    if args.mode == "ivf":
        stage_m = None
    elif args.mode == "dts111":
        stage_m = 1
    elif args.mode == "dts242":
        stage_m = 2
    else:
        raise ValueError(args.mode)

    retrieved = []
    per_query = []
    raw_counts = []
    stage2_counts = []
    final_counts = []
    t_eval = time.time()
    for qi, (qid, q_vec) in enumerate(zip(query_ids, queries), 1):
        q_vec = np.asarray(q_vec, dtype=np.float32)
        q_code = quantize_chunk(q_vec[None, :], vmin, vmax)[0]
        candidate_ids = selected_candidate_ids(
            centroids, centroids_q, offsets, list_ids, q_vec, q_code, args.nprobe, stage_m
        )
        raw_count = int(candidate_ids.size)
        if raw_count == 0:
            final = np.empty(0, dtype=np.int64)
            stage2 = np.empty(0, dtype=np.int64)
        elif args.mode == "ivf":
            stage2 = chunked_l2_topk(embeddings, candidate_ids, q_vec, min(args.k2, raw_count), args.score_chunk_size)
            final = chunked_l2_topk(embeddings, stage2, q_vec, min(args.kfinal, stage2.size), args.score_chunk_size)
        else:
            stage2 = chunked_dts_topk(base_q, candidate_ids, q_code, stage_m, min(args.k2, raw_count), args.score_chunk_size)
            final = chunked_dts_topk(base_q, stage2, q_code, stage_m, min(args.kfinal, stage2.size), args.score_chunk_size)
        final_list = [int(x) for x in final[:args.kfinal]]
        retrieved.append(final_list)
        raw_counts.append(raw_count)
        stage2_counts.append(int(stage2.size))
        final_counts.append(len(final_list))
        gt = qrels_rows.get(qid, set())
        per_query.append({
            "query_id": qid,
            "raw_candidates": raw_count,
            "stage2_candidates": int(stage2.size),
            "final_candidates": len(final_list),
            "positives": len(gt),
            "H@25": int(bool(gt & set(final_list[:25]))),
            "H@100": int(bool(gt & set(final_list[:100]))),
            "R@25": (len(gt & set(final_list[:25])) / len(gt)) if gt else 0.0,
            "R@100": (len(gt & set(final_list[:100])) / len(gt)) if gt else 0.0,
        })
        if qi % args.progress_every_queries == 0 or qi == len(query_ids):
            print(f"[eval] queries={qi}/{len(query_ids)} last_raw={raw_count}", flush=True)
    t_eval = time.time() - t_eval

    metric_row = metrics(query_ids, retrieved, qrels_rows)
    candidate_stats = {}
    candidate_stats.update(stat_block(raw_counts, "raw_candidates"))
    candidate_stats.update(stat_block(stage2_counts, "stage2_candidates"))
    candidate_stats.update(stat_block(final_counts, "final_candidates"))
    summary = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "encoder_tag": paths["tag"],
        "mode": args.mode,
        "nlist": args.nlist,
        "nprobe": args.nprobe,
        "k2": args.k2,
        "kfinal": args.kfinal,
        "qrels_split": args.qrels_split,
        "queries": len(query_ids),
        "missing_positive_qrels": missing_qrels,
        "timing_load_s": t_load,
        "timing_eval_s": t_eval,
        "ms_per_query": (t_eval * 1000.0 / len(query_ids)) if query_ids else 0.0,
        "qps": (len(query_ids) / t_eval) if t_eval > 0 else 0.0,
        "faiss_threads": args.faiss_threads,
        "score_chunk_size": args.score_chunk_size,
        **metric_row,
        **candidate_stats,
        "timestamp": time.time(),
    }
    write_json(summary_json, summary)
    write_csv(out_dir / "summary.csv", [summary], list(summary.keys()))
    write_csv(out_dir / "per_query.csv", per_query, list(per_query[0].keys()) if per_query else [])
    print(f"[done] wrote {summary_json}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["build-index", "eval"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--mode", choices=["ivf", "dts111", "dts242"], default="ivf")
    parser.add_argument("--data_root", default=os.getenv("DATA_ROOT", "/mnt/work/VectorDB_MICRO/datasets/semantic"))
    parser.add_argument("--intermediate_root", default=os.getenv("INTERMEDIATE_ROOT", "/mnt/work/VectorDB_MICRO/intermediate_data"))
    parser.add_argument("--index_root", default=os.getenv("INDEX_ROOT", "/mnt/work/VectorDB_MICRO/large_scale_indices"))
    parser.add_argument("--run_root", default=os.getenv("RUN_ROOT", "/mnt/work/VectorDB_MICRO/large_scale_runs")
                        )
    parser.add_argument("--qrels_split", default="dev")
    parser.add_argument("--nlist", type=int, required=True)
    parser.add_argument("--nprobe", type=int, default=128)
    parser.add_argument("--k2", type=int, default=1000)
    parser.add_argument("--kfinal", type=int, default=100)
    parser.add_argument("--faiss_threads", type=int, default=4)
    parser.add_argument("--train_size", type=int, default=1000000)
    parser.add_argument("--add_batch_size", type=int, default=100000)
    parser.add_argument("--quant_batch_size", type=int, default=200000)
    parser.add_argument("--score_chunk_size", type=int, default=250000)
    parser.add_argument("--progress_every", type=int, default=1000000)
    parser.add_argument("--progress_every_queries", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.stage == "build-index":
        build_index(args)
    else:
        eval_retrieval(args)


if __name__ == "__main__":
    main()
