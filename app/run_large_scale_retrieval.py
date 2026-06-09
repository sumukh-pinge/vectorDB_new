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


def pct(done, total) -> str:
    return f"{done}/{total} ({(100.0 * done / total) if total else 0.0:.1f}%)"


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


def load_query_ids_fallback(query_ids_path: Path, queries_jsonl: Path):
    if query_ids_path.exists():
        return load_query_ids(query_ids_path)
    ids = []
    with queries_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            ids.append(str(json.loads(line)["_id"]))
    print(f"[paths] query_ids fallback from {queries_jsonl} count={len(ids)}", flush=True)
    return ids


def existing_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


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


def scan_positive_doc_rows_with_progress(corpus_file: Path, wanted_docids: set[str], progress_every: int):
    doc_to_row = {}
    if not wanted_docids:
        return doc_to_row
    t0 = time.time()
    with corpus_file.open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            obj = json.loads(line)
            docid = str(obj["_id"])
            if docid in wanted_docids:
                doc_to_row[docid] = row_idx
                if len(doc_to_row) == len(wanted_docids):
                    break
            if progress_every and (row_idx + 1) % progress_every == 0:
                print(
                    f"[stage=positive-row-cache] scanned={row_idx + 1} "
                    f"found={len(doc_to_row)}/{len(wanted_docids)} elapsed_s={time.time() - t0:.1f}",
                    flush=True,
                )
    return doc_to_row


def adapter_slug(args, tag):
    return args.adapter_slug or f"large_scale_adapter_{args.dataset}_{tag}_nlist{args.nlist}"


def adapter_ckpt_path(paths, slug):
    return paths["inter_dir"] / f"W_{slug}_adapter.pt"


def adapter_ready_path(paths, slug):
    return paths["artifacts"] / f"ADAPTER_READY_{slug}.json"


def parse_int_tuple(text: str):
    vals = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if len(vals) != 3:
        raise ValueError(f"expected three comma-separated ints, got: {text}")
    return vals


def ms_tag(ms):
    return "_".join(str(x) for x in ms)


def hard_cache_slug(args):
    ms = parse_int_tuple(args.hard_ms)
    subset = int(args.hard_max_queries or args.adapter_max_queries or 0)
    return (
        f"{args.dataset}_{encoder_tag(args.encoder)}_nlist{args.nlist}"
        f"_split{args.adapter_train_split}_np{args.nprobe}_ms{ms_tag(ms)}"
        f"_seed{args.seed}_subset{subset}_k{args.hard_negatives_per_query}"
    )


def hard_cache_dir(args, paths):
    root = Path(args.hard_cache_root) if args.hard_cache_root else paths["artifacts"] / "hard_negative_cache"
    return root / hard_cache_slug(args)


def hard_shard_path(args, paths, shard_id=None):
    shard = args.hard_shard_id if shard_id is None else shard_id
    return hard_cache_dir(args, paths) / f"shard_{shard:04d}_of_{args.hard_num_shards:04d}.json"


def hard_merged_path(args, paths):
    return hard_cache_dir(args, paths) / "merged.json"


def positive_row_cache_path(args, paths):
    return hard_cache_dir(args, paths) / "positive_rows.json"


def eval_positive_row_cache_path(paths, qrels_split):
    return paths["artifacts"] / f"positive_rows_{qrels_split}.json"


def eval_base_out_dir(args, paths):
    mode_tag = args.mode
    adapter_result_tag = args.result_adapter_tag or args.adapter
    return (
        Path(args.run_root) / args.dataset / paths["tag"] / f"nlist{args.nlist}"
        / f"np{args.nprobe}" / mode_tag / f"adapter_{adapter_result_tag}"
    )


def eval_out_dir(args, paths):
    out_dir = eval_base_out_dir(args, paths)
    if int(args.eval_num_shards) > 1:
        out_dir = out_dir / "shards" / f"shard_{args.eval_shard_id:04d}_of_{args.eval_num_shards:04d}"
    return out_dir


def load_adapter_module(in_dim, bottleneck=64):
    import torch
    import torch.nn as nn

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(in_dim)
            self.down = nn.Linear(in_dim, bottleneck, bias=False)
            self.up = nn.Linear(bottleneck, in_dim, bias=False)
            self.act = nn.GELU()
            nn.init.zeros_(self.up.weight)

        def forward(self, x):
            y = self.down(self.ln(x))
            y = self.act(y)
            y = self.up(y)
            return x + y

    return Adapter


def resolve_device(device_arg):
    if device_arg != "auto":
        return device_arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def apply_adapter_to_queries(queries, ckpt_path: Path, device_arg: str, batch_size: int):
    import torch

    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing adapter checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    in_dim = int(ckpt.get("in_dim", queries.shape[1]))
    bottleneck = int(ckpt.get("bottleneck", 64))
    Adapter = load_adapter_module(in_dim, bottleneck)
    device = resolve_device(device_arg)
    adapter = Adapter().to(device)
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()
    out = np.empty(tuple(queries.shape), dtype=np.float32)
    total = queries.shape[0]
    print(f"[adapter] applying checkpoint={ckpt_path} queries={total} device={device}", flush=True)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = torch.from_numpy(np.asarray(queries[start:end], dtype=np.float32)).to(device)
            out[start:end] = adapter(batch).detach().cpu().numpy().astype(np.float32)
            print(f"[adapter] applied_queries={pct(end, total)}", flush=True)
    return out


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


def load_train_sample(embeddings, n_docs, dim, train_n, seed, blocks=64):
    train = np.empty((train_n, dim), dtype=np.float32)
    if train_n == n_docs:
        train[:] = np.asarray(embeddings, dtype=np.float32)
        return train

    rng = np.random.default_rng(seed)
    blocks = max(1, min(blocks, train_n))
    base = train_n // blocks
    extra = train_n % blocks
    max_block = base + (1 if extra else 0)
    max_start = max(0, n_docs - max_block)
    anchors = np.linspace(0, max_start, blocks, dtype=np.int64)
    jitter_width = max(1, n_docs // (blocks * 4))
    pos = 0
    for block_idx, anchor in enumerate(anchors):
        take = base + (1 if block_idx < extra else 0)
        if take <= 0:
            continue
        jitter = int(rng.integers(-jitter_width, jitter_width + 1))
        start = int(np.clip(anchor + jitter, 0, n_docs - take))
        end = start + take
        train[pos:pos + take] = np.asarray(embeddings[start:end], dtype=np.float32)
        pos += take
    return train


def dbam_dual_scores(q_code, base_q, alpha, m):
    dim = q_code.size
    group_count = dim // m
    bg = base_q.reshape(-1, group_count, m).astype(np.int16, copy=False)
    qg = q_code.reshape(group_count, m).astype(np.int16, copy=False)
    ub_orig = np.all(bg <= (qg + alpha), axis=2)
    ub_dual = np.all((LEVELS - 1 - bg) <= ((LEVELS - 1 - qg) + alpha), axis=2)
    return (ub_orig.sum(axis=1) + ub_dual.sum(axis=1)).astype(np.int16, copy=False)


def dbam_dual_scores_fast(q_code, base_q, alpha, m):
    if m == 1:
        upper = np.minimum(q_code.astype(np.int16) + alpha, LEVELS - 1).astype(np.uint8)
        lower = np.maximum(q_code.astype(np.int16) - alpha, 0).astype(np.uint8)
        orig = base_q <= upper
        dual = base_q >= lower
        return (orig.sum(axis=1) + dual.sum(axis=1)).astype(np.int16, copy=False)
    if m == 2:
        upper = np.minimum(q_code.astype(np.int16) + alpha, LEVELS - 1).astype(np.uint8)
        lower = np.maximum(q_code.astype(np.int16) - alpha, 0).astype(np.uint8)
        orig = (base_q[:, 0::2] <= upper[0::2]) & (base_q[:, 1::2] <= upper[1::2])
        dual = (base_q[:, 0::2] >= lower[0::2]) & (base_q[:, 1::2] >= lower[1::2])
        return (orig.sum(axis=1) + dual.sum(axis=1)).astype(np.int16, copy=False)
    return dbam_dual_scores(q_code, base_q, alpha, m)


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


def chunked_dts_topk(base_q, ids, q_code, m, k, chunk_size, backend="default"):
    best_ids = np.empty(0, dtype=np.int64)
    best_scores = np.empty(0, dtype=np.int16)
    for start in range(0, ids.size, chunk_size):
        chunk_ids = ids[start:start + chunk_size]
        codes = np.asarray(base_q[chunk_ids], dtype=np.uint8)
        if backend == "fast":
            scores = dbam_dual_scores_fast(q_code, codes, alpha=2, m=m)
        else:
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
    qrels_split = args.qrels_split
    return {
        "tag": tag,
        "dataset_dir": dataset_dir,
        "inter_dir": inter_dir,
        "embeddings": inter_dir / f"passage_embeddings_{args.dataset}_{tag}.npy",
        "queries": existing_path([
            inter_dir / f"query_embeddings_{args.dataset}_{tag}_{qrels_split}.npy",
            inter_dir / f"query_embeddings_{args.dataset}_{tag}.npy",
        ]),
        "query_ids": inter_dir / f"query_ids_{args.dataset}_{tag}_{qrels_split}.json",
        "queries_jsonl": dataset_dir / "queries.jsonl",
        "corpus": dataset_dir / "corpus.jsonl",
        "qrels": dataset_dir / "qrels" / f"{qrels_split}.tsv",
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
    print(f"[stage=index] dataset={args.dataset} encoder={args.encoder} embeddings={embeddings.shape} nlist={args.nlist}", flush=True)

    train_n = min(args.train_size, n_docs)
    t_sample = time.time()
    train = load_train_sample(embeddings, n_docs, dim, train_n, args.seed, args.train_blocks)
    print(f"[stage=index] train_sample_n={train_n} train_sample_s={time.time() - t_sample:.1f}", flush=True)

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, args.nlist, faiss.METRIC_L2)
    t0 = time.time()
    print("[stage=index] faiss_train_start", flush=True)
    index.train(train)
    del train
    print(f"[stage=index] faiss_train_done train_s={time.time() - t0:.1f}", flush=True)

    t_add = time.time()
    print("[stage=index] faiss_add_start", flush=True)
    for start in range(0, n_docs, args.add_batch_size):
        end = min(start + args.add_batch_size, n_docs)
        index.add(np.asarray(embeddings[start:end], dtype=np.float32))
        if end % args.progress_every < args.add_batch_size or end == n_docs:
            print(f"[stage=index] added={pct(end, n_docs)}", flush=True)
    print(f"[stage=index] faiss_add_done add_s={time.time() - t_add:.1f}", flush=True)

    print("[stage=index] write_ivf_start", flush=True)
    faiss.write_index(index, str(paths["artifacts"] / "ivf_flat.faiss"))
    print("[stage=index] write_ivf_done", flush=True)

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
    print("[stage=index] quant_minmax_start", flush=True)
    for start in range(0, n_docs, args.quant_batch_size):
        end = min(start + args.quant_batch_size, n_docs)
        chunk = np.asarray(embeddings[start:end], dtype=np.float32)
        vmin = np.minimum(vmin, chunk.min(axis=0))
        vmax = np.maximum(vmax, chunk.max(axis=0))
        if end % args.progress_every < args.quant_batch_size or end == n_docs:
            print(f"[stage=index] minmax={pct(end, n_docs)}", flush=True)
    np.save(paths["artifacts"] / "quant_vmin.npy", vmin)
    np.save(paths["artifacts"] / "quant_vmax.npy", vmax)
    centroids_q = quantize_chunk(centroids, vmin, vmax)
    np.save(paths["artifacts"] / "centroids_q.npy", centroids_q)

    codes_path = paths["artifacts"] / "base_q.uint8.npy"
    codes = np.lib.format.open_memmap(codes_path, mode="w+", dtype=np.uint8, shape=(n_docs, dim))
    print("[stage=index] quant_encode_start", flush=True)
    for start in range(0, n_docs, args.quant_batch_size):
        end = min(start + args.quant_batch_size, n_docs)
        codes[start:end] = quantize_chunk(np.asarray(embeddings[start:end], dtype=np.float32), vmin, vmax)
        if end % args.progress_every < args.quant_batch_size or end == n_docs:
            print(f"[stage=index] quant_encoded={pct(end, n_docs)}", flush=True)
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
    print(f"[stage=index] INDEX_READY written: {done}", flush=True)


def train_adapter(args):
    import torch
    import torch.nn.functional as F

    paths = get_paths(args)
    paths["artifacts"].mkdir(parents=True, exist_ok=True)
    slug = adapter_slug(args, paths["tag"])
    ckpt_path = adapter_ckpt_path(paths, slug)
    ready = adapter_ready_path(paths, slug)
    if ready.exists() and ckpt_path.exists() and not args.force:
        print(f"[skip] adapter already ready: {ready}", flush=True)
        return

    if args.adapter_source_slug:
        source = adapter_ckpt_path(paths, args.adapter_source_slug)
        if not source.exists():
            raise FileNotFoundError(f"missing source adapter: {source}")
        ckpt_path.write_bytes(source.read_bytes())
        w_source = paths["inter_dir"] / f"W_{args.adapter_source_slug}.npy"
        w_dest = paths["inter_dir"] / f"W_{slug}.npy"
        if w_source.exists():
            w_dest.write_bytes(w_source.read_bytes())
        write_json(ready, {
            "dataset": args.dataset,
            "encoder": args.encoder,
            "adapter_slug": slug,
            "source_slug": args.adapter_source_slug,
            "checkpoint": str(ckpt_path),
            "timestamp": time.time(),
        })
        print(f"[stage=adapter-train] ADAPTER_READY copied source={source} dest={ckpt_path}", flush=True)
        return

    embeddings = np.load(paths["embeddings"], mmap_mode="r")
    train_paths = get_paths(argparse.Namespace(**{**vars(args), "qrels_split": args.adapter_train_split}))
    queries = np.load(train_paths["queries"], mmap_mode="r")
    query_ids_all = load_query_ids_fallback(train_paths["query_ids"], train_paths["queries_jsonl"])
    qrels = read_qrels(train_paths["qrels"])
    keep = [i for i, qid in enumerate(query_ids_all) if qid in qrels]
    if args.adapter_max_queries:
        keep = keep[:args.adapter_max_queries]
    query_ids = [query_ids_all[i] for i in keep]
    wanted_docids = set().union(*(qrels[qid] for qid in query_ids)) if query_ids else set()
    print(f"[stage=adapter-train] scan_positive_rows positives={len(wanted_docids)}", flush=True)
    doc_to_row = scan_positive_doc_rows(paths["corpus"], wanted_docids)

    pairs = []
    for qi, qid in zip(keep, query_ids):
        for docid in qrels[qid]:
            row = doc_to_row.get(docid)
            if row is not None:
                pairs.append((qi, int(row)))
                if len(pairs) >= args.adapter_max_pairs:
                    break
        if len(pairs) >= args.adapter_max_pairs:
            break
    if not pairs:
        raise RuntimeError("no adapter training pairs found")

    rng = np.random.default_rng(args.seed)
    neg_rows = rng.choice(embeddings.shape[0], size=min(args.adapter_negatives, embeddings.shape[0]), replace=False)
    device = resolve_device(args.adapter_device)
    dim = embeddings.shape[1]
    Adapter = load_adapter_module(dim, args.adapter_bottleneck)
    adapter = Adapter().to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.adapter_lr, weight_decay=1e-4)
    neg_tensor = torch.from_numpy(np.asarray(embeddings[neg_rows], dtype=np.float32)).to(device)
    neg_tensor = F.normalize(neg_tensor, p=2, dim=1)
    pair_arr = np.asarray(pairs, dtype=np.int64)
    steps_per_epoch = int(math.ceil(len(pair_arr) / args.adapter_qbatch))
    print(
        f"[stage=adapter-train] dataset={args.dataset} pairs={len(pair_arr)} negatives={len(neg_rows)} "
        f"epochs={args.adapter_epochs} batch={args.adapter_qbatch} device={device}",
        flush=True,
    )

    for epoch in range(args.adapter_epochs):
        order = rng.permutation(len(pair_arr))
        running = 0.0
        for step in range(steps_per_epoch):
            idx = order[step * args.adapter_qbatch:(step + 1) * args.adapter_qbatch]
            batch = pair_arr[idx]
            q_np = np.asarray(queries[batch[:, 0]], dtype=np.float32)
            p_np = np.asarray(embeddings[batch[:, 1]], dtype=np.float32)
            q = torch.from_numpy(q_np).to(device)
            p = torch.from_numpy(p_np).to(device)
            q_proj = adapter(q)
            q_norm = F.normalize(q_proj, p=2, dim=1)
            p_norm = F.normalize(p, p=2, dim=1)
            pos = (q_norm * p_norm).sum(dim=1, keepdim=True)
            neg = q_norm @ neg_tensor.t()
            logits = torch.cat([pos, neg], dim=1) / args.adapter_tau
            labels = torch.zeros(q.shape[0], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.adapter_grad_clip)
            opt.step()
            running += float(loss.detach().cpu())
            if (step + 1) % args.adapter_progress_every == 0 or step + 1 == steps_per_epoch:
                print(
                    f"[stage=adapter-train] epoch={epoch + 1}/{args.adapter_epochs} "
                    f"steps={pct(step + 1, steps_per_epoch)} loss={running / (step + 1):.4f}",
                    flush=True,
                )

    torch.save({"state_dict": adapter.state_dict(), "in_dim": dim, "bottleneck": args.adapter_bottleneck}, ckpt_path)
    np.save(paths["inter_dir"] / f"W_{slug}.npy", np.eye(dim, dtype=np.float32))
    write_json(ready, {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "adapter_slug": slug,
        "checkpoint": str(ckpt_path),
        "pairs": int(len(pair_arr)),
        "negatives": int(len(neg_rows)),
        "epochs": int(args.adapter_epochs),
        "timestamp": time.time(),
    })
    print(f"[stage=adapter-train] ADAPTER_READY written: {ready}", flush=True)


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


def load_training_queries(args, paths):
    train_paths = get_paths(argparse.Namespace(**{**vars(args), "qrels_split": args.adapter_train_split}))
    queries = np.load(train_paths["queries"], mmap_mode="r")
    query_ids_all = load_query_ids_fallback(train_paths["query_ids"], train_paths["queries_jsonl"])
    qrels = read_qrels(train_paths["qrels"])
    keep = [i for i, qid in enumerate(query_ids_all) if qid in qrels]
    max_queries = int(args.hard_max_queries or args.adapter_max_queries or 0)
    if max_queries:
        keep = keep[:max_queries]
    if args.hard_num_shards < 1:
        raise ValueError("--hard_num_shards must be >= 1")
    if not (0 <= args.hard_shard_id < args.hard_num_shards):
        raise ValueError("--hard_shard_id must be in [0, hard_num_shards)")
    shard_keep = keep[args.hard_shard_id::args.hard_num_shards]
    shard_qids = [query_ids_all[i] for i in shard_keep]
    return train_paths, queries, query_ids_all, qrels, keep, shard_keep, shard_qids


def selected_training_query_rows(args, paths):
    train_paths = get_paths(argparse.Namespace(**{**vars(args), "qrels_split": args.adapter_train_split}))
    query_ids_all = load_query_ids_fallback(train_paths["query_ids"], train_paths["queries_jsonl"])
    qrels = read_qrels(train_paths["qrels"])
    keep = [i for i, qid in enumerate(query_ids_all) if qid in qrels]
    max_queries = int(args.hard_max_queries or args.adapter_max_queries or 0)
    if max_queries:
        keep = keep[:max_queries]
    qids = [query_ids_all[i] for i in keep]
    return train_paths, query_ids_all, qrels, keep, qids


def build_positive_row_cache(args):
    paths = get_paths(args)
    out_path = positive_row_cache_path(args, paths)
    if out_path.exists() and not args.force:
        print(f"[skip] positive row cache exists: {out_path}", flush=True)
        return

    t0 = time.time()
    train_paths, query_ids_all, qrels, keep, qids = selected_training_query_rows(args, paths)
    wanted_docids = set().union(*(qrels[qid] for qid in qids)) if qids else set()
    print(
        f"[stage=positive-row-cache] dataset={args.dataset} split={args.adapter_train_split} "
        f"selected_queries={len(keep)} wanted_docids={len(wanted_docids)} out={out_path}",
        flush=True,
    )
    doc_to_row = scan_positive_doc_rows_with_progress(paths["corpus"], wanted_docids, args.progress_every)
    missing = sorted(wanted_docids.difference(doc_to_row))
    obj = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "encoder_tag": paths["tag"],
        "adapter_train_split": args.adapter_train_split,
        "nlist": args.nlist,
        "nprobe": args.nprobe,
        "k2": args.k2,
        "kfinal": args.kfinal,
        "hard_ms": args.hard_ms,
        "seed": args.seed,
        "selected_queries": len(keep),
        "wanted_docids": len(wanted_docids),
        "found_docids": len(doc_to_row),
        "missing_docids": len(missing),
        "doc_to_row": {str(k): int(v) for k, v in doc_to_row.items()},
        "missing_docid_examples": missing[:20],
        "elapsed_s": time.time() - t0,
        "timestamp": time.time(),
    }
    write_json(out_path, obj)
    print(
        f"[stage=positive-row-cache] wrote {out_path} "
        f"found={len(doc_to_row)}/{len(wanted_docids)} elapsed_s={time.time() - t0:.1f}",
        flush=True,
    )


def load_positive_row_cache(args, paths, wanted_docids):
    path = positive_row_cache_path(args, paths)
    if not path.exists():
        raise FileNotFoundError(
            f"missing positive row cache: {path}; run --stage build-positive-row-cache first"
        )
    cache = json.loads(path.read_text())
    raw = cache.get("doc_to_row", {})
    doc_to_row = {str(docid): int(row) for docid, row in raw.items() if str(docid) in wanted_docids}
    missing = wanted_docids.difference(doc_to_row)
    if missing:
        print(
            f"[stage=mine-hard] positive_row_cache missing={len(missing)}/{len(wanted_docids)} "
            f"examples={sorted(missing)[:5]}",
            flush=True,
        )
    return doc_to_row


def build_eval_positive_row_cache(args):
    paths = get_paths(args)
    out_path = eval_positive_row_cache_path(paths, args.qrels_split)
    if out_path.exists() and not args.force:
        print(f"[skip] eval positive row cache exists: {out_path}", flush=True)
        return

    t0 = time.time()
    query_ids_all = load_query_ids_fallback(paths["query_ids"], paths["queries_jsonl"])
    qrels = read_qrels(paths["qrels"])
    keep = [i for i, qid in enumerate(query_ids_all) if qid in qrels]
    if args.max_queries:
        keep = keep[:args.max_queries]
    qids = [query_ids_all[i] for i in keep]
    wanted_docids = set().union(*(qrels[qid] for qid in qids)) if qids else set()
    print(
        f"[stage=eval-positive-row-cache] dataset={args.dataset} split={args.qrels_split} "
        f"selected_queries={len(keep)} wanted_docids={len(wanted_docids)} out={out_path}",
        flush=True,
    )
    doc_to_row = scan_positive_doc_rows_with_progress(paths["corpus"], wanted_docids, args.progress_every)
    missing = sorted(wanted_docids.difference(doc_to_row))
    write_json(out_path, {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "encoder_tag": paths["tag"],
        "qrels_split": args.qrels_split,
        "nlist": args.nlist,
        "selected_queries": len(keep),
        "wanted_docids": len(wanted_docids),
        "found_docids": len(doc_to_row),
        "missing_docids": len(missing),
        "doc_to_row": {str(k): int(v) for k, v in doc_to_row.items()},
        "missing_docid_examples": missing[:20],
        "elapsed_s": time.time() - t0,
        "timestamp": time.time(),
    })
    print(
        f"[stage=eval-positive-row-cache] wrote {out_path} "
        f"found={len(doc_to_row)}/{len(wanted_docids)} elapsed_s={time.time() - t0:.1f}",
        flush=True,
    )


def load_eval_positive_row_cache(args, paths, wanted_docids):
    path = eval_positive_row_cache_path(paths, args.qrels_split)
    if not path.exists():
        print(f"[stage=eval] eval positive row cache missing; scanning corpus: {path}", flush=True)
        return scan_positive_doc_rows(paths["corpus"], wanted_docids)
    cache = json.loads(path.read_text())
    raw = cache.get("doc_to_row", {})
    doc_to_row = {str(docid): int(row) for docid, row in raw.items() if str(docid) in wanted_docids}
    missing = wanted_docids.difference(doc_to_row)
    if missing:
        print(
            f"[stage=eval] eval_positive_row_cache missing={len(missing)}/{len(wanted_docids)} "
            f"examples={sorted(missing)[:5]}",
            flush=True,
        )
    return doc_to_row


def retrieve_rows_for_mining(
    embeddings,
    base_q,
    centroids,
    centroids_q,
    offsets,
    list_ids,
    vmin,
    vmax,
    q_vec,
    nprobe,
    k2,
    kfinal,
    score_chunk_size,
    ms,
    dts_backend="default",
):
    q_vec = np.asarray(q_vec, dtype=np.float32)
    q_code = quantize_chunk(q_vec[None, :], vmin, vmax)[0]
    candidate_ids = selected_candidate_ids(centroids, centroids_q, offsets, list_ids, q_vec, q_code, nprobe, ms[0])
    if candidate_ids.size == 0:
        return np.empty(0, dtype=np.int64), 0
    stage2 = chunked_dts_topk(
        base_q, candidate_ids, q_code, ms[1], min(k2, candidate_ids.size),
        score_chunk_size, dts_backend
    )
    final = chunked_dts_topk(
        base_q, stage2, q_code, ms[2], min(kfinal, stage2.size),
        score_chunk_size, dts_backend
    )
    return final, int(candidate_ids.size)


def mine_hard_negatives(args):
    paths = get_paths(args)
    ready = paths["artifacts"] / "INDEX_READY.json"
    if not ready.exists():
        raise FileNotFoundError(f"missing index marker: {ready}")

    out_path = hard_shard_path(args, paths)
    if out_path.exists() and not args.force:
        print(f"[skip] hard-negative shard exists: {out_path}", flush=True)
        return

    ms = parse_int_tuple(args.hard_ms)
    t0 = time.time()
    train_paths, queries, query_ids_all, qrels, keep_all, shard_keep, shard_qids = load_training_queries(args, paths)
    wanted_docids = set().union(*(qrels[qid] for qid in shard_qids)) if shard_qids else set()
    print(
        f"[stage=mine-hard] dataset={args.dataset} split={args.adapter_train_split} "
        f"selected_total={len(keep_all)} shard={args.hard_shard_id}/{args.hard_num_shards} "
        f"shard_queries={len(shard_keep)} positives={len(wanted_docids)} ms={ms}",
        flush=True,
    )
    doc_to_row = load_positive_row_cache(args, paths, wanted_docids)

    embeddings = np.load(paths["embeddings"], mmap_mode="r")
    centroids = np.load(paths["artifacts"] / "centroids.npy", mmap_mode="r")
    centroids_q = np.load(paths["artifacts"] / "centroids_q.npy", mmap_mode="r")
    offsets = np.load(paths["artifacts"] / "list_offsets.npy", mmap_mode="r")
    list_ids = np.load(paths["artifacts"] / "list_ids.npy", mmap_mode="r")
    vmin = np.load(paths["artifacts"] / "quant_vmin.npy")
    vmax = np.load(paths["artifacts"] / "quant_vmax.npy")
    base_q = np.load(paths["artifacts"] / "base_q.uint8.npy", mmap_mode="r")

    pairs = []
    negatives = []
    rows = []
    skipped_no_positive = 0
    skipped_no_negative = 0
    raw_counts = []
    for local_idx, (qrow, qid) in enumerate(zip(shard_keep, shard_qids), 1):
        pos_rows = []
        for docid in qrels[qid]:
            row = doc_to_row.get(docid)
            if row is not None:
                pos_rows.append(int(row))
                pairs.append([int(qrow), int(row)])
        if not pos_rows:
            skipped_no_positive += 1
            continue
        final, raw_count = retrieve_rows_for_mining(
            embeddings, base_q, centroids, centroids_q, offsets, list_ids, vmin, vmax,
            np.asarray(queries[qrow], dtype=np.float32), args.nprobe, args.k2, args.kfinal,
            args.score_chunk_size, ms, args.dts_backend,
        )
        raw_counts.append(raw_count)
        pos_set = set(pos_rows)
        neg_rows = [int(row) for row in final if int(row) not in pos_set][:args.hard_negatives_per_query]
        if not neg_rows:
            skipped_no_negative += 1
        negatives.extend(neg_rows)
        rows.append({
            "query_id": qid,
            "query_row": int(qrow),
            "positive_rows": pos_rows,
            "negative_rows": neg_rows,
            "raw_candidates": int(raw_count),
        })
        if local_idx % args.progress_every_queries == 0 or local_idx == len(shard_keep):
            print(
                f"[stage=mine-hard] queries={pct(local_idx, len(shard_keep))} "
                f"pairs={len(pairs)} negatives={len(negatives)} last_raw={raw_count}",
                flush=True,
            )

    obj = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "encoder_tag": paths["tag"],
        "adapter_train_split": args.adapter_train_split,
        "nlist": args.nlist,
        "nprobe": args.nprobe,
        "k2": args.k2,
        "kfinal": args.kfinal,
        "ms": list(ms),
        "dts_backend": args.dts_backend,
        "seed": args.seed,
        "selected_total_queries": len(keep_all),
        "shard_id": args.hard_shard_id,
        "num_shards": args.hard_num_shards,
        "shard_queries": len(shard_keep),
        "pairs": pairs,
        "negative_rows": negatives,
        "rows": rows,
        "skipped_no_positive": skipped_no_positive,
        "skipped_no_negative": skipped_no_negative,
        "raw_candidates_avg": float(np.mean(raw_counts)) if raw_counts else 0.0,
        "elapsed_s": time.time() - t0,
        "timestamp": time.time(),
    }
    write_json(out_path, obj)
    print(f"[stage=mine-hard] wrote {out_path} pairs={len(pairs)} negatives={len(negatives)}", flush=True)


def merge_hard_negatives(args):
    paths = get_paths(args)
    out_path = hard_merged_path(args, paths)
    if out_path.exists() and not args.force:
        print(f"[skip] merged hard-negative cache exists: {out_path}", flush=True)
        return

    shards = []
    missing = []
    for shard_id in range(args.hard_num_shards):
        path = hard_shard_path(args, paths, shard_id)
        if path.exists():
            shards.append(json.loads(path.read_text()))
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} shard files; first={missing[:3]}")

    pairs = []
    negatives = []
    rows = []
    for shard in sorted(shards, key=lambda x: int(x["shard_id"])):
        pairs.extend(shard.get("pairs", []))
        negatives.extend(shard.get("negative_rows", []))
        rows.extend(shard.get("rows", []))
    rows = sorted(rows, key=lambda r: int(r["query_row"]))
    obj = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "encoder_tag": paths["tag"],
        "adapter_train_split": args.adapter_train_split,
        "nlist": args.nlist,
        "nprobe": args.nprobe,
        "k2": args.k2,
        "kfinal": args.kfinal,
        "ms": list(parse_int_tuple(args.hard_ms)),
        "seed": args.seed,
        "num_shards": args.hard_num_shards,
        "pairs": pairs,
        "negative_rows": negatives,
        "rows": rows,
        "query_count": len(rows),
        "pair_count": len(pairs),
        "negative_count": len(negatives),
        "timestamp": time.time(),
    }
    write_json(out_path, obj)
    print(f"[stage=merge-hard] wrote {out_path} queries={len(rows)} pairs={len(pairs)} negatives={len(negatives)}", flush=True)


def train_adapter_from_neg_cache(args):
    import torch
    import torch.nn.functional as F

    profile_t0 = time.time()
    last_t = profile_t0

    def profile_mark(name):
        nonlocal last_t
        if not args.profile_timings:
            return
        now = time.time()
        print(
            f"[profile=train-hard] {name} delta_s={now - last_t:.6f} elapsed_s={now - profile_t0:.6f}",
            flush=True,
        )
        last_t = now

    paths = get_paths(args)
    cache_path = hard_merged_path(args, paths)
    if not cache_path.exists() and args.hard_num_shards == 1:
        cache_path = hard_shard_path(args, paths, 0)
    if not cache_path.exists():
        raise FileNotFoundError(f"missing hard-negative cache: {cache_path}")

    slug = adapter_slug(args, paths["tag"])
    final_ckpt_path = adapter_ckpt_path(paths, slug)
    ready = adapter_ready_path(paths, slug)
    if ready.exists() and final_ckpt_path.exists() and not args.force:
        print(f"[skip] cached-hard adapter already ready: {ready}", flush=True)
        return

    profile_mark("paths_ready")
    cache = json.loads(cache_path.read_text())
    profile_mark("cache_json_read")
    pairs = cache.get("pairs", [])
    neg_rows_all = [int(x) for x in cache.get("negative_rows", [])]
    profile_mark("cache_parse_rows")
    if not pairs:
        raise RuntimeError(f"hard-negative cache has no positive pairs: {cache_path}")
    if not neg_rows_all:
        raise RuntimeError(f"hard-negative cache has no negatives: {cache_path}")

    rng = np.random.default_rng(args.seed)
    if args.hard_train_negatives and len(neg_rows_all) > args.hard_train_negatives:
        idx = rng.choice(len(neg_rows_all), size=args.hard_train_negatives, replace=False)
        neg_rows = [neg_rows_all[int(i)] for i in idx]
    else:
        neg_rows = neg_rows_all
    profile_mark("negative_subset_select")

    embeddings = np.load(paths["embeddings"], mmap_mode="r")
    profile_mark("passage_mmap_open")
    train_paths = get_paths(argparse.Namespace(**{**vars(args), "qrels_split": args.adapter_train_split}))
    queries = np.load(train_paths["queries"], mmap_mode="r")
    profile_mark("query_mmap_open")
    device = resolve_device(args.adapter_device)
    dim = embeddings.shape[1]
    Adapter = load_adapter_module(dim, args.adapter_bottleneck)
    adapter = Adapter().to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.adapter_lr, weight_decay=1e-4)
    profile_mark("adapter_optimizer_init")
    neg_tensor = torch.from_numpy(np.asarray(embeddings[neg_rows], dtype=np.float32)).to(device)
    profile_mark("negative_embedding_gather_to_tensor")
    neg_tensor = F.normalize(neg_tensor, p=2, dim=1)
    profile_mark("negative_tensor_normalize")
    pair_arr = np.asarray(pairs[:args.adapter_max_pairs] if args.adapter_max_pairs else pairs, dtype=np.int64)
    profile_mark("pair_array_build")
    steps_per_epoch = int(math.ceil(len(pair_arr) / args.adapter_qbatch))
    checkpoint_epochs = sorted({int(x.strip()) for x in args.adapter_checkpoint_epochs.split(",") if x.strip()})
    print(
        f"[stage=train-hard] dataset={args.dataset} slug={slug} pairs={len(pair_arr)} "
        f"cache_negatives={len(neg_rows_all)} train_negatives={len(neg_rows)} "
        f"epochs={args.adapter_epochs} checkpoints={checkpoint_epochs} batch={args.adapter_qbatch} device={device}",
        flush=True,
    )

    epoch_losses = []
    for epoch in range(args.adapter_epochs):
        order = rng.permutation(len(pair_arr))
        running = 0.0
        for step in range(steps_per_epoch):
            idx = order[step * args.adapter_qbatch:(step + 1) * args.adapter_qbatch]
            batch = pair_arr[idx]
            q_np = np.asarray(queries[batch[:, 0]], dtype=np.float32)
            p_np = np.asarray(embeddings[batch[:, 1]], dtype=np.float32)
            q = torch.from_numpy(q_np).to(device)
            p = torch.from_numpy(p_np).to(device)
            q_proj = adapter(q)
            q_norm = F.normalize(q_proj, p=2, dim=1)
            p_norm = F.normalize(p, p=2, dim=1)
            pos = (q_norm * p_norm).sum(dim=1, keepdim=True)
            neg = q_norm @ neg_tensor.t()
            logits = torch.cat([pos, neg], dim=1) / args.adapter_tau
            labels = torch.zeros(q.shape[0], dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.adapter_grad_clip)
            opt.step()
            running += float(loss.detach().cpu())
            if (step + 1) % args.adapter_progress_every == 0 or step + 1 == steps_per_epoch:
                print(
                    f"[stage=train-hard] epoch={epoch + 1}/{args.adapter_epochs} "
                    f"steps={pct(step + 1, steps_per_epoch)} loss={running / (step + 1):.4f}",
                    flush=True,
                )
        epoch_loss = running / max(1, steps_per_epoch)
        epoch_losses.append(epoch_loss)
        if (epoch + 1) in checkpoint_epochs:
            epoch_slug = f"{slug}_epoch{epoch + 1}"
            epoch_ckpt = adapter_ckpt_path(paths, epoch_slug)
            torch.save({"state_dict": adapter.state_dict(), "in_dim": dim, "bottleneck": args.adapter_bottleneck}, epoch_ckpt)
            np.save(paths["inter_dir"] / f"W_{epoch_slug}.npy", np.eye(dim, dtype=np.float32))
            write_json(adapter_ready_path(paths, epoch_slug), {
                "dataset": args.dataset,
                "encoder": args.encoder,
                "adapter_slug": epoch_slug,
                "source_cache": str(cache_path),
                "epoch": epoch + 1,
                "pairs": int(len(pair_arr)),
                "cache_negatives": int(len(neg_rows_all)),
                "train_negatives": int(len(neg_rows)),
                "loss": float(epoch_loss),
                "timestamp": time.time(),
            })
            print(f"[stage=train-hard] checkpoint epoch={epoch + 1} slug={epoch_slug}", flush=True)
            profile_mark(f"checkpoint_epoch_{epoch + 1}")

    torch.save({"state_dict": adapter.state_dict(), "in_dim": dim, "bottleneck": args.adapter_bottleneck}, final_ckpt_path)
    np.save(paths["inter_dir"] / f"W_{slug}.npy", np.eye(dim, dtype=np.float32))
    write_json(ready, {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "adapter_slug": slug,
        "source_cache": str(cache_path),
        "pairs": int(len(pair_arr)),
        "cache_negatives": int(len(neg_rows_all)),
        "train_negatives": int(len(neg_rows)),
        "epochs": int(args.adapter_epochs),
        "epoch_losses": [float(x) for x in epoch_losses],
        "timestamp": time.time(),
    })
    profile_mark("final_checkpoint_and_ready")
    print(f"[stage=train-hard] ADAPTER_READY written: {ready}", flush=True)


def eval_retrieval(args):
    paths = get_paths(args)
    adapter_result_tag = args.result_adapter_tag or args.adapter
    out_dir = eval_out_dir(args, paths)
    summary_json = out_dir / "summary.json"
    if summary_json.exists() and not args.force:
        print(f"[skip] summary exists: {summary_json}", flush=True)
        return
    if args.eval_num_shards < 1:
        raise ValueError("--eval_num_shards must be >= 1")
    if not (0 <= args.eval_shard_id < args.eval_num_shards):
        raise ValueError("--eval_shard_id must be in [0, eval_num_shards)")

    ready = paths["artifacts"] / "INDEX_READY.json"
    if not ready.exists():
        raise FileNotFoundError(f"missing index marker: {ready}")
    faiss.omp_set_num_threads(args.faiss_threads)

    t_load = time.time()
    embeddings = np.load(paths["embeddings"], mmap_mode="r")
    queries = np.load(paths["queries"], mmap_mode="r")
    query_ids_all = load_query_ids_fallback(paths["query_ids"], paths["queries_jsonl"])
    qrels = read_qrels(paths["qrels"])
    keep = [i for i, qid in enumerate(query_ids_all) if qid in qrels]
    if args.max_queries:
        keep = keep[:args.max_queries]
    total_queries_before_shard = len(keep)
    if args.eval_num_shards > 1:
        keep = keep[args.eval_shard_id::args.eval_num_shards]
    query_ids = [query_ids_all[i] for i in keep]
    queries = queries[keep]
    if args.adapter == "on":
        slug = adapter_slug(args, paths["tag"])
        ckpt_path = adapter_ckpt_path(paths, slug)
        queries = apply_adapter_to_queries(queries, ckpt_path, args.adapter_device, args.adapter_batch_size)
    wanted_docids = set().union(*(qrels[qid] for qid in query_ids)) if query_ids else set()
    print(
        f"[stage=eval] dataset={args.dataset} mode={args.mode} adapter={args.adapter} "
        f"nlist={args.nlist} nprobe={args.nprobe} shard={args.eval_shard_id}/{args.eval_num_shards} "
        f"queries={len(query_ids)}/{total_queries_before_shard} qrels_positives={len(wanted_docids)}",
        flush=True,
    )
    doc_to_row = load_eval_positive_row_cache(args, paths, wanted_docids)
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
            stage2 = chunked_dts_topk(
                base_q, candidate_ids, q_code, stage_m, min(args.k2, raw_count),
                args.score_chunk_size, args.dts_backend
            )
            final = chunked_dts_topk(
                base_q, stage2, q_code, stage_m, min(args.kfinal, stage2.size),
                args.score_chunk_size, args.dts_backend
            )
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
            print(f"[stage=eval] queries={pct(qi, len(query_ids))} last_raw={raw_count}", flush=True)
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
        "adapter": adapter_result_tag,
        "adapter_mode": args.adapter,
        "adapter_slug": adapter_slug(args, paths["tag"]) if args.adapter == "on" else "",
        "dts_backend": args.dts_backend,
        "nlist": args.nlist,
        "nprobe": args.nprobe,
        "k2": args.k2,
        "kfinal": args.kfinal,
        "qrels_split": args.qrels_split,
        "queries": len(query_ids),
        "total_queries_before_shard": total_queries_before_shard,
        "eval_num_shards": int(args.eval_num_shards),
        "eval_shard_id": int(args.eval_shard_id),
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
    print(f"[stage=eval] summary written: {summary_json}", flush=True)


def merge_eval_shards(args):
    paths = get_paths(args)
    base_dir = eval_base_out_dir(args, paths)
    summary_json = base_dir / "summary.json"
    if summary_json.exists() and not args.force:
        print(f"[skip] merged eval summary exists: {summary_json}", flush=True)
        return
    if args.eval_num_shards < 2:
        raise ValueError("--eval_num_shards must be >= 2 for merge-eval-shards")

    shard_summaries = []
    per_query = []
    missing = []
    for shard_id in range(args.eval_num_shards):
        shard_dir = base_dir / "shards" / f"shard_{shard_id:04d}_of_{args.eval_num_shards:04d}"
        s_path = shard_dir / "summary.json"
        p_path = shard_dir / "per_query.csv"
        if not s_path.exists() or not p_path.exists():
            missing.append(str(s_path))
            continue
        shard_summaries.append(json.loads(s_path.read_text()))
        with p_path.open("r", newline="") as f:
            for row in csv.DictReader(f):
                per_query.append(row)
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} shard summaries; first={missing[:3]}")
    if not per_query:
        raise RuntimeError("no per-query rows found in eval shards")

    for row in per_query:
        for key in ["raw_candidates", "stage2_candidates", "final_candidates", "positives"]:
            row[key] = int(float(row[key]))
        for key in ["H@25", "H@100", "R@25", "R@100"]:
            row[key] = float(row[key])

    raw_counts = [row["raw_candidates"] for row in per_query]
    stage2_counts = [row["stage2_candidates"] for row in per_query]
    final_counts = [row["final_candidates"] for row in per_query]
    metric_row = {
        "H@25": float(np.mean([row["H@25"] for row in per_query])),
        "H@100": float(np.mean([row["H@100"] for row in per_query])),
        "R@25": float(np.mean([row["R@25"] for row in per_query])),
        "R@100": float(np.mean([row["R@100"] for row in per_query])),
    }
    candidate_stats = {}
    candidate_stats.update(stat_block(raw_counts, "raw_candidates"))
    candidate_stats.update(stat_block(stage2_counts, "stage2_candidates"))
    candidate_stats.update(stat_block(final_counts, "final_candidates"))

    first = shard_summaries[0]
    eval_times = [float(s.get("timing_eval_s", 0.0)) for s in shard_summaries]
    load_times = [float(s.get("timing_load_s", 0.0)) for s in shard_summaries]
    merged = {
        **{k: first.get(k) for k in [
            "dataset", "encoder", "encoder_tag", "mode", "adapter", "adapter_mode",
            "adapter_slug", "nlist", "nprobe", "k2", "kfinal", "qrels_split",
            "faiss_threads", "score_chunk_size", "dts_backend"
        ]},
        "queries": int(len(per_query)),
        "total_queries_before_shard": int(sum(int(s.get("queries", 0)) for s in shard_summaries)),
        "eval_num_shards": int(args.eval_num_shards),
        "eval_shard_id": "merged",
        "missing_positive_qrels": int(sum(int(s.get("missing_positive_qrels", 0)) for s in shard_summaries)),
        "timing_load_s": float(sum(load_times)),
        "timing_eval_s": float(sum(eval_times)),
        "timing_eval_wall_s_if_parallel": float(max(eval_times) if eval_times else 0.0),
        "ms_per_query": float(sum(eval_times) * 1000.0 / len(per_query)),
        "ms_per_query_wall_if_parallel": float(max(eval_times) * 1000.0 / len(per_query)) if eval_times else 0.0,
        "qps": float(len(per_query) / sum(eval_times)) if sum(eval_times) > 0 else 0.0,
        "qps_wall_if_parallel": float(len(per_query) / max(eval_times)) if eval_times and max(eval_times) > 0 else 0.0,
        **metric_row,
        **candidate_stats,
        "timestamp": time.time(),
    }
    write_json(summary_json, merged)
    write_csv(base_dir / "summary.csv", [merged], list(merged.keys()))
    write_csv(base_dir / "per_query.csv", per_query, list(per_query[0].keys()) if per_query else [])
    print(
        f"[stage=merge-eval-shards] wrote {summary_json} queries={len(per_query)} "
        f"sum_eval_s={sum(eval_times):.2f} max_eval_s={max(eval_times) if eval_times else 0.0:.2f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "build-index",
            "train-adapter",
            "prep",
            "build-positive-row-cache",
            "build-eval-positive-row-cache",
            "mine-hard-negatives",
            "merge-hard-negatives",
            "train-from-neg-cache",
            "eval",
            "merge-eval-shards",
        ],
        required=True,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--mode", choices=["ivf", "dts111", "dts242"], default="ivf")
    parser.add_argument("--adapter", choices=["off", "on"], default="off")
    parser.add_argument("--adapter_slug", default="")
    parser.add_argument("--result_adapter_tag", default="")
    parser.add_argument("--adapter_source_slug", default="")
    parser.add_argument("--adapter_train_split", default="train")
    parser.add_argument("--adapter_device", default="auto")
    parser.add_argument("--adapter_batch_size", type=int, default=1024)
    parser.add_argument("--adapter_bottleneck", type=int, default=64)
    parser.add_argument("--adapter_epochs", type=int, default=1)
    parser.add_argument("--adapter_lr", type=float, default=1e-3)
    parser.add_argument("--adapter_tau", type=float, default=0.07)
    parser.add_argument("--adapter_grad_clip", type=float, default=1.0)
    parser.add_argument("--adapter_qbatch", type=int, default=256)
    parser.add_argument("--adapter_negatives", type=int, default=4096)
    parser.add_argument("--adapter_max_queries", type=int, default=50000)
    parser.add_argument("--adapter_max_pairs", type=int, default=100000)
    parser.add_argument("--adapter_progress_every", type=int, default=25)
    parser.add_argument("--adapter_checkpoint_epochs", default="1,2,3,5")
    parser.add_argument("--hard_cache_root", default="")
    parser.add_argument("--hard_ms", default="1,1,1")
    parser.add_argument("--hard_max_queries", type=int, default=0)
    parser.add_argument("--hard_num_shards", type=int, default=1)
    parser.add_argument("--hard_shard_id", type=int, default=0)
    parser.add_argument("--hard_negatives_per_query", type=int, default=10)
    parser.add_argument("--hard_train_negatives", type=int, default=50000)
    parser.add_argument("--profile_timings", action="store_true")
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
    parser.add_argument("--train_blocks", type=int, default=64)
    parser.add_argument("--add_batch_size", type=int, default=100000)
    parser.add_argument("--quant_batch_size", type=int, default=200000)
    parser.add_argument("--score_chunk_size", type=int, default=250000)
    parser.add_argument("--dts_backend", choices=["default", "fast"], default="default")
    parser.add_argument("--progress_every", type=int, default=1000000)
    parser.add_argument("--progress_every_queries", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--eval_num_shards", type=int, default=1)
    parser.add_argument("--eval_shard_id", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.stage == "build-index":
        build_index(args)
    elif args.stage == "train-adapter":
        train_adapter(args)
    elif args.stage == "prep":
        print("[stage=prep] start index_then_adapter", flush=True)
        build_index(args)
        print("[stage=prep] index phase complete; starting adapter phase", flush=True)
        train_adapter(args)
        print("[stage=prep] complete INDEX_READY and ADAPTER_READY", flush=True)
    elif args.stage == "build-positive-row-cache":
        build_positive_row_cache(args)
    elif args.stage == "build-eval-positive-row-cache":
        build_eval_positive_row_cache(args)
    elif args.stage == "mine-hard-negatives":
        mine_hard_negatives(args)
    elif args.stage == "merge-hard-negatives":
        merge_hard_negatives(args)
    elif args.stage == "train-from-neg-cache":
        train_adapter_from_neg_cache(args)
    elif args.stage == "merge-eval-shards":
        merge_eval_shards(args)
    else:
        eval_retrieval(args)


if __name__ == "__main__":
    main()
