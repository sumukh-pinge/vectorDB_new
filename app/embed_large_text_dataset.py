#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def encoder_tag(name):
    return name.replace("/", "_")


def resolve_dataset_dir(data_root, dataset):
    root = Path(data_root) / dataset
    nested = root / dataset
    if (nested / "corpus.jsonl").exists():
        return nested
    if (root / "corpus.jsonl").exists():
        return root
    raise FileNotFoundError(f"Could not find corpus.jsonl under {root}")


def count_jsonl(path):
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


def read_corpus_batches(corpus_file, batch_size, max_docs=0):
    ids = []
    texts = []
    seen = 0
    with open(corpus_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if max_docs and seen >= max_docs:
                break
            obj = json.loads(line)
            ids.append(str(obj["_id"]))
            texts.append((obj.get("title", "") + " " + obj.get("text", "")).strip())
            seen += 1
            if len(texts) >= batch_size:
                yield ids, texts
                ids = []
                texts = []
    if texts:
        yield ids, texts


def load_positive_qids(qrels_file):
    qids = set()
    with open(qrels_file, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            if parts[0].lower().startswith("query"):
                continue
            try:
                score = float(parts[2])
            except ValueError:
                continue
            if score > 0:
                qids.add(parts[0])
    return qids


def read_queries(queries_file, keep_qids):
    query_ids = []
    query_texts = []
    with open(queries_file, "r", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            qid = str(obj["_id"])
            if qid in keep_qids:
                query_ids.append(qid)
                query_texts.append(obj["text"])
    return query_ids, query_texts


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def wait_for_path(path, timeout_sec, poll_sec):
    start = time.time()
    while not Path(path).exists():
        elapsed = time.time() - start
        if timeout_sec > 0 and elapsed > timeout_sec:
            raise TimeoutError(f"Timed out waiting for {path}")
        print(f"[wait] {path} ({int(elapsed)}s elapsed)", flush=True)
        time.sleep(poll_sec)


def encode_corpus(args, model, ds_dir, inter_dir, tag):
    corpus_file = ds_dir / "corpus.jsonl"
    embed_path = inter_dir / f"passage_embeddings_{args.dataset}_{tag}.npy"
    done_path = inter_dir / f"passage_embeddings_{args.dataset}_{tag}.done.json"
    if done_path.exists() and embed_path.exists() and not args.force:
        print(f"[skip] corpus embeddings ready: {embed_path}", flush=True)
        return json.loads(done_path.read_text())

    if args.max_docs:
        total = args.max_docs
    elif args.expected_docs:
        total = args.expected_docs
    else:
        total = count_jsonl(corpus_file)
    print(f"[corpus] total_docs={total} output={embed_path}", flush=True)

    tmp_path = embed_path.with_suffix(".npy.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    inter_dir.mkdir(parents=True, exist_ok=True)

    dim = model.get_sentence_embedding_dimension()
    mmap = np.lib.format.open_memmap(tmp_path, mode="w+", dtype="float32", shape=(total, dim))

    offset = 0
    started = time.time()
    for _, texts in tqdm(read_corpus_batches(corpus_file, args.corpus_batch_size, args.max_docs), desc="Encoding corpus"):
        emb = model.encode(
            texts,
            batch_size=args.corpus_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=args.normalize,
        ).astype("float32")
        end = offset + emb.shape[0]
        mmap[offset:end] = emb
        offset = end
        if offset % args.flush_every_docs < args.corpus_batch_size:
            mmap.flush()
            print(f"[corpus] encoded={offset}/{total}", flush=True)
    mmap.flush()
    del mmap

    if offset != total:
        raise RuntimeError(f"Encoded {offset} docs, expected {total}")
    os.replace(tmp_path, embed_path)
    stats = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "encoder_tag": tag,
        "kind": "corpus",
        "path": str(embed_path),
        "rows": total,
        "dim": dim,
        "normalize": args.normalize,
        "elapsed_sec": time.time() - started,
    }
    write_json(done_path, stats)
    print(json.dumps(stats, indent=2), flush=True)
    return stats


def encode_queries_for_split(args, model, ds_dir, inter_dir, tag, split):
    qrels_file = ds_dir / "qrels" / f"{split}.tsv"
    if not qrels_file.exists():
        print(f"[queries] skip missing split={split}", flush=True)
        return None
    queries_file = ds_dir / "queries.jsonl"
    qembed_path = inter_dir / f"query_embeddings_{args.dataset}_{tag}_{split}.npy"
    done_path = inter_dir / f"query_embeddings_{args.dataset}_{tag}_{split}.done.json"
    if done_path.exists() and qembed_path.exists() and not args.force:
        print(f"[skip] query embeddings ready: {qembed_path}", flush=True)
        return json.loads(done_path.read_text())

    keep_qids = load_positive_qids(qrels_file)
    query_ids, query_texts = read_queries(queries_file, keep_qids)
    print(f"[queries] split={split} positive_qids={len(keep_qids)} query_texts={len(query_texts)}", flush=True)
    started = time.time()
    emb = model.encode(
        query_texts,
        batch_size=args.query_batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=args.normalize,
    ).astype("float32")
    np.save(qembed_path, emb)
    with open(inter_dir / f"query_ids_{args.dataset}_{tag}_{split}.json", "w", encoding="utf-8") as handle:
        json.dump(query_ids, handle)
    stats = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "encoder_tag": tag,
        "kind": "queries",
        "split": split,
        "path": str(qembed_path),
        "rows": len(query_ids),
        "dim": int(emb.shape[1]) if emb.ndim == 2 else 0,
        "positive_qids": len(keep_qids),
        "normalize": args.normalize,
        "elapsed_sec": time.time() - started,
    }
    write_json(done_path, stats)
    print(json.dumps(stats, indent=2), flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--data_root", default=os.getenv("DATA_ROOT", "/mnt/work/VectorDB_MICRO/datasets/semantic"))
    parser.add_argument("--intermediate_root", default=os.getenv("INTERMEDIATE_ROOT", "/mnt/work/VectorDB_MICRO/intermediate_data"))
    parser.add_argument("--corpus_batch_size", type=int, default=512)
    parser.add_argument("--query_batch_size", type=int, default=512)
    parser.add_argument("--flush_every_docs", type=int, default=100000)
    parser.add_argument("--query_splits", default="dev,train")
    parser.add_argument("--wait_timeout_sec", type=int, default=86400)
    parser.add_argument("--wait_poll_sec", type=int, default=60)
    parser.add_argument("--max_docs", type=int, default=0)
    parser.add_argument("--expected_docs", type=int, default=0)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no_normalize", dest="normalize", action="store_false")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    expected_ready_path = Path(args.data_root) / args.dataset / args.dataset / "DATASET_READY.json"
    fallback_ready_path = Path(args.data_root) / args.dataset / "DATASET_READY.json"
    ready_path = expected_ready_path if expected_ready_path.parent.exists() or not fallback_ready_path.exists() else fallback_ready_path
    wait_for_path(ready_path, args.wait_timeout_sec, args.wait_poll_sec)
    ds_dir = resolve_dataset_dir(args.data_root, args.dataset)

    tag = encoder_tag(args.encoder)
    inter_dir = Path(args.intermediate_root) / args.dataset / tag
    inter_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}", flush=True)
    if device != "cuda":
        print("[warning] CUDA not available; embedding will be slow", flush=True)
    model = SentenceTransformer(args.encoder, device=device)

    corpus_stats = encode_corpus(args, model, ds_dir, inter_dir, tag)
    query_stats = []
    for split in [s.strip() for s in args.query_splits.split(",") if s.strip()]:
        stats = encode_queries_for_split(args, model, ds_dir, inter_dir, tag, split)
        if stats is not None:
            query_stats.append(stats)

    all_stats = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "encoder_tag": tag,
        "corpus": corpus_stats,
        "queries": query_stats,
        "timestamp": time.time(),
    }
    write_json(inter_dir / f"EMBEDDINGS_READY_{args.dataset}_{tag}.json", all_stats)
    print("[done] embeddings ready", flush=True)
    print(json.dumps(all_stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
