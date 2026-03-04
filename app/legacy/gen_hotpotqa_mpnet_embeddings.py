#!/usr/bin/env python3
import os
import sys
import json
import argparse
import datetime as dt
import numpy as np
from sentence_transformers import SentenceTransformer


def make_logger(log_dir: str, dataset: str):
    os.makedirs(log_dir, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"gen_{dataset}_mpnet_{ts}.log")
    f = open(log_path, "a", buffering=1)

    def log(msg: str):
        line = str(msg)
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()

    return log


def get_embed_path(data_dir: str, dataset: str) -> str:
    # Match your previous naming
    if dataset in ("hotpotqa", "hotpotqa_mpnet"):
        name = "sample_passage_embeddings_hotpotqa.npy"
    elif dataset in ("beir_nq", "beir_nq_mpnet"):
        name = "sample_passage_embeddings_nq.npy"
    else:
        name = f"sample_passage_embeddings_{dataset}.npy"
    return os.path.join(data_dir, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/mnt/work/datasets")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    data_dir = os.path.join(args.data_root, args.dataset)
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    embed_path = get_embed_path(data_dir, args.dataset)
    log = make_logger(data_dir, args.dataset)

    log(f"=== MPNet embedding generation for {args.dataset} ===")
    log(f"[paths] data_dir = {data_dir}")
    log(f"[paths] corpus  = {corpus_path}")
    log(f"[paths] output  = {embed_path}")

    if not os.path.exists(corpus_path):
        log(f"❌ corpus.jsonl not found at {corpus_path}")
        sys.exit(1)

    # If a valid file exists, skip recompute
    if os.path.exists(embed_path):
        try:
            tmp = np.load(embed_path, mmap_mode="r")
            log(f"🔄 Found existing embeddings at {embed_path}, shape={tmp.shape}, dtype={tmp.dtype}")
            log("✅ Skipping generation.")
            return
        except Exception as e:
            log(f"⚠️ Existing embeddings unreadable, regenerating. Error: {e}")
            try:
                os.remove(embed_path)
            except Exception as e2:
                log(f"❌ Failed to remove bad file: {e2}")
                sys.exit(1)

    # Load model
    log("[model] Loading all-mpnet-base-v2")
    encoder = SentenceTransformer("all-mpnet-base-v2")

    # Load all passages (simple like your notebook)
    log("[data] Loading passages from corpus.jsonl")
    passage_texts = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = (obj.get("title", "") + " " + obj.get("text", "")).strip()
            passage_texts.append(text)

    total = len(passage_texts)
    if total == 0:
        log("❌ No passages found in corpus.jsonl")
        sys.exit(1)
    log(f"[data] Total passages: {total}")

    # Encode in batches, log each batch
    batch_size = args.batch_size
    embeddings_batches = []

    log(f"[encode] Starting encoding with batch_size={batch_size}")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = passage_texts[start:end]
        emb = encoder.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        embeddings_batches.append(emb)

        pct = (end / total) * 100.0
        batch_idx = start // batch_size
        log(f"[encode] batch={batch_idx} done={end}/{total} ({pct:.4f}%)")

    # Stack and save
    embeddings = np.vstack(embeddings_batches)
    np.save(embed_path, embeddings)
    log(f"✅ Saved embeddings to: {embed_path}")

    # Reload check
    try:
        test = np.load(embed_path, mmap_mode="r")
        log(f"🔍 Reload check OK: shape={test.shape}, dtype={test.dtype}")
    except Exception as e:
        log(f"❌ Reload check FAILED for {embed_path}: {e}")
        sys.exit(1)

    log("🎉 Done.")
    log("=========================================")


if __name__ == "__main__":
    main()
