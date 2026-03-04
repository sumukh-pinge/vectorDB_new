# beir_jsonl_loader.py
#!/usr/bin/env python3
import os, json
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def _resolve_ds_dir(data_root: str, dataset: str) -> str:
    """
    Handles both layouts:
      data_root/dataset/corpus.jsonl
      data_root/dataset/dataset/corpus.jsonl
    """
    d1 = os.path.join(data_root, dataset)
    if os.path.exists(os.path.join(d1, "corpus.jsonl")):
        return d1
    d2 = os.path.join(d1, dataset)
    if os.path.exists(os.path.join(d2, "corpus.jsonl")):
        return d2
    raise FileNotFoundError(f"Could not find corpus.jsonl under {d1} or {d2}")


def _pick_qrels(ds_dir: str) -> Tuple[str, str]:
    qrels_dir = os.path.join(ds_dir, "qrels")
    for split in ["test", "dev", "train"]:
        p = os.path.join(qrels_dir, f"{split}.tsv")
        if os.path.exists(p):
            return p, split
    raise FileNotFoundError(f"No qrels split found under {qrels_dir} (expected test/dev/train.tsv)")


def _read_jsonl_ids(path: str, key: str = "_id") -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(str(json.loads(line)[key]))
    return out


def _load_qrels_tsv(path: str) -> pd.DataFrame:
    # Robust: handles header/no-header, extra columns
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, engine="python")
    if df.shape[1] >= 3:
        df = df.iloc[:, :3]
    df.columns = ["query_id", "corpus_id", "score"]

    # Drop header-like first row if present
    first = df.iloc[0].astype(str).str.lower().tolist()
    if ("query" in first[0]) and ("corpus" in first[1]) and ("score" in first[2]):
        df = df.iloc[1:].reset_index(drop=True)

    df["score_num"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    return df


def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
    return (x / n).astype("float32")


def load_beir_jsonl(
    data_root: str,
    dataset: str,
    encoder_name: str,
    device: str,
    embed_path: Optional[str] = None,
    query_embed_path: Optional[str] = None,
    passage_batch_size: int = 512,
    query_batch_size: int = 128,
    normalize: bool = True,
):
    """
    Returns:
      embeddings (np.float32 [N,D]),
      queries_emb (np.float32 [Q,D]),
      passage_ids (list[str]),
      query_ids (list[str]),
      query_to_gt (dict[qid] -> list[corpus_id]),
      meta (dict)
    """
    ds_dir = _resolve_ds_dir(data_root, dataset)
    corpus_file = os.path.join(ds_dir, "corpus.jsonl")
    queries_file = os.path.join(ds_dir, "queries.jsonl")
    qrels_file, qrels_split = _pick_qrels(ds_dir)

    encoder = SentenceTransformer(encoder_name, device=device)

    # Passage IDs
    passage_ids = _read_jsonl_ids(corpus_file)

    # Passage embeddings
    need_rebuild_passages = False
    if embed_path and os.path.exists(embed_path):
        print(f"🔄 loading passage embeddings: {embed_path}", flush=True)
        embeddings = np.load(embed_path).astype("float32")
        if embeddings.shape[0] != len(passage_ids):
            print(
                f"⚠️ passage embed cache mismatch: embed_N={embeddings.shape[0]} vs corpus_N={len(passage_ids)}. Rebuilding...",
                flush=True,
            )
            need_rebuild_passages = True
    else:
        need_rebuild_passages = True

    if need_rebuild_passages:
        print("⚙️ building passage embeddings (no/invalid cache)", flush=True)
        texts = []
        passage_ids = []
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading corpus"):
                obj = json.loads(line)
                passage_ids.append(str(obj["_id"]))
                texts.append((obj.get("title", "") + " " + obj.get("text", "")).strip())

        chunks = []
        for i in tqdm(range(0, len(texts), passage_batch_size), desc="Encoding passages"):
            batch = texts[i : i + passage_batch_size]
            emb = encoder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            chunks.append(emb)

        embeddings = np.vstack(chunks).astype("float32")
        if normalize:
            embeddings = _l2norm(embeddings)

        if embed_path:
            os.makedirs(os.path.dirname(embed_path), exist_ok=True)
            np.save(embed_path, embeddings)
            print(f"✅ saved passage embeddings: {embed_path}", flush=True)
    else:
        if normalize:
            embeddings = _l2norm(embeddings)

    # Qrels (positives only)
    qrels = _load_qrels_tsv(qrels_file)
    pos = qrels[qrels["score_num"] > 0].reset_index(drop=True)
    keep_qids = set(pos["query_id"].astype(str).unique().tolist())

    # Queries
    qs = [json.loads(line) for line in open(queries_file, "r", encoding="utf-8")]
    qs = [q for q in qs if str(q["_id"]) in keep_qids]
    query_ids = [str(q["_id"]) for q in qs]
    query_texts = [q["text"] for q in qs]

    # Query embeddings
    need_rebuild_queries = False
    if query_embed_path and os.path.exists(query_embed_path):
        print(f"🔄 loading query embeddings: {query_embed_path}", flush=True)
        queries_emb = np.load(query_embed_path).astype("float32")
        if queries_emb.shape[0] != len(query_ids):
            print(
                f"⚠️ query embed cache mismatch: embed_Q={queries_emb.shape[0]} vs queries_Q={len(query_ids)}. Rebuilding...",
                flush=True,
            )
            need_rebuild_queries = True
    else:
        need_rebuild_queries = True

    if need_rebuild_queries:
        queries_emb = encoder.encode(
            query_texts,
            batch_size=query_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")
        if normalize:
            queries_emb = _l2norm(queries_emb)

        if query_embed_path:
            os.makedirs(os.path.dirname(query_embed_path), exist_ok=True)
            np.save(query_embed_path, queries_emb)
            print(f"✅ saved query embeddings: {query_embed_path}", flush=True)
    else:
        if normalize:
            queries_emb = _l2norm(queries_emb)

    # Ground truth map
    query_to_gt: Dict[str, List[str]] = {}
    for _, r in pos.iterrows():
        qid = str(r["query_id"])
        pid = str(r["corpus_id"])
        if qid in keep_qids:
            query_to_gt.setdefault(qid, []).append(pid)

    meta = {
        "dataset": dataset,
        "ds_dir": ds_dir,
        "qrels_split": qrels_split,
        "paths": {"corpus": corpus_file, "queries": queries_file, "qrels": qrels_file},
        "counts": {
            "passages": len(passage_ids),
            "queries": len(query_ids),
            "pos_qrels": int(len(pos)),
        },
        "normalized": bool(normalize),
    }
    print(f"✅ loaded: passages={len(passage_ids)} queries={len(query_ids)} split={qrels_split} norm={normalize}", flush=True)
    return embeddings, queries_emb, passage_ids, query_ids, query_to_gt, meta