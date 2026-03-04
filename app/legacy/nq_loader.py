# nq_loader.py
import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def _read_jsonl_ids(path, key="_id"):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line)[key] for line in f]

def load_all(corpus_file, queries_file, qrels_file, device, embed_path=None,
             encoder_name=None, query_batch_size=128):
    """
    Returns:
      embeddings (np.float32 [N,D]),
      queries_emb (np.float32 [Q,D]),
      passage_ids (list[str]),
      query_ids (list[str]),
      query_to_gt (dict[qid] -> list[corpus_id])
    """
    # Encoder
    if encoder_name is None:
        encoder_name = os.getenv("ENCODER_NAME", "all-MiniLM-L6-v2")
    encoder = SentenceTransformer(encoder_name, device=device)

    # Passage IDs
    passage_ids = _read_jsonl_ids(corpus_file)

    # Passage embeddings: load cache or build
    if embed_path and os.path.exists(embed_path):
        print(f"🔄 Loading cached passage embeddings: {embed_path}", flush=True)
        embeddings = np.load(embed_path).astype("float32")
    else:
        print("⚙️ Building passage embeddings (no cache found)...", flush=True)
        texts, passage_ids = [], []
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading corpus"):
                obj = json.loads(line)
                passage_ids.append(obj["_id"])
                texts.append((obj.get("title","") + " " + obj.get("text","")).strip())
        embeddings_chunks = []
        bs = 512
        for i in tqdm(range(0, len(texts), bs), desc="Encoding passages"):
            batch = texts[i:i+bs]
            emb = encoder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings_chunks.append(emb)
        embeddings = np.vstack(embeddings_chunks).astype("float32")
        if embed_path:
            os.makedirs(os.path.dirname(embed_path), exist_ok=True)
            np.save(embed_path, embeddings)
            print(f"✅ Saved passage embeddings to: {embed_path}", flush=True)

    # Queries with positive qrels only
    dev_qrels = pd.read_csv(qrels_file, sep="\t", header=0,
                            names=["query_id","corpus_id","score"], dtype=str)
    pos_qrels = dev_qrels[dev_qrels.score != "0"].reset_index(drop=True)
    keep_qids = set(pos_qrels["query_id"].unique().tolist())

    queries = [json.loads(l) for l in open(queries_file, "r", encoding="utf-8")]
    queries = [q for q in queries if str(q["_id"]) in keep_qids]
    query_ids = [str(q["_id"]) for q in queries]
    query_texts = [q["text"] for q in queries]

    queries_emb = encoder.encode(query_texts, batch_size=query_batch_size,
                                 show_progress_bar=True, convert_to_numpy=True).astype("float32")

    # Ground truth map
    query_to_gt = {}
    for _, r in pos_qrels.iterrows():
        qid = r.query_id
        if qid in keep_qids:
            query_to_gt.setdefault(qid, []).append(r.corpus_id)

    print(f"✅ Ready with {len(passage_ids)} passages and {len(query_ids)} queries.", flush=True)
    return embeddings, queries_emb, passage_ids, query_ids, query_to_gt
