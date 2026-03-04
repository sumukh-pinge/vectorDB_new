import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.autonotebook import tqdm

# --- Metric ---
def compute_recall(query_ids, retrieved_ids, query_to_gt, ks):
    out = {}
    for k in ks:
        rr = []
        for qid, preds in zip(query_ids, retrieved_ids):
            gt = set(query_to_gt.get(qid, []))
            if not gt: continue
            rr.append(len(gt & set(preds[:k])) / len(gt))
        out[f"R@{k}"] = (sum(rr) / len(rr)) if rr else 0.0
    return out

def compute_hit_at_k(query_ids, retrieved_ids, query_to_gt, ks):
    out = {}
    for k in ks:
        hits = 0; denom = 0
        for qid, preds in zip(query_ids, retrieved_ids):
            gt = set(query_to_gt.get(qid, []))
            if not gt:
                continue
            denom += 1
            if any(pid in gt for pid in preds[:k]):
                hits += 1
        out[f"H@{k}"] = (hits / denom) if denom else 0.0
    return out

def compute_mrr(query_ids, retrieved_ids, query_to_gt, ks):
    out = {}
    for k in ks:
        rr = []
        for qid, preds in zip(query_ids, retrieved_ids):
            gt = set(query_to_gt.get(qid, []))
            if not gt: continue
            for rank, pid in enumerate(preds[:k], 1):
                if pid in gt:
                    rr.append(1.0 / rank); break
            else:
                rr.append(0.0)
        out[f"MRR@{k}"] = (sum(rr) / len(rr)) if rr else 0.0
    return out

# --- DBAM core ---
def dbam_direct(q_code, base_q, alpha, m, g):
    bg = base_q.reshape(-1, g, m)
    qg = q_code.reshape(g, m)
    ub = np.all(bg <= (qg + alpha), axis=2)
    lb = np.any(bg >= (qg - alpha), axis=2)
    return ub.sum(axis=1) + lb.sum(axis=1)

def dbam_dual(q_code, base_q, base_dual, alpha, m, g, levels):
    bg = base_q.reshape(-1, g, m)
    bd = base_dual.reshape(-1, g, m)
    qg = q_code.reshape(g, m)
    ub_orig = np.all(bg <= (qg + alpha), axis=2)
    ub_dual = np.all(bd <= ((levels - 1 - qg) + alpha), axis=2)
    return ub_orig.sum(axis=1) + ub_dual.sum(axis=1)

def quantize_np_perdim_minmax(X, vmin_vec, vmax_vec, levels):
    rng_vec  = np.maximum(vmax_vec - vmin_vec, 1e-8)
    Y = (X - vmin_vec) / rng_vec * (levels - 1)
    return np.clip(np.rint(Y), 0, levels - 1).astype(np.int32)

# --- Adapter module ---
class Adapter(nn.Module):
    def __init__(self, in_dim=384, bottleneck=64, act="gelu"):
        super().__init__()
        self.ln   = nn.LayerNorm(in_dim)
        self.down = nn.Linear(in_dim, bottleneck, bias=False)
        self.up   = nn.Linear(bottleneck, in_dim, bias=False)
        self.act  = {"gelu": nn.GELU(), "silu": nn.SiLU(), "relu": nn.ReLU()}[act]
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        y = self.down(self.ln(x))
        y = self.act(y)
        y = self.up(y)
        return x + y

# --- InfoNCE loss (original) ---
def info_nce_loss(query_vecs, pos_vecs, all_passages, temperature=0.07):
    query_vecs = F.normalize(query_vecs, p=2, dim=1)
    pos_vecs   = F.normalize(pos_vecs, p=2, dim=1)
    all_passages = F.normalize(all_passages, p=2, dim=1)
    pos_sim = (query_vecs * pos_vecs).sum(dim=1, keepdim=True)
    logits_neg = query_vecs @ all_passages.t()
    logits = torch.cat([pos_sim, logits_neg], dim=1) / temperature
    labels = torch.zeros(query_vecs.size(0), dtype=torch.long, device=query_vecs.device)
    return F.cross_entropy(logits, labels)

# --- InfoNCE loss with prenormalized negatives (identical math) ---
def info_nce_loss_with_prenorm(query_vecs, pos_vecs, negs_prenorm, temperature=0.07):
    q = F.normalize(query_vecs, p=2, dim=1)
    p = F.normalize(pos_vecs,   p=2, dim=1)
    pos_sim = (q * p).sum(dim=1, keepdim=True)
    logits_neg = q @ negs_prenorm.t()
    logits = torch.cat([pos_sim, logits_neg], dim=1) / temperature
    labels = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
    return F.cross_entropy(logits, labels)

# --- One-time global hard negatives (identical selection to your old loop) ---
def build_global_negatives_once(query_ids_sample, query_to_gt, pipeline_data,
                                K_final, SELECT_NPROBE, ALPHAS_INFER, MS_INFER):
    params = {
        "stages": ("dual", "direct", "dual"),
        "alphas": ALPHAS_INFER,
        "ms": MS_INFER,
        "k_vals": (1000, K_final),
        "nprobe": SELECT_NPROBE,
    }
    all_negs = []
    for qi, qid in tqdm(enumerate(query_ids_sample),
                        total=len(query_ids_sample),
                        desc="Retrieval (build global negatives)"):
        qid = str(qid)
        # q_vec isn't consulted by ("dual","direct","dual"), keep API shape
        q_vec_dummy = pipeline_data["queries_float"][qi]
        retrieved = retrieve_pipeline(
            q_vec_dummy, pipeline_data["queries_q"][qi], params, pipeline_data
        )
        pos_set = set(query_to_gt.get(qid, []))
        negs = [pid for pid in retrieved[:K_final] if pid not in pos_set]
        if len(negs) == 0:
            rng = np.random.default_rng()
            negs = rng.choice(pipeline_data["passage_ids_sample"], size=K_final, replace=False).tolist()
        all_negs.extend(negs)  # keep as ints
    return all_negs

# --- Train adapter (fast, accuracy-identical) ---
def train_W_param(emb_np, que_np, d_out, save_path, passage_ids_sample,
                  query_ids_sample, query_to_gt,
                  device, epochs, lr, subset, q_batch,
                  tau, beta, gamma, grad_clip, verbose_every=10,
                  pipeline_data=None, ALPHAS_INFER=(2,2,2), MS_INFER=(4,4,1),
                  SELECT_NPROBE=16, K_final=10):
    """
    Adapter training with a prebuilt global negatives bank.
    Retrieval config is unchanged; results are identical to the original code.
    """

    Nb, D = emb_np.shape
    Q_all = torch.from_numpy(que_np).to(device=device, dtype=torch.float32)
    passage_full = torch.from_numpy(emb_np).to(device=device, dtype=torch.float32)

    # Let FAISS use reasonable threading (no effect on results)
    try:
        faiss.omp_set_num_threads(max(1, os.cpu_count() // 2))
    except Exception:
        pass

    # passage id → index mapping
    pid_to_idx = {str(pid): i for i, pid in enumerate(passage_ids_sample)}
    # query id → row mapping
    qid_to_row = {str(qid): i for i, qid in enumerate(query_ids_sample)}

    # positive pairs
    train_qidx, train_pidx = [], []
    for qid in query_ids_sample:
        qid = str(qid)
        pos_list = query_to_gt.get(qid, [])
        for pid in pos_list:
            idx = pid_to_idx.get(str(pid), None)
            if idx is not None:
                train_qidx.append(qid)
                train_pidx.append(idx)
    train_q = torch.tensor([qid_to_row[q] for q in train_qidx], dtype=torch.long)
    train_p = torch.tensor(train_pidx, dtype=torch.long)

    # Adapter
    adapter = Adapter(in_dim=D, bottleneck=64).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)

    steps_per_epoch = int(np.ceil(len(train_q) / q_batch))
    global_step = 0

    # --- NEW: build the global negative pool ONCE (identical set to old per-epoch mining) ---
    all_negs = build_global_negatives_once(
        query_ids_sample=query_ids_sample,
        query_to_gt=query_to_gt,
        pipeline_data=pipeline_data,
        K_final=K_final,
        SELECT_NPROBE=SELECT_NPROBE,
        ALPHAS_INFER=ALPHAS_INFER,
        MS_INFER=MS_INFER,
    )
    # map to indices (pid_to_idx keys are strings)
    neg_indices = [pid_to_idx[str(n)] if not isinstance(n, str) else pid_to_idx[n]
                   for n in all_negs if (str(n) in pid_to_idx)]
    X_neg = passage_full[neg_indices].contiguous()
    with torch.no_grad():
        X_neg_norm = F.normalize(X_neg, p=2, dim=1)  # prenormalize once

    for ep in range(epochs):
        print(f"\n⚙️ [Epoch {ep+1}/{epochs}] Training with prebuilt negatives...", flush=True)

        # --- Training loop ---
        perm = torch.randperm(len(train_q))
        tq = train_q[perm]
        tp = train_p[perm]
        running = 0.0

        pbar = tqdm(range(steps_per_epoch), desc=f"Adapter Epoch {ep+1}/{epochs}")
        for i in pbar:
            b0 = i*q_batch; b1 = min((i+1)*q_batch, len(tq))
            if b0 >= b1: break

            q_idx = tq[b0:b1].to(device)
            p_idx = tp[b0:b1].to(device)

            q_vecs = Q_all[q_idx]
            pos_vecs = passage_full[p_idx]
            q_proj = adapter(q_vecs)

            # InfoNCE with prenormalized global negatives (identical logits)
            loss = info_nce_loss_with_prenorm(q_proj, pos_vecs, X_neg_norm, temperature=0.07)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), grad_clip)
            opt.step()
            running += loss.item()
            global_step += 1
            pbar.set_postfix(loss=loss.item())

        print(f"[train] epoch {ep+1}/{epochs} avg_loss={running/max(1,steps_per_epoch):.4f}", flush=True)

    # Save adapter
    base = os.path.splitext(save_path)[0]
    adapter_path = base + "_adapter.pt"
    torch.save({"state_dict": adapter.state_dict(), "in_dim": D, "bottleneck": 64}, adapter_path)
    print(f"✅ Saved Adapter: {adapter_path}", flush=True)

    # Save dummy W (identity matrix)
    W_np = np.eye(D, dtype=np.float32)
    np.save(save_path, W_np)
    print(f"✅ Saved identity W: {save_path}", flush=True)
    return W_np, [{"avg_loss": float(running/max(1,steps_per_epoch))}]


def build_pipeline_with_W(W_np, trial_slug, embeddings, queries_emb_sample, passage_ids_sample,
                          nlist, bits_sq, intermediate_dir, device,
                          require_adapter=True, fallback_to_baseline=False):
    """
    Build a retrieval pipeline using a previously trained Adapter (identified by trial_slug).
    - If the adapter checkpoint is missing:
        * fallback_to_baseline=True  -> build and return a baseline pipeline (no adapter)
        * require_adapter=True (default) -> raise a clear FileNotFoundError
    """
    import os
    import numpy as np
    import pickle
    import torch
    import faiss

    dim = embeddings.shape[1]
    base = os.path.join(intermediate_dir, f"W_{trial_slug}")
    adapter_path = base + "_adapter.pt"

    # --- Handle "no adapter found" cases
    if not os.path.exists(adapter_path):
        msg = f"No adapter checkpoint found: {adapter_path}"
        if fallback_to_baseline:
            print(f"⚠️ {msg} -> Falling back to baseline pipeline.", flush=True)
            return build_pipeline_baseline(
                slug=f"baseline_{trial_slug}",
                embeddings=embeddings,
                queries_emb_sample=queries_emb_sample,
                passage_ids_sample=passage_ids_sample,
                bits_sq=bits_sq,
                nlist=nlist,
            ), None
        if require_adapter:
            raise FileNotFoundError(msg)
        else:
            print(f"⚠️ {msg} -> Proceeding WITHOUT adapter (queries will be unchanged).", flush=True)
            adapter = None
    else:
        # --- Load adapter
        ckpt = torch.load(adapter_path, map_location=device)
        adapter = Adapter(in_dim=ckpt["in_dim"], bottleneck=ckpt["bottleneck"]).to(device)
        adapter.load_state_dict(ckpt["state_dict"])
        adapter.eval()

    # --- Apply adapter to queries (or keep as-is)
    with torch.no_grad():
        q_tensor = torch.from_numpy(queries_emb_sample).to(device)
        if adapter is not None:
            queries_W = adapter(q_tensor).detach().cpu().numpy().astype("float32")
        else:
            queries_W = q_tensor.detach().cpu().numpy().astype("float32")
    embeddings_W = embeddings  # passages unchanged

    # --- IVF index
    quantizer = faiss.IndexFlatL2(dim)
    ivf_flat = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    if not ivf_flat.is_trained:
        ivf_flat.train(embeddings_W)
    ivf_flat.add(embeddings_W)

    # --- Cluster assignments (passages -> nearest centroid)
    _, base_cids = ivf_flat.quantizer.search(embeddings_W, 1)  # (N,1)
    base_cids = base_cids.reshape(-1)
    cluster_to_idxs = {}
    for idx, cid in enumerate(base_cids):
        cluster_to_idxs.setdefault(int(cid), []).append(int(idx))

    # --- Quantization params (per-dim min/max from passages)
    vmin_vec = embeddings_W.min(axis=0, keepdims=True).astype("float32")
    vmax_vec = embeddings_W.max(axis=0, keepdims=True).astype("float32")
    levels = 2 ** bits_sq

    def quantize_np_perdim(X):
        return quantize_np_perdim_minmax(X, vmin_vec, vmax_vec, levels)

    # --- Centroids (robust reconstruct)
    qz = faiss.downcast_index(ivf_flat.quantizer)
    try:
        centroids = qz.reconstruct_n(0, nlist)
    except AttributeError:
        centroids = np.empty((nlist, dim), dtype="float32")
        for i in range(nlist):
            centroids[i] = qz.reconstruct(i)

    # --- Quantize base / queries / centroids (contiguous for speed)
    base_q      = np.ascontiguousarray(quantize_np_perdim(embeddings_W), dtype=np.int32)
    queries_q   = np.ascontiguousarray(quantize_np_perdim(queries_W),   dtype=np.int32)
    centroids_q = np.ascontiguousarray(quantize_np_perdim(centroids),   dtype=np.int32)

    pipeline_data = {
        "ivf_flat": ivf_flat,
        "cluster_to_idxs": cluster_to_idxs,
        "embeddings": np.ascontiguousarray(embeddings_W, dtype=np.float32),
        "queries_float": np.ascontiguousarray(queries_W, dtype=np.float32),
        "passage_ids_sample": passage_ids_sample,
        "levels": int(levels),
        "base_q": base_q,
        "base_dual": np.ascontiguousarray((levels - 1) - base_q, dtype=np.int32),
        "queries_q": queries_q,
        "centroids_q": centroids_q,
        "centroids_dual": np.ascontiguousarray((levels - 1) - centroids_q, dtype=np.int32),
        "W": W_np,
        "quant_params": {"vmin": vmin_vec, "vmax": vmax_vec},
        "quant_mode": "per_dim_minmax",
        "bits_sq": int(bits_sq),
        "nlist": int(nlist),
        "adapter_path": adapter_path if os.path.exists(adapter_path) else None,
    }

    preproc_path = os.path.join(intermediate_dir, f"preproc_W_{trial_slug}.pkl")
    with open(preproc_path, "wb") as f:
        pickle.dump(pipeline_data, f)
    print(f"✅ Cached pipeline: {preproc_path}", flush=True)
    return pipeline_data, preproc_path


def build_pipeline_baseline(slug, embeddings, queries_emb_sample, passage_ids_sample,
                            bits_sq, nlist):
    print(f"[baseline] Build pipeline (no adapter): {slug}", flush=True)
    D = embeddings.shape[1]
    X = np.ascontiguousarray(embeddings.astype('float32', copy=False))
    Q = np.ascontiguousarray(queries_emb_sample.astype('float32', copy=False))
    quantizer = faiss.IndexFlatL2(D)
    ivf = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_L2)
    if not ivf.is_trained:
        ivf.train(X)
    ivf.add(X)

    qz = faiss.downcast_index(ivf.quantizer)
    try:
        centroids = qz.reconstruct_n(0, nlist)
    except AttributeError:
        centroids = np.empty((nlist, D), dtype='float32')
        for i in range(nlist):
            centroids[i] = qz.reconstruct(i)

    levels = 2 ** bits_sq
    x_min = X.min(axis=0); x_max = X.max(axis=0)
    q_min = Q.min(axis=0); q_max = Q.max(axis=0)

    base_q       = np.ascontiguousarray(quantize_np_perdim_minmax(X,         x_min, x_max, levels), dtype=np.int32)
    centroids_q  = np.ascontiguousarray(quantize_np_perdim_minmax(centroids, x_min, x_max, levels), dtype=np.int32)
    queries_q    = np.ascontiguousarray(quantize_np_perdim_minmax(Q,         q_min, q_max, levels), dtype=np.int32)

    base_dual      = np.ascontiguousarray((levels - 1 - base_q), dtype=np.int32)
    centroids_dual = np.ascontiguousarray((levels - 1 - centroids_q), dtype=np.int32)

    _, x_cids = ivf.quantizer.search(X, 1)
    x_cids = x_cids.reshape(-1)
    cluster_to_idxs = {}
    for i, cid in enumerate(x_cids):
        cluster_to_idxs.setdefault(int(cid), []).append(int(i))

    pipeline_data = {
        "ivf_flat": ivf,
        "embeddings": X,
        "base_q": base_q,
        "base_dual": base_dual,
        "centroids_q": centroids_q,
        "centroids_dual": centroids_dual,
        "queries_float": Q,
        "queries_q": queries_q,
        "cluster_to_idxs": cluster_to_idxs,
        "passage_ids_sample": passage_ids_sample,
        "levels": int(levels),
    }
    print("[baseline] done.", flush=True)
    return pipeline_data


def retrieve_pipeline(q_vec, q_code, params, data):
    s1, s2, s3 = params["stages"]
    a1, a2, a3 = params["alphas"]
    m1, m2, m3 = params["ms"]
    g1, g2, g3 = q_code.size // m1, q_code.size // m2, q_code.size // m3
    nprobe = params["nprobe"]
    K2, K_final = params["k_vals"]

    if s1 == "ivf":
        selected_clusters = data["ivf_flat"].quantizer.search(q_vec[None, :], nprobe)[1][0]
    elif s1 == "dual":
        sc = dbam_dual(q_code, data["centroids_q"], data["centroids_dual"], a1, m1, g1, data["levels"])
        selected_clusters = np.argsort(-sc)[:nprobe]
    elif s1 == "ivf_int4":
        diffs = data["centroids_q"] - q_code   # INT4 quantized centroids vs quantized query
        order = np.argsort(np.einsum("ij,ij->i", diffs, diffs))
        selected_clusters = order[:nprobe]
    else:
        sc = dbam_direct(q_code, data["centroids_q"], a1, m1, g1)
        selected_clusters = np.argsort(-sc)[:nprobe]

    candidate_indices = np.array([idx for cid in selected_clusters
                                  for idx in data["cluster_to_idxs"].get(int(cid), [])],
                                 dtype=np.int64)
    if candidate_indices.size == 0:
        return []

    if s2 == "ivf":
        diffs = data["embeddings"][candidate_indices] - q_vec
        order = np.argsort(np.einsum("ij,ij->i", diffs, diffs))
    elif s2 == "direct":
        order = np.argsort(-dbam_direct(q_code, data["base_q"][candidate_indices], a2, m2, g2))
    elif s2 == "ivf_int4":
        diffs = data["base_q"][candidate_indices] - q_code  # INT4 passages vs quantized query
        order = np.argsort(np.einsum("ij,ij->i", diffs, diffs))
    else:
        order = np.argsort(-dbam_dual(q_code, data["base_q"][candidate_indices],
                                      data["base_dual"][candidate_indices], a2, m2, g2, data["levels"]))
    stage2_candidates = candidate_indices[order[:min(len(order), K2)]]

    if s3 == "ivf":
        diffs = data["embeddings"][stage2_candidates] - q_vec
        order = np.argsort(np.einsum("ij,ij->i", diffs, diffs))
    elif s3 == "ivf_int4":
        diffs = data["base_q"][stage2_candidates] - q_code
        order = np.argsort(np.einsum("ij,ij->i", diffs, diffs))
    elif s3 == "direct":
        order = np.argsort(-dbam_direct(q_code, data["base_q"][stage2_candidates], a3, m3, g3))
    elif s3 == "dual":
        order = np.argsort(-dbam_dual(q_code, data["base_q"][stage2_candidates],
                                      data["base_dual"][stage2_candidates], a3, m3, g3, data["levels"]))
    else:
        s_dir  = dbam_direct(q_code, data["base_q"][stage2_candidates], a3, m3, g3)
        s_dual = dbam_dual(q_code, data["base_q"][stage2_candidates],
                           data["base_dual"][stage2_candidates], a3, m3, g3, data["levels"])
        order = np.argsort(-(s_dir + s_dual))

    final_indices = stage2_candidates[order[:min(len(order), K_final)]]
    return [data["passage_ids_sample"][i] for i in final_indices]

# --- Cache helpers for reusing the same adapter ---
def is_cache_stale(adapter_path, preproc_path):
    """Return True if preproc cache is missing or older than the adapter checkpoint."""
    if not os.path.exists(preproc_path):
        return True
    if not os.path.exists(adapter_path):
        return True
    return os.path.getmtime(preproc_path) < os.path.getmtime(adapter_path)

def load_or_build_pipeline_for_adapter(trial_slug, embeddings, queries_emb_sample, passage_ids_sample,
                                       nlist, bits_sq, intermediate_dir, device,
                                       fallback_to_baseline=False):
    """
    Reuse the same adapter (by trial_slug):
    - If preproc cache is up to date: load and return it.
    - Else: rebuild via build_pipeline_with_W.
    """
    adapter_path = os.path.join(intermediate_dir, f"W_{trial_slug}_adapter.pt")
    preproc_path = os.path.join(intermediate_dir, f"preproc_W_{trial_slug}2.pkl")

    if not is_cache_stale(adapter_path, preproc_path):
        with open(preproc_path, "rb") as f:
            pipeline_data = pickle.load(f)
        print(f"🔄 Loaded up-to-date cache: {preproc_path}", flush=True)
        return pipeline_data, preproc_path

    print("⚙️ Cache missing or stale. Rebuilding pipeline...", flush=True)
    W_dummy = np.eye(embeddings.shape[1], dtype=np.float32)  # W isn’t used; identity is fine
    pipeline_data, preproc_path = build_pipeline_with_W(
        W_np=W_dummy,
        trial_slug=trial_slug,
        embeddings=embeddings,
        queries_emb_sample=queries_emb_sample,
        passage_ids_sample=passage_ids_sample,
        nlist=nlist,
        bits_sq=bits_sq,
        intermediate_dir=intermediate_dir,
        device=device,
        fallback_to_baseline=fallback_to_baseline,
    )
    return pipeline_data, preproc_path

def run_and_evaluate(config, output_dir, pipeline_data, query_ids_sample, query_to_gt):
    results_list = []
    s1, s2, s3 = config["stage_methods"]["s1"], config["stage_methods"]["s2"], config["stage_methods"]["s3"]
    k_final_values = config.get("k_final_values", [100])
    K_final = max(k_final_values)

    def _print_row(tag, row_dict):
        fields = ["nprobe", "K2"] + [f"R@{k}" for k in k_final_values] + [f"MRR@{k}" for k in k_final_values]
        s = " | ".join([f"{f}={row_dict.get(f, '')}" for f in fields])
        print(f"[{tag}] {s}", flush=True)

    if "k2_sweep_values" in config:
        nprobe = config["nprobe"]
        for k2 in config["k2_sweep_values"]:
            row = {"S1": s1, "S2": s2, "S3": s3, "nprobe": nprobe, "K2": k2}
            params = {"stages": (s1, s2, s3), "alphas": config["alphas"],
                      "ms": config["ms"], "k_vals": (k2, K_final), "nprobe": nprobe}
            retrieved_all = [
                retrieve_pipeline(pipeline_data["queries_float"][i], pipeline_data["queries_q"][i],
                                  params, pipeline_data)
                for i in tqdm(range(len(pipeline_data["queries_float"])), desc=f"Stage sweep K2={k2}")
            ]
            rec = compute_recall(query_ids_sample, retrieved_all, query_to_gt, k_final_values)
            mrr = compute_mrr(query_ids_sample, retrieved_all, query_to_gt, k_final_values)
            hit = compute_hit_at_k(query_ids_sample, retrieved_all, query_to_gt, k_final_values)
            row.update(hit)
            row.update(rec); row.update(mrr)
            results_list.append(row)
            _print_row("row", row)

    elif "nprobe_sweep_values" in config:
        k2 = config["k2_fixed"]
        for nprobe in config["nprobe_sweep_values"]:
            row = {"S1": s1, "S2": s2, "S3": s3, "nprobe": nprobe, "K2": k2}
            params = {"stages": (s1, s2, s3), "alphas": config["alphas"],
                      "ms": config["ms"], "k_vals": (k2, K_final), "nprobe": nprobe}
            retrieved_all = [
                retrieve_pipeline(pipeline_data["queries_float"][i], pipeline_data["queries_q"][i],
                                  params, pipeline_data)
                for i in tqdm(range(len(pipeline_data["queries_float"])), desc=f"nprobe={nprobe}")
            ]
            rec = compute_recall(query_ids_sample, retrieved_all, query_to_gt, k_final_values)
            mrr = compute_mrr(query_ids_sample, retrieved_all, query_to_gt, k_final_values)
            hit = compute_hit_at_k(query_ids_sample, retrieved_all, query_to_gt, k_final_values)
            row.update(hit)
            row.update(rec); row.update(mrr)
            results_list.append(row)
            _print_row("row", row)
    else:
        print("❌ Configuration missing sweep type.")
        return None

    results_df = pd.DataFrame(results_list)
    out_csv = os.path.join(output_dir, f"{config['experiment_name']}_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n✅ Results saved to: {out_csv}", flush=True)
    return results_df

# quick eval for ivf pipeline
def quick_eval_for_pipeline_ivf(pipeline_data, trial_slug, SELECT_NPROBE, ALPHAS_INFER, MS_INFER, K2_FIXED, K_FINALS,
                            results_dir, query_ids_sample, query_to_gt):
    cfg = {
        "experiment_name": f"quick_nprobe{SELECT_NPROBE}_{trial_slug}_ivf",
        "nprobe_sweep_values": [SELECT_NPROBE],
        "stage_methods": { "s1": "ivf", "s2": "ivf", "s3": "ivf" },
        "alphas": ALPHAS_INFER,
        "ms": MS_INFER,
        "k2_fixed": K2_FIXED,
        "k_final_values": K_FINALS
    }
    df = run_and_evaluate(cfg, results_dir, pipeline_data, query_ids_sample, query_to_gt)
    row = df.iloc[0].to_dict()
    score = (row["R@10"], row["MRR@10"])
    print(f"[select] {trial_slug} -> R@10={row['R@10']:.4f}, MRR@10={row['MRR@10']:.4f}", flush=True)
    return score, row

# quick eval for ivf_int4 pipeline
def quick_eval_for_pipeline_ivf_int4(pipeline_data, trial_slug, SELECT_NPROBE, ALPHAS_INFER, MS_INFER, K2_FIXED, K_FINALS,
                            results_dir, query_ids_sample, query_to_gt):
    cfg = {
        "experiment_name": f"quick_nprobe{SELECT_NPROBE}_{trial_slug}_ivf",
        "nprobe_sweep_values": [SELECT_NPROBE],
        "stage_methods": { "s1": "ivf_int4", "s2": "ivf_int4", "s3": "ivf_int4" },
        "alphas": ALPHAS_INFER,
        "ms": MS_INFER,
        "k2_fixed": K2_FIXED,
        "k_final_values": K_FINALS
    }
    df = run_and_evaluate(cfg, results_dir, pipeline_data, query_ids_sample, query_to_gt)
    row = df.iloc[0].to_dict()
    score = (row["R@10"], row["MRR@10"])
    print(f"[select] {trial_slug} -> R@10={row['R@10']:.4f}, MRR@10={row['MRR@10']:.4f}", flush=True)
    return score, row

# quick eval for dual direct dual pipeline
def quick_eval_for_pipeline_ddd(pipeline_data, trial_slug, SELECT_NPROBE, ALPHAS_INFER, MS_INFER, K2_FIXED, K_FINALS,
                            results_dir, query_ids_sample, query_to_gt):
    cfg = {
        "experiment_name": f"quick_nprobe{SELECT_NPROBE}_{trial_slug}_ddd",
        "nprobe_sweep_values": [SELECT_NPROBE],
        "stage_methods": { "s1": "dual", "s2": "direct", "s3": "dual" },
        "alphas": ALPHAS_INFER,
        "ms": MS_INFER,
        "k2_fixed": K2_FIXED,
        "k_final_values": K_FINALS
    }
    df = run_and_evaluate(cfg, results_dir, pipeline_data, query_ids_sample, query_to_gt)
    row = df.iloc[0].to_dict()
    score = (row["R@10"], row["MRR@10"])
    print(f"[select] {trial_slug} -> R@10={row['R@10']:.4f}, MRR@10={row['MRR@10']:.4f}", flush=True)
    return score, row

# quick eval for dual dual dual pipeline
def quick_eval_for_pipeline_dual(pipeline_data, trial_slug, SELECT_NPROBE, ALPHAS_INFER, MS_INFER, K2_FIXED, K_FINALS,
                            results_dir, query_ids_sample, query_to_gt):
    cfg = {
        "experiment_name": f"quick_nprobe{SELECT_NPROBE}_{trial_slug}_ddd",
        "nprobe_sweep_values": [SELECT_NPROBE],
        "stage_methods": { "s1": "dual", "s2": "dual", "s3": "dual" },
        "alphas": ALPHAS_INFER,
        "ms": MS_INFER,
        "k2_fixed": K2_FIXED,
        "k_final_values": K_FINALS
    }
    df = run_and_evaluate(cfg, results_dir, pipeline_data, query_ids_sample, query_to_gt)
    row = df.iloc[0].to_dict()
    score = (row["R@10"], row["MRR@10"])
    print(f"[select] {trial_slug} -> R@10={row['R@10']:.4f}, MRR@10={row['MRR@10']:.4f}", flush=True)
    return score, row

# quick eval for direct direct direct pipeline
def quick_eval_for_pipeline_direct(pipeline_data, trial_slug, SELECT_NPROBE, ALPHAS_INFER, MS_INFER, K2_FIXED, K_FINALS,
                            results_dir, query_ids_sample, query_to_gt):
    cfg = {
        "experiment_name": f"quick_nprobe{SELECT_NPROBE}_{trial_slug}_direct",
        "nprobe_sweep_values": [SELECT_NPROBE],
        "stage_methods": { "s1": "direct", "s2": "direct", "s3": "direct" },
        "alphas": ALPHAS_INFER,
        "ms": MS_INFER,
        "k2_fixed": K2_FIXED,
        "k_final_values": K_FINALS
    }
    df = run_and_evaluate(cfg, results_dir, pipeline_data, query_ids_sample, query_to_gt)
    row = df.iloc[0].to_dict()
    score = (row["R@10"], row["MRR@10"])
    print(f"[select] {trial_slug} -> R@10={row['R@10']:.4f}, MRR@10={row['MRR@10']:.4f}", flush=True)
    return score, row
