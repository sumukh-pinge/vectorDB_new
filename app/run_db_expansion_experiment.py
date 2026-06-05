#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import fcntl
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm

from utilis_dbam_v3_new import (
    build_pipeline_baseline,
    load_or_build_pipeline_for_adapter,
    run_and_evaluate,
    train_W_param,
)


def parse_int_tuple(text):
    vals = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if len(vals) != 3:
        raise ValueError(f"Expected 3 comma-separated ints, got: {text}")
    return vals


def parse_str_tuple(text):
    vals = tuple(x.strip() for x in text.split(",") if x.strip())
    if len(vals) != 3:
        raise ValueError(f"Expected 3 comma-separated strings, got: {text}")
    return vals


def parse_int_list(text):
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer")
    return sorted(set(vals))


def parse_ms_values(text):
    out = []
    for part in text.split(";"):
        part = part.strip()
        if part:
            out.append(parse_int_tuple(part))
    if not out:
        raise ValueError("Expected at least one DTS m tuple")
    return out


def ms_label(ms):
    return "DTS" + "".join(str(x) for x in ms)


def frac_tag(value):
    return str(value).replace(".", "p")


def l2norm(x, eps=1e-12):
    denom = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
    return (x / denom).astype("float32")


@contextmanager
def file_lock(lock_path):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def resolve_dataset_dir(data_root, dataset):
    ds_dir = Path(data_root) / dataset
    if not (ds_dir / "corpus.jsonl").exists() and (ds_dir / dataset / "corpus.jsonl").exists():
        ds_dir = ds_dir / dataset
    if not (ds_dir / "corpus.jsonl").exists():
        raise FileNotFoundError(f"Could not find corpus.jsonl under {Path(data_root) / dataset}")
    return ds_dir


def choose_qrels_file(ds_dir):
    qrels_dir = ds_dir / "qrels"
    for split in ["test", "dev", "train"]:
        cand = qrels_dir / f"{split}.tsv"
        if cand.exists():
            return split, cand
    raise FileNotFoundError(f"No qrels split found under {qrels_dir}")


def load_or_build_passage_embeddings(corpus_file, embed_path, encoder, batch_size):
    passage_ids = []
    with open(corpus_file, "r", encoding="utf-8") as handle:
        for line in handle:
            passage_ids.append(str(json.loads(line)["_id"]))

    if os.path.exists(embed_path):
        embeddings = np.load(embed_path).astype("float32")
        if embeddings.shape[0] == len(passage_ids):
            return passage_ids, embeddings
        print(
            f"passage cache mismatch: embed_N={embeddings.shape[0]} corpus_N={len(passage_ids)}; rebuilding",
            flush=True,
        )

    with file_lock(embed_path + ".lock"):
        if os.path.exists(embed_path):
            embeddings = np.load(embed_path).astype("float32")
            if embeddings.shape[0] == len(passage_ids):
                return passage_ids, embeddings

        passage_ids = []
        chunks = []
        texts = []
        with open(corpus_file, "r", encoding="utf-8") as handle:
            for line in tqdm(handle, desc=f"Encoding passages -> {os.path.basename(embed_path)}"):
                obj = json.loads(line)
                passage_ids.append(str(obj["_id"]))
                texts.append((obj.get("title", "") + " " + obj.get("text", "")).strip())
                if len(texts) >= batch_size:
                    emb = encoder.encode(
                        texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=batch_size,
                    )
                    chunks.append(emb.astype("float32"))
                    texts = []

        if texts:
            emb = encoder.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )
            chunks.append(emb.astype("float32"))

        embeddings = np.vstack(chunks).astype("float32")
        os.makedirs(os.path.dirname(embed_path), exist_ok=True)
        np.save(embed_path, embeddings)
        return passage_ids, embeddings


def load_qrels(qrels_file):
    qrels = pd.read_csv(qrels_file, sep="\t", header=None, dtype=str, engine="python")
    if qrels.shape[1] >= 3:
        qrels = qrels.iloc[:, :3]
    qrels.columns = ["query_id", "corpus_id", "score"]
    first = qrels.iloc[0].astype(str).str.lower().tolist()
    if "query" in first[0] and "corpus" in first[1]:
        qrels = qrels.iloc[1:].reset_index(drop=True)
    qrels["score_num"] = pd.to_numeric(qrels["score"], errors="coerce").fillna(0)
    return qrels[qrels["score_num"] > 0].reset_index(drop=True)


def load_or_build_query_embeddings(queries_file, qrels_file, qembed_path, encoder, batch_size):
    pos_qrels = load_qrels(qrels_file)
    keep_qids = set(pos_qrels["query_id"].astype(str).unique().tolist())

    queries = [json.loads(line) for line in open(queries_file, "r", encoding="utf-8")]
    queries = [q for q in queries if str(q["_id"]) in keep_qids]
    query_ids = [str(q["_id"]) for q in queries]
    query_texts = [q["text"] for q in queries]

    if os.path.exists(qembed_path):
        queries_emb = np.load(qembed_path).astype("float32")
        if queries_emb.shape[0] != len(query_ids):
            print(
                f"query cache mismatch: embed_Q={queries_emb.shape[0]} query_Q={len(query_ids)}; rebuilding",
                flush=True,
            )
            queries_emb = None
    else:
        queries_emb = None

    if queries_emb is None:
        with file_lock(qembed_path + ".lock"):
            if os.path.exists(qembed_path):
                queries_emb = np.load(qembed_path).astype("float32")
                if queries_emb.shape[0] == len(query_ids):
                    pass
                else:
                    queries_emb = None
            if queries_emb is None:
                queries_emb = encoder.encode(
                    query_texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                ).astype("float32")
                os.makedirs(os.path.dirname(qembed_path), exist_ok=True)
                np.save(qembed_path, queries_emb)

    query_to_gt = {}
    for _, row in pos_qrels.iterrows():
        query_to_gt.setdefault(str(row["query_id"]), []).append(str(row["corpus_id"]))

    return query_ids, queries_emb, query_to_gt, pos_qrels


def deterministic_split(items, train_frac, seed, allow_empty_train=False):
    ordered = np.array(sorted(items), dtype=object)
    if len(ordered) == 0:
        return set(), set()
    perm = np.random.default_rng(seed).permutation(len(ordered))
    ordered = ordered[perm].tolist()
    n_train = int(round(train_frac * len(ordered)))
    if allow_empty_train:
        n_train = max(0, min(len(ordered), n_train))
    else:
        n_train = max(1, min(len(ordered) - 1, n_train))
    return set(ordered[:n_train]), set(ordered[n_train:])


def make_document_expansion_split(passage_ids, query_ids, query_to_gt, holdout_doc_frac, base_train_frac, new_eval_frac, seed):
    passage_ids_sorted = sorted(str(pid) for pid in passage_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(passage_ids_sorted))
    added_count = max(1, int(round(holdout_doc_frac * len(passage_ids_sorted))))
    added_doc_ids = set(passage_ids_sorted[i] for i in perm[:added_count])
    base_doc_ids = set(passage_ids_sorted) - added_doc_ids

    new_doc_qids = []
    base_only_qids = []
    for qid in query_ids:
        positives = set(str(pid) for pid in query_to_gt.get(str(qid), []))
        has_added = bool(positives & added_doc_ids)
        has_base = bool(positives & base_doc_ids)
        if has_added:
            new_doc_qids.append(str(qid))
        elif has_base:
            base_only_qids.append(str(qid))

    if len(new_doc_qids) < 2:
        fallback_train, fallback_eval = deterministic_split(query_ids, 1.0 - holdout_doc_frac, seed)
        forced_eval = fallback_eval or fallback_train
        for qid in forced_eval:
            added_doc_ids.update(str(pid) for pid in query_to_gt.get(str(qid), []))
        base_doc_ids = set(passage_ids_sorted) - added_doc_ids
        new_doc_qids = []
        base_only_qids = []
        for qid in query_ids:
            positives = set(str(pid) for pid in query_to_gt.get(str(qid), []))
            if positives & added_doc_ids:
                new_doc_qids.append(str(qid))
            elif positives & base_doc_ids:
                base_only_qids.append(str(qid))

    refresh_new_train, expanded_eval_new = deterministic_split(
        new_doc_qids,
        train_frac=1.0 - new_eval_frac,
        seed=seed + 17,
        allow_empty_train=True,
    )
    base_train, base_eval = deterministic_split(
        base_only_qids,
        train_frac=base_train_frac,
        seed=seed + 29,
        allow_empty_train=False,
    )

    return {
        "base_doc_ids": base_doc_ids,
        "added_doc_ids": added_doc_ids,
        "base_train_qids": base_train,
        "base_eval_qids": base_eval,
        "refresh_new_train_qids": refresh_new_train,
        "expanded_eval_new_qids": expanded_eval_new,
        "new_doc_qids_all": set(new_doc_qids),
        "base_only_qids_all": set(base_only_qids),
    }


def subset_passages(passage_ids, embeddings, keep_doc_ids):
    idx = [i for i, pid in enumerate(passage_ids) if str(pid) in keep_doc_ids]
    return [str(passage_ids[i]) for i in idx], embeddings[idx]


def subset_queries(query_ids, queries_emb, keep_qids, query_to_gt, allowed_doc_ids):
    idx = [i for i, qid in enumerate(query_ids) if str(qid) in keep_qids]
    out_qids = []
    out_idx = []
    out_q2gt = {}
    for i in idx:
        qid = str(query_ids[i])
        positives = [str(pid) for pid in query_to_gt.get(qid, []) if str(pid) in allowed_doc_ids]
        if positives:
            out_qids.append(qid)
            out_idx.append(i)
            out_q2gt[qid] = positives
    if out_idx:
        return out_qids, queries_emb[out_idx], out_q2gt
    return [], np.empty((0, queries_emb.shape[1]), dtype="float32"), {}


def metric_row(df):
    return df.iloc[0].to_dict()


def eval_mode(name, pipeline, query_ids, query_to_gt, results_dir, nprobe, k2, k_finals, alphas, stages, ms):
    config = {
        "experiment_name": name,
        "nprobe_sweep_values": [nprobe],
        "stage_methods": {"s1": stages[0], "s2": stages[1], "s3": stages[2]},
        "alphas": alphas,
        "ms": ms,
        "k2_fixed": k2,
        "k_final_values": k_finals,
    }
    return metric_row(run_and_evaluate(config, results_dir, pipeline, query_ids, query_to_gt))


def ensure_adapter_alias(intermediate_dir, source_slug, alias_slug):
    source_ckpt = os.path.join(intermediate_dir, f"W_{source_slug}_adapter.pt")
    alias_ckpt = os.path.join(intermediate_dir, f"W_{alias_slug}_adapter.pt")
    if not os.path.exists(source_ckpt):
        raise FileNotFoundError(f"Frozen adapter checkpoint missing: {source_ckpt}")

    source_w = os.path.join(intermediate_dir, f"W_{source_slug}.npy")
    alias_w = os.path.join(intermediate_dir, f"W_{alias_slug}.npy")

    needs_copy = (
        not os.path.exists(alias_ckpt)
        or os.path.getmtime(source_ckpt) > os.path.getmtime(alias_ckpt)
    )
    if needs_copy:
        shutil.copyfile(source_ckpt, alias_ckpt)
        os.utime(alias_ckpt, None)
        if os.path.exists(source_w):
            shutil.copyfile(source_w, alias_w)
            os.utime(alias_w, None)
    return alias_ckpt


def train_adapter_if_needed(slug, embeddings, queries_emb, passage_ids, query_ids, query_to_gt, pipeline_train, cfg):
    w_path = os.path.join(cfg["intermediate_dir"], f"W_{slug}.npy")
    ckpt_path = os.path.join(cfg["intermediate_dir"], f"W_{slug}_adapter.pt")
    if os.path.exists(ckpt_path):
        print(f"Reusing adapter checkpoint: {ckpt_path}", flush=True)
        return ckpt_path

    print(f"Training adapter: {ckpt_path}", flush=True)
    train_W_param(
        emb_np=embeddings,
        que_np=queries_emb,
        d_out=embeddings.shape[1],
        save_path=w_path,
        passage_ids_sample=passage_ids,
        query_ids_sample=query_ids,
        query_to_gt=query_to_gt,
        device=cfg["device"],
        epochs=cfg["adapt_epochs"],
        lr=cfg["adapt_lr"],
        subset=cfg["adapt_subset"],
        q_batch=cfg["adapt_qbatch"],
        tau=cfg["tau"],
        beta=cfg["beta"],
        gamma=cfg["gamma"],
        grad_clip=cfg["grad_clip"],
        pipeline_data=pipeline_train,
        ALPHAS_INFER=cfg["alphas"],
        MS_INFER=cfg["adapter_mining_ms"],
        STAGES_INFER=cfg["adapter_mining_stages"],
        SELECT_NPROBE=cfg["adapter_mining_nprobe"],
        K_final=cfg["adapter_neg_kfinal"],
    )
    return ckpt_path


def final_table_row(setting, adapter, ivf_r100, dts_r100_by_label, expanded_no_gap_by_label):
    row = {
        "Setting": setting,
        "Adapter": adapter,
        "IVF R@100": ivf_r100,
        "DTS111 R@100": dts_r100_by_label.get("DTS111"),
        "DTS111-IVF Gap": None,
        "DTS111 Gap Recovery": None,
        "DTS242 R@100": dts_r100_by_label.get("DTS242"),
        "DTS242-IVF Gap": None,
        "DTS242 Gap Recovery": None,
    }

    for label in ["DTS111", "DTS242"]:
        dts_r100 = dts_r100_by_label.get(label)
        gap_col = f"{label}-IVF Gap"
        rec_col = f"{label} Gap Recovery"
        if dts_r100 is None or ivf_r100 is None:
            continue
        gap = dts_r100 - ivf_r100
        row[gap_col] = gap
        if setting == "Base DB":
            row[rec_col] = "NA"
        elif adapter == "none":
            row[rec_col] = 0.0
        else:
            row[rec_col] = gap - expanded_no_gap_by_label[label]
    return row


def dataframe_to_markdown(df):
    headers = list(df.columns)

    def fmt(value):
        if value is None:
            return ""
        if isinstance(value, float):
            if np.isnan(value):
                return ""
            return f"{value:.6f}"
        return str(value)

    rows = [[fmt(value) for value in row] for row in df.to_numpy().tolist()]
    widths = [
        max(len(str(header)), *(len(row[i]) for row in rows)) if rows else len(str(header))
        for i, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + row_lines) + "\n"


def stage_tags(args):
    encoder_tag = args.encoder.replace("/", "_")
    adapter_mining_nprobe = args.adapter_mining_nprobe or args.eval_nprobe
    split_tag = (
        f"doc{frac_tag(1.0 - args.holdout_doc_frac)}_{frac_tag(args.holdout_doc_frac)}"
        f"_neweval{frac_tag(args.new_eval_frac)}_seed{args.seed}"
        f"_nlist{args.nlist}_b{args.bits_sq}"
    )
    run_tag = f"{split_tag}_np{args.eval_nprobe}_mine{adapter_mining_nprobe}"
    inter_dir = os.path.join(args.intermediate_root, args.dataset, encoder_tag, "db_expansion", run_tag)
    out_dir = os.path.join(
        args.run_root,
        args.dataset,
        encoder_tag,
        "results",
        "db_expansion",
        args.run_label,
        f"np{args.eval_nprobe}",
        run_tag,
    )
    slugs = {
        "base": f"base_adapter_{args.dataset}_{encoder_tag}_{run_tag}",
        "frozen": f"frozen_base_adapter_on_expanded_{args.dataset}_{encoder_tag}_{run_tag}",
        "refreshed": f"refreshed_adapter_{args.dataset}_{encoder_tag}_{run_tag}",
    }
    return encoder_tag, split_tag, run_tag, inter_dir, out_dir, slugs


def build_runtime(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_tag, split_tag, run_tag, inter_dir, out_dir, slugs = stage_tags(args)
    os.makedirs(inter_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    adapter_mining_nprobe = args.adapter_mining_nprobe or args.eval_nprobe
    return {
        "device": device,
        "encoder_tag": encoder_tag,
        "split_tag": split_tag,
        "run_tag": run_tag,
        "inter_dir": inter_dir,
        "out_dir": out_dir,
        "slugs": slugs,
        "k_finals": sorted(set(parse_int_list(args.kfinals)) | {100}),
        "alphas": parse_int_tuple(args.alphas),
        "ms_values": parse_ms_values(args.ms_values),
        "cfg": {
            "device": device,
            "intermediate_dir": inter_dir,
            "alphas": parse_int_tuple(args.alphas),
            "adapter_mining_stages": parse_str_tuple(args.adapter_mining_stages),
            "adapter_mining_ms": parse_int_tuple(args.adapter_mining_ms),
            "adapter_mining_nprobe": adapter_mining_nprobe,
            "adapter_neg_kfinal": args.adapter_neg_kfinal,
            "adapt_epochs": args.adapt_epochs,
            "adapt_lr": args.adapt_lr,
            "adapt_subset": args.adapt_subset,
            "adapt_qbatch": args.adapt_qbatch,
            "tau": args.tau,
            "beta": args.beta,
            "gamma": args.gamma,
            "grad_clip": args.grad_clip,
        },
    }


def artifact_paths(rt):
    inter_dir = rt["inter_dir"]
    return {
        "prep_done": os.path.join(inter_dir, "prep_done.json"),
        "split_meta": os.path.join(inter_dir, "split_meta.json"),
        "base_ckpt": os.path.join(inter_dir, f"W_{rt['slugs']['base']}_adapter.pt"),
        "frozen_ckpt": os.path.join(inter_dir, f"W_{rt['slugs']['frozen']}_adapter.pt"),
        "refreshed_ckpt": os.path.join(inter_dir, f"W_{rt['slugs']['refreshed']}_adapter.pt"),
    }


def wait_for_path(path, timeout_sec, poll_sec=60):
    start = time.time()
    while True:
        if os.path.exists(path):
            return
        elapsed = time.time() - start
        if timeout_sec <= 0 or elapsed > timeout_sec:
            raise TimeoutError(f"Timed out waiting for {path}")
        print(f"waiting for {path} ({int(elapsed)}s elapsed)", flush=True)
        time.sleep(poll_sec)


def load_data(args, rt):
    ds_dir = resolve_dataset_dir(args.data_root, args.dataset)
    qrels_split, qrels_file = choose_qrels_file(ds_dir)
    corpus_file = str(ds_dir / "corpus.jsonl")
    queries_file = str(ds_dir / "queries.jsonl")
    embed_path = os.path.join(
        args.intermediate_root,
        args.dataset,
        rt["encoder_tag"],
        f"passage_embeddings_{args.dataset}_{rt['encoder_tag']}.npy",
    )
    qembed_path = os.path.join(
        args.intermediate_root,
        args.dataset,
        rt["encoder_tag"],
        f"query_embeddings_{args.dataset}_{rt['encoder_tag']}_{qrels_split}.npy",
    )

    print(f"device: {rt['device']}", flush=True)
    print(f"dataset: {args.dataset} qrels_split: {qrels_split}", flush=True)
    print(f"encoder: {args.encoder}", flush=True)
    print(f"intermediate: {rt['inter_dir']}", flush=True)
    print(f"output: {rt['out_dir']}", flush=True)

    encoder = SentenceTransformer(args.encoder, device=rt["device"])
    passage_ids_all, embeddings_all = load_or_build_passage_embeddings(
        corpus_file=corpus_file,
        embed_path=embed_path,
        encoder=encoder,
        batch_size=args.passage_batch_size,
    )
    query_ids_all, queries_emb_all, query_to_gt_all, pos_qrels = load_or_build_query_embeddings(
        queries_file=queries_file,
        qrels_file=str(qrels_file),
        qembed_path=qembed_path,
        encoder=encoder,
        batch_size=args.query_batch_size,
    )
    return {
        "ds_dir": ds_dir,
        "qrels_split": qrels_split,
        "qrels_file": str(qrels_file),
        "corpus_file": corpus_file,
        "queries_file": queries_file,
        "embed_path": embed_path,
        "qembed_path": qembed_path,
        "passage_ids_all": passage_ids_all,
        "embeddings_all": l2norm(embeddings_all),
        "query_ids_all": query_ids_all,
        "queries_emb_all": l2norm(queries_emb_all),
        "query_to_gt_all": query_to_gt_all,
        "pos_qrels": pos_qrels,
    }


def build_split_data(args, data):
    split = make_document_expansion_split(
        passage_ids=data["passage_ids_all"],
        query_ids=data["query_ids_all"],
        query_to_gt=data["query_to_gt_all"],
        holdout_doc_frac=args.holdout_doc_frac,
        base_train_frac=args.base_train_frac,
        new_eval_frac=args.new_eval_frac,
        seed=args.seed,
    )
    base_ids, base_embeddings = subset_passages(data["passage_ids_all"], data["embeddings_all"], split["base_doc_ids"])
    added_ids = sorted(split["added_doc_ids"])
    expanded_ids, expanded_embeddings = data["passage_ids_all"], data["embeddings_all"]
    expanded_doc_ids = set(str(pid) for pid in expanded_ids)
    base_train_qids, base_train_qemb, base_train_q2gt = subset_queries(
        data["query_ids_all"], data["queries_emb_all"], split["base_train_qids"], data["query_to_gt_all"], split["base_doc_ids"]
    )
    base_eval_qids, base_eval_qemb, base_eval_q2gt = subset_queries(
        data["query_ids_all"], data["queries_emb_all"], split["base_eval_qids"], data["query_to_gt_all"], split["base_doc_ids"]
    )
    expanded_eval_qids, expanded_eval_qemb, expanded_eval_q2gt = subset_queries(
        data["query_ids_all"], data["queries_emb_all"], split["expanded_eval_new_qids"], data["query_to_gt_all"], expanded_doc_ids
    )
    expanded_refresh_train_set = set(base_train_qids) | set(split["refresh_new_train_qids"])
    expanded_train_qids, expanded_train_qemb, expanded_train_q2gt = subset_queries(
        data["query_ids_all"], data["queries_emb_all"], expanded_refresh_train_set, data["query_to_gt_all"], expanded_doc_ids
    )

    if not base_train_qids:
        raise RuntimeError("No base training queries with positives in Base DB.")
    if not base_eval_qids:
        raise RuntimeError("No base evaluation queries with positives in Base DB.")
    if not expanded_eval_qids:
        raise RuntimeError("No expanded/new-doc evaluation queries with positives in Expanded DB.")
    if not expanded_train_qids and not args.skip_refreshed:
        raise RuntimeError("No expanded training queries for refreshed adapter.")

    return {
        "split": split,
        "base_ids": base_ids,
        "base_embeddings": base_embeddings,
        "added_ids": added_ids,
        "expanded_ids": expanded_ids,
        "expanded_embeddings": expanded_embeddings,
        "expanded_doc_ids": expanded_doc_ids,
        "base_train_qids": base_train_qids,
        "base_train_qemb": base_train_qemb,
        "base_train_q2gt": base_train_q2gt,
        "base_eval_qids": base_eval_qids,
        "base_eval_qemb": base_eval_qemb,
        "base_eval_q2gt": base_eval_q2gt,
        "expanded_eval_qids": expanded_eval_qids,
        "expanded_eval_qemb": expanded_eval_qemb,
        "expanded_eval_q2gt": expanded_eval_q2gt,
        "expanded_train_qids": expanded_train_qids,
        "expanded_train_qemb": expanded_train_qemb,
        "expanded_train_q2gt": expanded_train_q2gt,
    }


def write_split_meta(args, rt, data, split_data, elapsed):
    paths = artifact_paths(rt)
    meta = {
        "goal": "D-NOVA E-2 database expansion prep artifacts",
        "dataset": args.dataset,
        "encoder": args.encoder,
        "encoder_tag": rt["encoder_tag"],
        "seed": args.seed,
        "run_label": args.run_label,
        "qrels_split": data["qrels_split"],
        "run_tag": rt["run_tag"],
        "settings": {
            "index": "IVF-Flat",
            "nlist": args.nlist,
            "eval_nprobe": args.eval_nprobe,
            "adapter_mining_nprobe": rt["cfg"]["adapter_mining_nprobe"],
            "k2": args.k2,
            "kfinals": rt["k_finals"],
            "alphas": rt["alphas"],
            "ms_values": rt["ms_values"],
            "holdout_doc_frac": args.holdout_doc_frac,
            "base_train_frac": args.base_train_frac,
            "new_eval_frac": args.new_eval_frac,
            "refreshed_adapter_training": "retrained on expanded DB training setup",
            "storage_side_dts_pipeline_changed": False,
        },
        "counts": {
            "passages_total": len(data["passage_ids_all"]),
            "base_docs": len(split_data["base_ids"]),
            "added_docs": len(split_data["added_ids"]),
            "queries_total_with_qrels": len(data["query_ids_all"]),
            "positive_qrels": int(len(data["pos_qrels"])),
            "base_train_queries": len(split_data["base_train_qids"]),
            "base_eval_queries": len(split_data["base_eval_qids"]),
            "refresh_new_doc_train_queries": len(split_data["split"]["refresh_new_train_qids"]),
            "expanded_train_queries": len(split_data["expanded_train_qids"]),
            "expanded_new_doc_eval_queries": len(split_data["expanded_eval_qids"]),
        },
        "adapter_slugs": rt["slugs"],
        "paths": {
            "dataset_dir": str(data["ds_dir"]),
            "corpus_file": data["corpus_file"],
            "queries_file": data["queries_file"],
            "qrels_file": data["qrels_file"],
            "passage_embedding_cache": data["embed_path"],
            "query_embedding_cache": data["qembed_path"],
            "intermediate_dir": rt["inter_dir"],
            "output_dir": rt["out_dir"],
            **paths,
        },
        "timing_sec": {
            "prep": elapsed,
        },
    }
    with open(paths["split_meta"], "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    with open(paths["prep_done"], "w", encoding="utf-8") as handle:
        json.dump({"status": "done", "split_meta": paths["split_meta"], "timestamp": time.time()}, handle, indent=2)
    print("saved:", paths["split_meta"], flush=True)
    print("saved:", paths["prep_done"], flush=True)


def run_prep(args, rt):
    start = time.time()
    data = load_data(args, rt)
    split_data = build_split_data(args, data)

    print("split counts:", flush=True)
    print(f"  base docs: {len(split_data['base_ids'])}", flush=True)
    print(f"  added docs: {len(split_data['added_ids'])}", flush=True)
    print(f"  base train queries: {len(split_data['base_train_qids'])}", flush=True)
    print(f"  base eval queries: {len(split_data['base_eval_qids'])}", flush=True)
    print(f"  refresh new-doc train queries: {len(split_data['split']['refresh_new_train_qids'])}", flush=True)
    print(f"  expanded/new-doc eval queries: {len(split_data['expanded_eval_qids'])}", flush=True)

    pipeline_base_train = build_pipeline_baseline(
        slug=f"base_train_{rt['run_tag']}",
        embeddings=split_data["base_embeddings"],
        queries_emb_sample=split_data["base_train_qemb"],
        passage_ids_sample=split_data["base_ids"],
        bits_sq=args.bits_sq,
        nlist=args.nlist,
    )
    train_adapter_if_needed(
        slug=rt["slugs"]["base"],
        embeddings=split_data["base_embeddings"],
        queries_emb=split_data["base_train_qemb"],
        passage_ids=split_data["base_ids"],
        query_ids=split_data["base_train_qids"],
        query_to_gt=split_data["base_train_q2gt"],
        pipeline_train=pipeline_base_train,
        cfg=rt["cfg"],
    )
    del pipeline_base_train

    ensure_adapter_alias(rt["inter_dir"], rt["slugs"]["base"], rt["slugs"]["frozen"])

    if not args.skip_refreshed:
        pipeline_expanded_train = build_pipeline_baseline(
            slug=f"expanded_train_refreshed_{rt['run_tag']}",
            embeddings=split_data["expanded_embeddings"],
            queries_emb_sample=split_data["expanded_train_qemb"],
            passage_ids_sample=split_data["expanded_ids"],
            bits_sq=args.bits_sq,
            nlist=args.nlist,
        )
        train_adapter_if_needed(
            slug=rt["slugs"]["refreshed"],
            embeddings=split_data["expanded_embeddings"],
            queries_emb=split_data["expanded_train_qemb"],
            passage_ids=split_data["expanded_ids"],
            query_ids=split_data["expanded_train_qids"],
            query_to_gt=split_data["expanded_train_q2gt"],
            pipeline_train=pipeline_expanded_train,
            cfg=rt["cfg"],
        )
        del pipeline_expanded_train

    write_split_meta(args, rt, data, split_data, time.time() - start)


EVAL_SETTINGS = {
    "base_none": ("Base DB", "none"),
    "expanded_none": ("Expanded DB", "none"),
    "expanded_frozen": ("Expanded DB", "frozen base adapter"),
    "expanded_refreshed": ("Expanded DB", "refreshed adapter"),
}


def build_eval_pipeline(args, rt, split_data):
    if args.eval_setting == "base_none":
        pipeline = build_pipeline_baseline(
            slug=f"base_eval_no_adapter_{rt['run_tag']}",
            embeddings=split_data["base_embeddings"],
            queries_emb_sample=split_data["base_eval_qemb"],
            passage_ids_sample=split_data["base_ids"],
            bits_sq=args.bits_sq,
            nlist=args.nlist,
        )
        return pipeline, split_data["base_eval_qids"], split_data["base_eval_q2gt"], None

    if args.eval_setting == "expanded_none":
        pipeline = build_pipeline_baseline(
            slug=f"expanded_no_adapter_{rt['run_tag']}",
            embeddings=split_data["expanded_embeddings"],
            queries_emb_sample=split_data["expanded_eval_qemb"],
            passage_ids_sample=split_data["expanded_ids"],
            bits_sq=args.bits_sq,
            nlist=args.nlist,
        )
        return pipeline, split_data["expanded_eval_qids"], split_data["expanded_eval_q2gt"], None

    if args.eval_setting == "expanded_frozen":
        ensure_adapter_alias(rt["inter_dir"], rt["slugs"]["base"], rt["slugs"]["frozen"])
        pipeline, preproc_path = load_or_build_pipeline_for_adapter(
            trial_slug=rt["slugs"]["frozen"],
            embeddings=split_data["expanded_embeddings"],
            queries_emb_sample=split_data["expanded_eval_qemb"],
            passage_ids_sample=split_data["expanded_ids"],
            nlist=args.nlist,
            bits_sq=args.bits_sq,
            intermediate_dir=rt["inter_dir"],
            device=rt["device"],
            fallback_to_baseline=False,
        )
        return pipeline, split_data["expanded_eval_qids"], split_data["expanded_eval_q2gt"], preproc_path

    if args.eval_setting == "expanded_refreshed":
        pipeline, preproc_path = load_or_build_pipeline_for_adapter(
            trial_slug=rt["slugs"]["refreshed"],
            embeddings=split_data["expanded_embeddings"],
            queries_emb_sample=split_data["expanded_eval_qemb"],
            passage_ids_sample=split_data["expanded_ids"],
            nlist=args.nlist,
            bits_sq=args.bits_sq,
            intermediate_dir=rt["inter_dir"],
            device=rt["device"],
            fallback_to_baseline=False,
        )
        return pipeline, split_data["expanded_eval_qids"], split_data["expanded_eval_q2gt"], preproc_path

    raise ValueError(f"Unsupported eval_setting: {args.eval_setting}")


def run_eval(args, rt):
    paths = artifact_paths(rt)
    wait_for_path(paths["prep_done"], args.wait_timeout_sec, args.wait_poll_sec)
    if args.eval_setting == "expanded_frozen":
        wait_for_path(paths["frozen_ckpt"], args.wait_timeout_sec, args.wait_poll_sec)
    if args.eval_setting == "expanded_refreshed":
        wait_for_path(paths["refreshed_ckpt"], args.wait_timeout_sec, args.wait_poll_sec)

    start = time.time()
    data = load_data(args, rt)
    split_data = build_split_data(args, data)
    pipeline, eval_qids, eval_q2gt, preproc_path = build_eval_pipeline(args, rt, split_data)

    setting, adapter = EVAL_SETTINGS[args.eval_setting]
    setting_slug = setting.lower().replace(" ", "_")
    adapter_slug = adapter.lower().replace(" ", "_")
    rows = []

    ivf_name = f"{args.eval_setting}_np{args.eval_nprobe}_ivf"
    ivf_metrics = eval_mode(
        name=ivf_name,
        pipeline=pipeline,
        query_ids=eval_qids,
        query_to_gt=eval_q2gt,
        results_dir=rt["out_dir"],
        nprobe=args.eval_nprobe,
        k2=args.k2,
        k_finals=rt["k_finals"],
        alphas=rt["alphas"],
        stages=("ivf", "ivf", "ivf"),
        ms=(1, 1, 1),
    )
    rows.append({
        "Setting": setting,
        "Adapter": adapter,
        "Retrieval": "IVF",
        "ms": "1,1,1",
        "eval_setting": args.eval_setting,
        **ivf_metrics,
    })

    for ms in rt["ms_values"]:
        label = ms_label(ms)
        dts_name = f"{args.eval_setting}_np{args.eval_nprobe}_{label.lower()}"
        dts_metrics = eval_mode(
            name=dts_name,
            pipeline=pipeline,
            query_ids=eval_qids,
            query_to_gt=eval_q2gt,
            results_dir=rt["out_dir"],
            nprobe=args.eval_nprobe,
            k2=args.k2,
            k_finals=rt["k_finals"],
            alphas=rt["alphas"],
            stages=("dual", "dual", "dual"),
            ms=ms,
        )
        rows.append({
            "Setting": setting,
            "Adapter": adapter,
            "Retrieval": label,
            "ms": ",".join(map(str, ms)),
            "eval_setting": args.eval_setting,
            **dts_metrics,
        })

    eval_dir = os.path.join(rt["out_dir"], "eval_parts")
    os.makedirs(eval_dir, exist_ok=True)
    csv_path = os.path.join(eval_dir, f"{args.eval_setting}.csv")
    meta_path = os.path.join(eval_dir, f"{args.eval_setting}.meta.json")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump({
            "setting": setting,
            "adapter": adapter,
            "eval_setting": args.eval_setting,
            "rows_csv": csv_path,
            "preproc_path": preproc_path,
            "eval_queries": len(eval_qids),
            "timing_sec": {"eval_job": time.time() - start},
        }, handle, indent=2)
    print("saved:", csv_path, flush=True)
    print("saved:", meta_path, flush=True)


def summarize_one(args, rt):
    paths = artifact_paths(rt)
    wait_for_path(paths["prep_done"], args.wait_timeout_sec, args.wait_poll_sec)
    eval_dir = os.path.join(rt["out_dir"], "eval_parts")
    frames = []
    for setting in EVAL_SETTINGS:
        part = os.path.join(eval_dir, f"{setting}.csv")
        wait_for_path(part, args.wait_timeout_sec, args.wait_poll_sec)
        frames.append(pd.read_csv(part))

    long_df = pd.concat(frames, ignore_index=True)
    table_inputs = {}
    for (setting, adapter), group in long_df.groupby(["Setting", "Adapter"]):
        ivf_rows = group[group["Retrieval"] == "IVF"]
        dts_rows = group[group["Retrieval"].astype(str).str.startswith("DTS")]
        if ivf_rows.empty:
            continue
        ivf_metrics = ivf_rows.iloc[0].to_dict()
        dts_metrics_by_label = {
            row["Retrieval"]: row.to_dict()
            for _, row in dts_rows.iterrows()
        }
        table_inputs[(setting, adapter)] = (ivf_metrics, dts_metrics_by_label)

    expanded_no_ivf, expanded_no_dts = table_inputs[("Expanded DB", "none")]
    expanded_no_gap_by_label = {}
    for label, metrics in expanded_no_dts.items():
        expanded_no_gap_by_label[label] = float(metrics.get("R@100", np.nan)) - float(expanded_no_ivf.get("R@100", np.nan))

    table_rows = []
    for setting, adapter in [
        ("Base DB", "none"),
        ("Expanded DB", "none"),
        ("Expanded DB", "frozen base adapter"),
        ("Expanded DB", "refreshed adapter"),
    ]:
        ivf_metrics, dts_metrics_by_label = table_inputs[(setting, adapter)]
        dts_r100_by_label = {
            label: metrics.get("R@100")
            for label, metrics in dts_metrics_by_label.items()
        }
        table_rows.append(final_table_row(setting, adapter, ivf_metrics.get("R@100"), dts_r100_by_label, expanded_no_gap_by_label))

    final_df = pd.DataFrame(table_rows)
    long_csv = os.path.join(rt["out_dir"], "summary_long.csv")
    final_csv = os.path.join(rt["out_dir"], "final_rebuttal_table.csv")
    final_md = os.path.join(rt["out_dir"], "final_rebuttal_table.md")
    long_df.to_csv(long_csv, index=False)
    final_df.to_csv(final_csv, index=False)
    with open(final_md, "w", encoding="utf-8") as handle:
        handle.write(dataframe_to_markdown(final_df))
    print("saved:", long_csv, flush=True)
    print("saved:", final_csv, flush=True)
    print("saved:", final_md, flush=True)


def run_all(args, rt):
    run_prep(args, rt)
    for setting in EVAL_SETTINGS:
        args.eval_setting = setting
        run_eval(args, rt)
    summarize_one(args, rt)


def run_aggregate(args, rt):
    root = os.path.join(args.run_root, args.dataset, rt["encoder_tag"], "results", "db_expansion", args.run_label)
    start = time.time()
    while True:
        csv_paths = list(Path(root).glob("np*/**/final_rebuttal_table.csv"))
        if len(csv_paths) >= args.aggregate_expected_tables:
            break
        elapsed = time.time() - start
        if args.wait_timeout_sec <= 0 or elapsed > args.wait_timeout_sec:
            raise TimeoutError(
                f"Timed out waiting for {args.aggregate_expected_tables} final tables under {root}; found {len(csv_paths)}"
            )
        print(
            f"waiting for aggregate inputs under {root}: found {len(csv_paths)}/{args.aggregate_expected_tables}",
            flush=True,
        )
        time.sleep(args.wait_poll_sec)

    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        parts = csv_path.parts
        nprobe = next((p for p in parts if p.startswith("np")), "")
        run_tag = csv_path.parent.name
        df.insert(0, "run_tag", run_tag)
        df.insert(0, "nprobe", nprobe)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No final_rebuttal_table.csv files found under {root}")

    all_df = pd.concat(frames, ignore_index=True)
    metric_cols = [
        "IVF R@100",
        "DTS111 R@100",
        "DTS111-IVF Gap",
        "DTS242 R@100",
        "DTS242-IVF Gap",
    ]
    numeric = all_df.copy()
    for col in metric_cols:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    group_cols = ["nprobe", "Setting", "Adapter"]
    agg = numeric.groupby(group_cols, as_index=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [
        "_".join([str(x) for x in col if x != ""]).rstrip("_") if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    out_dir = os.path.join(root, "aggregate")
    os.makedirs(out_dir, exist_ok=True)
    all_csv = os.path.join(out_dir, "all_final_tables.csv")
    agg_csv = os.path.join(out_dir, "aggregate_mean_std.csv")
    all_df.to_csv(all_csv, index=False)
    agg.to_csv(agg_csv, index=False)
    print("saved:", all_csv, flush=True)
    print("saved:", agg_csv, flush=True)


def build_parser():
    parser = argparse.ArgumentParser("Run D-NOVA E-2 database expansion experiment.")
    parser.add_argument("--stage", choices=["all", "prep", "eval", "summarize", "aggregate"], default="all")
    parser.add_argument("--eval_setting", choices=list(EVAL_SETTINGS.keys()), default="expanded_none")
    parser.add_argument("--dataset", default="nq")
    parser.add_argument("--encoder", default="all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_label", default="e2_db_expansion")

    parser.add_argument("--data_root", default=os.getenv("DATA_ROOT", "/mnt/work/VectorDB_MICRO/datasets/semantic"))
    parser.add_argument("--intermediate_root", default=os.getenv("INTERMEDIATE_ROOT", "/mnt/work/VectorDB_MICRO/intermediate_data"))
    parser.add_argument("--run_root", default=os.getenv("RUN_ROOT", "/mnt/work/VectorDB_MICRO/DBAM/runs"))

    parser.add_argument("--holdout_doc_frac", type=float, default=0.20)
    parser.add_argument("--base_train_frac", type=float, default=0.80)
    parser.add_argument("--new_eval_frac", type=float, default=0.50)
    parser.add_argument("--passage_batch_size", type=int, default=512)
    parser.add_argument("--query_batch_size", type=int, default=128)

    parser.add_argument("--bits_sq", type=int, default=4)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--eval_nprobe", type=int, default=64)
    parser.add_argument("--k2", type=int, default=1000)
    parser.add_argument("--kfinals", default="10,25,50,100")
    parser.add_argument("--alphas", default="2,2,2")
    parser.add_argument("--ms_values", default="2,4,2;1,1,1")

    parser.add_argument("--adapter_mining_stages", default="dual,dual,dual")
    parser.add_argument("--adapter_mining_ms", default="2,4,2")
    parser.add_argument("--adapter_mining_nprobe", type=int, default=None)
    parser.add_argument("--adapter_neg_kfinal", type=int, default=10)
    parser.add_argument("--adapt_epochs", type=int, default=5)
    parser.add_argument("--adapt_lr", type=float, default=5e-4)
    parser.add_argument("--adapt_subset", type=int, default=50000)
    parser.add_argument("--adapt_qbatch", type=int, default=64)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=6.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--skip_refreshed", action="store_true")
    parser.add_argument("--wait_timeout_sec", type=int, default=43200)
    parser.add_argument("--wait_poll_sec", type=int, default=60)
    parser.add_argument("--aggregate_expected_tables", type=int, default=1)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    rt = build_runtime(args)
    if args.stage == "prep":
        run_prep(args, rt)
    elif args.stage == "eval":
        run_eval(args, rt)
    elif args.stage == "summarize":
        summarize_one(args, rt)
    elif args.stage == "aggregate":
        run_aggregate(args, rt)
    else:
        run_all(args, rt)


if __name__ == "__main__":
    main()
