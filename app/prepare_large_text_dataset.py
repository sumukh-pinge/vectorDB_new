#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def ensure_package(import_name, pip_name=None):
    try:
        __import__(import_name)
    except ImportError:
        pkg = pip_name or import_name
        print(f"[deps] installing missing package: {pkg}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)


def atomic_path(path):
    return path.with_suffix(path.suffix + ".tmp")


def download_bytes(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def read_tsv_url(url):
    text = download_bytes(url).decode("utf-8")
    rows = []
    for line in text.splitlines():
        if line.strip():
            rows.append(line.rstrip("\n").split("\t"))
    return rows


def write_beir_qrels(path, qrels_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = atomic_path(path)
    total = 0
    positives = 0
    query_ids = set()
    positive_query_ids = set()
    with open(tmp, "w", encoding="utf-8") as out:
        out.write("query-id\tcorpus-id\tscore\n")
        for row in qrels_rows:
            if len(row) < 4:
                continue
            qid, _, docid, score = row[:4]
            try:
                rel = int(score)
            except ValueError:
                continue
            total += 1
            query_ids.add(qid)
            if rel > 0:
                positives += 1
                positive_query_ids.add(qid)
                out.write(f"{qid}\t{docid}\t{rel}\n")
    os.replace(tmp, path)
    return {
        "input_judgments": total,
        "positive_qrels": positives,
        "queries_in_qrels": len(query_ids),
        "positive_queries": len(positive_query_ids),
        "avg_positives_per_positive_query": positives / max(len(positive_query_ids), 1),
    }


def write_beir_queries_from_tsv(path, topic_rows_by_split, keep_positive_qids):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = atomic_path(path)
    seen = set()
    written = 0
    with open(tmp, "w", encoding="utf-8") as out:
        for split, rows in topic_rows_by_split.items():
            for row in rows:
                if len(row) < 2:
                    continue
                qid = row[0]
                if qid in seen or qid not in keep_positive_qids:
                    continue
                text = row[1]
                out.write(json.dumps({"_id": qid, "text": text, "metadata": {"split": split}}, ensure_ascii=False) + "\n")
                seen.add(qid)
                written += 1
    os.replace(tmp, path)
    return written


def list_hf_dataset_files(repo):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=1"
    with urllib.request.urlopen(url, timeout=120) as response:
        data = json.load(response)
    return [item["path"] for item in data if item.get("type") == "file"]


def prepare_miracl_en(args, out_dir):
    corpus_file = out_dir / "corpus.jsonl"
    queries_file = out_dir / "queries.jsonl"
    qrels_dir = out_dir / "qrels"
    ready = out_dir / "DATASET_READY.json"
    if ready.exists() and not args.force:
        print(f"[skip] {ready} exists", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir.mkdir(parents=True, exist_ok=True)

    qrels_stats = {}
    keep_positive_qids = set()
    topic_rows = {}
    for split in ["train", "dev"]:
        qrels_url = HF_RESOLVE.format(
            repo="miracl/miracl",
            path=f"miracl-v1.0-en/qrels/qrels.miracl-v1.0-en-{split}.tsv",
        )
        topics_url = HF_RESOLVE.format(
            repo="miracl/miracl",
            path=f"miracl-v1.0-en/topics/topics.miracl-v1.0-en-{split}.tsv",
        )
        qrels_rows = read_tsv_url(qrels_url)
        qrels_stats[split] = write_beir_qrels(qrels_dir / f"{split}.tsv", qrels_rows)
        for row in qrels_rows:
            if len(row) >= 4:
                try:
                    if int(row[3]) > 0:
                        keep_positive_qids.add(row[0])
                except ValueError:
                    pass
        topic_rows[split] = read_tsv_url(topics_url)

    queries_written = write_beir_queries_from_tsv(queries_file, topic_rows, keep_positive_qids)

    files = sorted(
        path for path in list_hf_dataset_files("miracl/miracl-corpus")
        if path.startswith("miracl-corpus-v1.0-en/docs-") and path.endswith(".jsonl.gz")
    )
    if not files:
        raise RuntimeError("Could not discover MIRACL-English corpus shards")

    if corpus_file.exists() and not args.force:
        print(f"[skip] existing corpus: {corpus_file}", flush=True)
        docs = sum(1 for _ in open(corpus_file, "r", encoding="utf-8"))
    else:
        tmp = atomic_path(corpus_file)
        docs = 0
        with open(tmp, "w", encoding="utf-8") as out:
            for index, path in enumerate(files, 1):
                url = HF_RESOLVE.format(repo="miracl/miracl-corpus", path=path)
                print(f"[miracl] shard {index}/{len(files)} {path}", flush=True)
                with urllib.request.urlopen(url, timeout=300) as response:
                    with gzip.GzipFile(fileobj=response) as gz:
                        for raw in gz:
                            obj = json.loads(raw.decode("utf-8"))
                            out.write(json.dumps({
                                "_id": str(obj["docid"]),
                                "title": obj.get("title", ""),
                                "text": obj.get("text", ""),
                                "metadata": {"source": "miracl-en"},
                            }, ensure_ascii=False) + "\n")
                            docs += 1
                print(f"[miracl] docs_written={docs}", flush=True)
        os.replace(tmp, corpus_file)

    stats = {
        "dataset": "miracl-en",
        "format": "beir-jsonl",
        "corpus_file": str(corpus_file),
        "queries_file": str(queries_file),
        "qrels_dir": str(qrels_dir),
        "documents": docs,
        "queries_written": queries_written,
        "qrels": qrels_stats,
        "timestamp": time.time(),
    }
    write_json(out_dir / "dataset_stats.json", stats)
    write_json(ready, stats)
    print(json.dumps(stats, indent=2), flush=True)


def prepare_dpr_w100(args, out_dir):
    ensure_package("ir_datasets")
    import ir_datasets

    corpus_file = out_dir / "corpus.jsonl"
    queries_file = out_dir / "queries.jsonl"
    qrels_dir = out_dir / "qrels"
    ready = out_dir / "DATASET_READY.json"
    if ready.exists() and not args.force:
        print(f"[skip] {ready} exists", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir.mkdir(parents=True, exist_ok=True)

    split_names = ["train", "dev"]
    datasets = {split: ir_datasets.load(f"dpr-w100/natural-questions/{split}") for split in split_names}

    qrels_stats = {}
    keep_positive_qids = set()
    for split, dataset in datasets.items():
        path = qrels_dir / f"{split}.tsv"
        tmp = atomic_path(path)
        total = 0
        positives = 0
        qids = set()
        positive_qids = set()
        with open(tmp, "w", encoding="utf-8") as out:
            out.write("query-id\tcorpus-id\tscore\n")
            for qrel in dataset.qrels_iter():
                total += 1
                qid = str(qrel.query_id)
                qids.add(qid)
                if int(qrel.relevance) > 0:
                    positives += 1
                    positive_qids.add(qid)
                    keep_positive_qids.add(qid)
                    out.write(f"{qid}\t{qrel.doc_id}\t{int(qrel.relevance)}\n")
        os.replace(tmp, path)
        qrels_stats[split] = {
            "input_judgments": total,
            "positive_qrels": positives,
            "queries_in_qrels": len(qids),
            "positive_queries": len(positive_qids),
            "avg_positives_per_positive_query": positives / max(len(positive_qids), 1),
        }

    tmp_queries = atomic_path(queries_file)
    seen = set()
    queries_written = 0
    with open(tmp_queries, "w", encoding="utf-8") as out:
        for split, dataset in datasets.items():
            for query in dataset.queries_iter():
                qid = str(query.query_id)
                if qid in seen or qid not in keep_positive_qids:
                    continue
                out.write(json.dumps({"_id": qid, "text": query.text, "metadata": {"split": split}}, ensure_ascii=False) + "\n")
                seen.add(qid)
                queries_written += 1
    os.replace(tmp_queries, queries_file)

    if corpus_file.exists() and not args.force:
        print(f"[skip] existing corpus: {corpus_file}", flush=True)
        docs = sum(1 for _ in open(corpus_file, "r", encoding="utf-8"))
    else:
        tmp_corpus = atomic_path(corpus_file)
        docs = 0
        with open(tmp_corpus, "w", encoding="utf-8") as out:
            for doc in datasets["dev"].docs_iter():
                out.write(json.dumps({
                    "_id": str(doc.doc_id),
                    "title": getattr(doc, "title", ""),
                    "text": doc.text,
                    "metadata": {"source": "dpr-w100-natural-questions"},
                }, ensure_ascii=False) + "\n")
                docs += 1
                if docs % 250000 == 0:
                    print(f"[dpr-w100] docs_written={docs}", flush=True)
        os.replace(tmp_corpus, corpus_file)

    stats = {
        "dataset": "dpr-w100",
        "format": "beir-jsonl",
        "corpus_file": str(corpus_file),
        "queries_file": str(queries_file),
        "qrels_dir": str(qrels_dir),
        "documents": docs,
        "queries_written": queries_written,
        "qrels": qrels_stats,
        "timestamp": time.time(),
    }
    write_json(out_dir / "dataset_stats.json", stats)
    write_json(ready, stats)
    print(json.dumps(stats, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["miracl-en", "dpr-w100"], required=True)
    parser.add_argument("--data_root", default=os.getenv("DATA_ROOT", "/mnt/work/VectorDB_MICRO/datasets/semantic"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.data_root) / args.dataset / args.dataset
    if args.dataset == "miracl-en":
        prepare_miracl_en(args, out_dir)
    elif args.dataset == "dpr-w100":
        prepare_dpr_w100(args, out_dir)


if __name__ == "__main__":
    main()
