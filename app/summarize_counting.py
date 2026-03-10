#!/usr/bin/env python3
import os, argparse, glob
import pandas as pd
from datetime import datetime

def main():
    p = argparse.ArgumentParser("Aggregate counting stage_stats.csv across runs.")
    p.add_argument("--run_root", default=os.getenv("RUN_ROOT", "/mnt/work/VectorDB_MICRO/DBAM/runs"))
    p.add_argument("--dataset", default=None)       # e.g. nq, hotpotqa
    p.add_argument("--encoder_tag", default=None)   # e.g. all-mpnet-base-v2
    p.add_argument("--out", default=None)
    args = p.parse_args()

    root = args.run_root
    parts = [root]
    if args.dataset:
        parts.append(args.dataset)
    else:
        parts.append("*")

    if args.encoder_tag:
        parts.append(args.encoder_tag)
    else:
        parts.append("*")

    parts += ["results", "counting", "runs", "*", "stage_stats.csv"]
    pattern = os.path.join(*parts)

    files = sorted(glob.glob(pattern))
    if not files:
        print("No files matched:", pattern)
        return

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # add path context
        # .../<dataset>/<encoder_tag>/results/counting/runs/<run_id>/stage_stats.csv
        toks = f.split(os.sep)
        # robust parse: find ".../runs/<run_id>/stage_stats.csv"
        run_id = toks[-2]
        # dataset & encoder_tag are two levels above "results"
        # .../<dataset>/<encoder_tag>/results/...
        try:
            idx = toks.index("results")
            encoder_tag = toks[idx-1]
            dataset = toks[idx-2]
        except ValueError:
            dataset = ""
            encoder_tag = ""
        df.insert(0, "dataset", dataset)
        df.insert(1, "encoder_tag", encoder_tag)
        df.insert(2, "run_id", run_id)
        df.insert(3, "stage_stats_csv", f)
        dfs.append(df)

    out_df = pd.concat(dfs, ignore_index=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.out
    if out_path is None:
        out_path = os.path.join(root, f"counting_aggregate__{ts}.csv")

    out_df.to_csv(out_path, index=False)
    print("✅ wrote:", out_path)
    print("rows:", len(out_df))

if __name__ == "__main__":
    main()
