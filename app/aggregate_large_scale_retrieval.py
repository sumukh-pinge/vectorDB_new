#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
from pathlib import Path


def encoder_tag(encoder: str) -> str:
    return encoder.replace("/", "_")


def write_csv(path: Path, rows, fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--nlist", type=int, required=True)
    parser.add_argument("--nprobes", required=True)
    parser.add_argument("--modes", default="ivf,dts111,dts242")
    parser.add_argument("--adapters", default="off,on")
    parser.add_argument("--run_root", default=os.getenv("RUN_ROOT", "/mnt/work/VectorDB_MICRO/large_scale_runs"))
    parser.add_argument("--wait_timeout_sec", type=int, default=604800)
    parser.add_argument("--wait_poll_sec", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    tag = encoder_tag(args.encoder)
    out_dir = Path(args.run_root) / args.dataset / tag / f"nlist{args.nlist}" / "aggregate"
    out_json = out_dir / "aggregate.json"
    if out_json.exists() and not args.force:
        print(f"[skip] aggregate exists: {out_json}", flush=True)
        return

    nprobes = [int(x) for x in args.nprobes.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    adapters = [x.strip() for x in args.adapters.split(",") if x.strip()]
    expected = [
        Path(args.run_root) / args.dataset / tag / f"nlist{args.nlist}" / f"np{nprobe}" / mode / f"adapter_{adapter}" / "summary.json"
        for nprobe in nprobes
        for mode in modes
        for adapter in adapters
    ]
    deadline = time.time() + args.wait_timeout_sec
    while True:
        missing = [str(path) for path in expected if not path.exists()]
        if not missing:
            break
        if time.time() > deadline:
            raise TimeoutError(f"timed out waiting for {len(missing)} summaries; first missing={missing[:3]}")
        print(f"[wait] missing={len(missing)} first={missing[0]}", flush=True)
        time.sleep(args.wait_poll_sec)

    rows = []
    for nprobe in nprobes:
        for adapter in adapters:
            by_mode = {}
            for mode in modes:
                summary_path = Path(args.run_root) / args.dataset / tag / f"nlist{args.nlist}" / f"np{nprobe}" / mode / f"adapter_{adapter}" / "summary.json"
                row = json.loads(summary_path.read_text())
                row["adapter"] = adapter
                by_mode[mode] = row
                rows.append(row)
            if "ivf" in by_mode:
                ivf = by_mode["ivf"]
                for mode in modes:
                    if mode == "ivf" or mode not in by_mode:
                        continue
                    by_mode[mode]["R@100-IVF_gap"] = by_mode[mode].get("R@100", 0.0) - ivf.get("R@100", 0.0)
                    by_mode[mode]["R@25-IVF_gap"] = by_mode[mode].get("R@25", 0.0) - ivf.get("R@25", 0.0)

    rows = sorted(rows, key=lambda r: (str(r.get("adapter", "")), int(r.get("nprobe", 0)), str(r.get("mode", ""))))
    metric_fields = [
        "dataset", "encoder_tag", "adapter", "mode", "nlist", "nprobe", "k2", "kfinal", "queries",
        "H@25", "R@25", "MRR@25", "H@100", "R@100", "MRR@100",
        "R@25-IVF_gap", "R@100-IVF_gap",
        "raw_candidates_avg", "raw_candidates_p50", "raw_candidates_p95", "raw_candidates_max",
        "stage2_candidates_avg", "final_candidates_avg",
    ]
    write_csv(out_dir / "aggregate.csv", rows, metric_fields)
    write_csv(out_dir / "aggregate_all_fields.csv", rows, sorted({k for row in rows for k in row.keys()}))
    obj = {
        "dataset": args.dataset,
        "encoder": args.encoder,
        "encoder_tag": tag,
        "nlist": args.nlist,
        "nprobes": nprobes,
        "modes": modes,
        "adapters": adapters,
        "rows": rows,
        "missing": missing,
        "timestamp": time.time(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(obj, indent=2, sort_keys=True))
    print(f"[done] aggregate rows={len(rows)} missing={len(missing)} out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
