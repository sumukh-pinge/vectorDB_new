#!/usr/bin/env python3
"""
Run the 6-point NRP smoke/baseline suite in sequence:

  1) IVF FP32 w/o adapter                -> ivf_fp32_baseline.csv
  2) D-BAM (DIRECT,DIRECT,DIRECT)        -> dbam_direct_baseline_mX-Y-Z.csv
  3) D2-BAM (DUAL,DUAL,DUAL)             -> dbam_dual_baseline_mX-Y-Z.csv
  4) IVF FP32 w/ adapter                 -> ivf_fp32_with_adapter.csv
  5) D-BAM (DIRECT,DIRECT,DIRECT) + W    -> dbam_direct_with_adapter_mX-Y-Z.csv
  6) D2-BAM (DUAL,DUAL,DUAL) + W         -> dbam_dual_with_adapter_mX-Y-Z.csv

All CSVs are written under:
  {run_root}/results/{results_tag or auto: np{nprobe}_{YYYYmmdd-HHMMSS}}/

Adapter: If missing for the chosen slug, nq_cli.py will auto-train it.
"""

import argparse, os, shlex, subprocess, sys
from pathlib import Path
from datetime import datetime

def run_cmd(cmd: str, env=None, tee=None):
    print(f"\n$ {cmd}\n", flush=True)
    p = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env or os.environ.copy()
    )
    for line in p.stdout:
        sys.stdout.write(line)
        if tee: 
          tee.write(line)
          tee.flush()
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd}")

def build_base_args(a):
    base = (
        f"--dataset {a.dataset} "
        f"--work_dir {a.work_dir} "
        f"--data_root {a.data_root} "
        f"--run_root {a.run_root} "
        f"--nlist {a.nlist} "
        f"--select_nprobe {a.select_nprobe} "
        f"--k2_fixed {a.k2_fixed} "
        f"--bits_sq {a.bits_sq} "
        f"--ms_infer {a.ms_infer} "
        f"--alphas {a.alphas}"
    )
    return base

def main():
    ap = argparse.ArgumentParser()
    # paths
    ap.add_argument("--dataset", default="beir_nq")
    ap.add_argument("--work_dir", default="/home/lemur")
    ap.add_argument("--data_root", default="/project/sumukh/vectorDB/datasets")
    ap.add_argument("--run_root", default="/home/lemur/VectorDB/DBAM/NRP_repos/nrp_01/runs/nq_local")

    # pipeline knobs
    ap.add_argument("--bits_sq", type=int, default=4)
    ap.add_argument("--nlist", type=int, default=2048)
    ap.add_argument("--select_nprobe", type=int, default=16)
    ap.add_argument("--k2_fixed", type=int, default=1000)
    ap.add_argument("--ms_infer", default="1,8,1")
    ap.add_argument("--alphas", default="2,2,2")

    # adapter knobs passed through to nq_cli (used if adapter needs training)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=6.0)
    ap.add_argument("--cands", type=int, default=2048)
    ap.add_argument("--teacher", default="cos")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--subset", type=int, default=50000)
    ap.add_argument("--q_batch", type=int, default=64)
    ap.add_argument("--slug", default="auto", help="Adapter slug (auto derives from hparams)")

    # results naming
    ap.add_argument("--results_tag", default="", help="If set, results subdir name; else auto np{nprobe}_<timestamp>.")

    args = ap.parse_args()

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    # Results subdir
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    auto_tag = f"np{args.select_nprobe}_{ts}"
    tag = args.results_tag.strip() or auto_tag
    results_dir = run_root / "results" / tag
    results_dir.mkdir(parents=True, exist_ok=True)

    # Log file
    log_path = results_dir / f"full_suite_{tag}.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"       
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")

    ms_tag = args.ms_infer.replace(",", "-")
    base = build_base_args(args)

    # Common suffix for adapter-run calls
    adapter_suffix = (
        f" --slug {args.slug} --tau {args.tau} --beta {args.beta} --cands {args.cands} "
        f"--teacher {args.teacher} --lr {args.lr} --epochs {args.epochs} "
        f"--subset {args.subset} --q_batch {args.q_batch}"
    )

    with open(log_path, "w") as tee:
        print(f"Logging to {log_path}")

        jobs = [
            # (mode, out_filename, extra_flags)
            # ("ivf_fp32",                "ivf_fp32_baseline.csv",                         ""),
            # ("dbam_direct_baseline",    f"dbam_direct_baseline_m{ms_tag}.csv",          ""),
            ("dbam_dual_baseline",      f"dbam_dual_baseline_m{ms_tag}.csv",            ""),
            # ("ivf_fp32_adapter",        "ivf_fp32_with_adapter.csv",                    adapter_suffix),
            # ("dbam_direct_adapter",     f"dbam_direct_with_adapter_m{ms_tag}.csv",      adapter_suffix),
            # ("dbam_dual_adapter",       f"dbam_dual_with_adapter_m{ms_tag}.csv",        adapter_suffix),
        ]

        for mode, fname, extra in jobs:
            out_csv = results_dir / fname
            if out_csv.exists():
                print(f"[skip] {out_csv} exists", flush=True)
                continue

            cmd = f"python -u nq_cli.py {base} --mode {mode} --out_csv {out_csv}"
            if extra:
                cmd += f" {extra}"

            run_cmd(cmd, env=env, tee=tee)

        print("\n✅ All six tests complete.")
        print(f"📂 Results in: {results_dir}")

if __name__ == "__main__":
    main()
