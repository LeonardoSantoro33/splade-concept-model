#!/usr/bin/env python
"""Paired significance test between two runs already evaluated (per-query metrics in results/*.json).

Usage (from the code folder, no GPU needed):
  python 12_significance.py --a splade_baseline_dev --b hyb_e2e_k2048_..._lam0.40_dev
  (names = json file names in WORK_DIR/results, with or without .json)

For each metric: mean A, mean B, difference B-A, 95% bootstrap CI of the difference,
paired t-test p-value (normal approx., fine for thousands of queries) and a paired
randomization (sign-flip) test p-value, two-sided.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from config import Paths
from utils import append_csv_row, now_tag, save_json


def load(results_dir, name):
    # names contain dots (e.g. "lam0.75"), so do not rely on Path.suffix
    if not name.endswith(".json"):
        name = f"{name}.json"
    p = Path(name)
    if not p.is_absolute() and not p.exists():
        p = results_dir / p.name
    data = json.loads(p.read_text())
    return data["per_query"], p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline results json (e.g. splade_baseline_dev)")
    ap.add_argument("--b", required=True, help="system results json")
    ap.add_argument("--metrics", nargs="+", default=["MRR@10", "nDCG@10", "R@1000"])
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results_dir = Paths().results
    pa, fa = load(results_dir, args.a)
    pb, fb = load(results_dir, args.b)
    ia = {str(q): i for i, q in enumerate(pa["qid"])}
    ib = {str(q): i for i, q in enumerate(pb["qid"])}
    common = sorted(set(ia) & set(ib))
    print(f"A = {fa.name}  ({len(ia)} queries)")
    print(f"B = {fb.name}  ({len(ib)} queries)")
    print(f"paired queries: {len(common)}")
    if len(common) != len(ia) or len(common) != len(ib):
        print("WARNING: the two runs do not contain the same queries")

    rng = np.random.default_rng(args.seed)
    name_a, name_b = fa.name[:-5], fb.name[:-5]
    row = {"a": name_a, "b": name_b, "n_queries": len(common), "iters": args.iters, "time": now_tag()}
    per_metric = {}
    print(f"\n{'metric':>9} {'A':>8} {'B':>8} {'B-A':>8}   {'95% CI':>19}  {'p(t)':>8} {'p(perm)':>8}")
    for m in args.metrics:
        if m not in pa or m not in pb:
            print(f"{m:>9}  (missing)")
            continue
        a = np.array([pa[m][ia[q]] for q in common], dtype=np.float64)
        b = np.array([pb[m][ib[q]] for q in common], dtype=np.float64)
        d = b - a
        n = len(d)
        mean = d.mean()
        sd = d.std(ddof=1)
        if sd > 0:
            t = mean / (sd / math.sqrt(n))
            p_t = math.erfc(abs(t) / math.sqrt(2))
        else:
            p_t = 1.0
        boot = np.empty(args.iters)
        for i in range(args.iters):
            boot[i] = d[rng.integers(0, n, n)].mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        hits, done = 0, 0
        while done < args.iters:  # in chunks, to keep memory small on the login node
            c = min(500, args.iters - done)
            signs = rng.choice([-1.0, 1.0], size=(c, n))
            hits += int(np.sum(np.abs((signs * d).mean(axis=1)) >= abs(mean)))
            done += c
        p_perm = (hits + 1) / (args.iters + 1)
        print(f"{m:>9} {a.mean():8.4f} {b.mean():8.4f} {mean:+8.4f}   [{lo:+.4f}, {hi:+.4f}]  {p_t:8.4f} {p_perm:8.4f}")
        res = {"A": a.mean(), "B": b.mean(), "diff": mean, "ci95_lo": lo, "ci95_hi": hi,
               "p_ttest": p_t, "p_perm": p_perm, "significant": bool(p_perm < 0.05 and (lo > 0 or hi < 0))}
        per_metric[m] = res
        row.update({f"{m}_{k}": v for k, v in res.items()})
    print("\np < 0.05 -> the difference is statistically significant (two-sided).")

    out_json = results_dir / "significance" / f"{name_b}__vs__{name_a}.json"
    save_json({**{k: row[k] for k in ("a", "b", "n_queries", "iters", "time")}, "metrics": per_metric}, out_json)
    out_csv = results_dir / "significance.csv"
    append_csv_row(out_csv, row)
    print(f"saved: {out_json}")
    print(f"row appended to: {out_csv}")


if __name__ == "__main__":
    main()
