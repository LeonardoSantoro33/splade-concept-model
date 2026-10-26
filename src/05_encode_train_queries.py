"""
05_encode_train_queries.py — SPLADE vectors for the MS MARCO TRAINING queries.

The Concept Model is trained on (query, relevant passage) pairs (Sec. 6-7 of the paper):
the passages are already in the provided data (documents.bin), the training queries are not.
We encode them with the same SPLADE model selected by 03_check_encoder.py.

We keep only the queries that have at least one relevant passage in qrels.train.tsv (~503k).
Time on an A100: a few minutes.
    sbatch -J enc_train jobs/gpu.sbatch 05_encode_train_queries.py
"""
import argparse

import numpy as np

from config import Paths, get_splade_model
from msmarco import read_qrels, read_queries
from splade_encoder import SpladeEncoder, top_tokens
from utils import get_device, log, log_run_header, timer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--amp", action="store_true", help="bf16: faster, slightly different values")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    dev = get_device()
    log_run_header("05_encode_train_queries.py", args, dev)

    model_name, scale = get_splade_model(paths)
    log(f"SPLADE model: {model_name} (scale {scale:.3f})")
    texts = read_queries(paths.queries_tsv("train"))
    qrels = read_qrels(paths.qrels_tsv("train"))
    qids = np.array(sorted(q for q in qrels if q in texts), dtype=np.int64)
    log(f"training queries with text and at least one positive: {len(qids):,}")
    if args.max_queries and args.max_queries < len(qids):
        rng = np.random.default_rng(args.seed)
        qids = np.sort(rng.choice(qids, args.max_queries, replace=False))
        log(f"random sample of {len(qids):,} queries")

    enc = SpladeEncoder(model_name).to(dev)
    with timer("encoding"):
        csr = enc.encode_to_csr([texts[q] for q in qids], dev, batch_size=args.batch_size,
                                max_length=args.max_length, amp=args.amp, ids=qids, scale=scale)
    log(f"statistics: {csr.stats()}")
    log(f"example: {texts[int(qids[0])]!r} -> {top_tokens(enc.tokenizer, *csr.row(0), n=10)}")
    csr.save(paths.train_queries_csr)
    log(f"saved to {paths.train_queries_csr}")


if __name__ == "__main__":
    main()
