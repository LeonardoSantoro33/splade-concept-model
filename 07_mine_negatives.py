"""
07_mine_negatives.py — Hard-negative mining (Sec. 6.3 of the paper, ANCE procedure).

  (a) the model weights are frozen during mining
  (b) all corpus documents are (re-)encoded with these weights
  (c) search index -> here EXACT search on GPU (no approximation)
  (d) each training query retrieves its `--depth` nearest documents
  (e) those NOT marked as relevant in the qrels become hard negatives

The "miner" can be:
  --miner splade                 : the base SPLADE (the natural Retriever_1 of Sec. 6.4)
  --miner concept --ckpt NAME    : an already trained Concept Model (frozen or e2e) -> later rounds

Output: WORK_DIR/negatives/<out>.npz with
  qids      (nq,)         training query ids
  neg_pids  (nq, keep)    negative pids in rank order (-1 = empty)

False negatives (Sec. 6.3): relevant but never judged passages end up among the negatives.
Simple remedy: in training use --neg-rank-min 3 (discard the top candidates, the most suspicious).

    sbatch -J mine_splade jobs/gpu.sbatch 07_mine_negatives.py --miner splade --out splade
    sbatch -J mine_r1 jobs/gpu.sbatch 07_mine_negatives.py --miner concept --ckpt frozen_k256_r1 --out r1
"""
import argparse

import numpy as np
import torch

from concept_model import load_checkpoint
from config import Paths, resolve_ckpt
from msmarco import Collection, positives_from_qrels, read_qrels, read_queries
from pipeline import (concept_vectors_from_csr, concept_vectors_from_texts, docs_to_device, encode_corpus_e2e,
                      free_cuda, load_docs)
from retrieval import dense_search, project_rows, sparse_search
from sparse_io import CSR
from utils import get_device, log, log_run_header, save_json, timer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miner", choices=["splade", "concept"], required=True)
    ap.add_argument("--ckpt", default=None, help="checkpoint name or path (concept miner)")
    ap.add_argument("--out", required=True, help="output file name (without .npz)")
    ap.add_argument("--depth", type=int, default=200, help="number of documents to retrieve per query")
    ap.add_argument("--keep", type=int, default=100, help="number of negatives to save per query")
    ap.add_argument("--max-queries", type=int, default=0, help="0 = all training queries")
    ap.add_argument("--q-batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-max-len", type=int, default=256)
    ap.add_argument("--q-max-len", type=int, default=64)
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    dev = get_device()
    log_run_header("07_mine_negatives.py", args, dev)

    train_q = CSR.load(paths.train_queries_csr)
    q_ids = np.asarray(train_q.ids, dtype=np.int64)
    positives = positives_from_qrels(read_qrels(paths.qrels_tsv("train")))
    rows = np.array([i for i, q in enumerate(q_ids.tolist()) if q in positives], dtype=np.int64)
    if args.max_queries and args.max_queries < len(rows):
        rows = np.sort(np.random.default_rng(args.seed).choice(rows, args.max_queries, replace=False))
    log(f"queries to process: {len(rows):,}")

    docs = load_docs(paths)
    doc_pids = np.asarray(docs.ids, dtype=np.int64)

    if args.miner == "splade":
        tdocs = docs_to_device(docs, dev)
        with timer("exact SPLADE search"):
            _, top = sparse_search(tdocs, train_q, args.depth, q_batch=args.q_batch, query_rows=rows)
        meta = {"miner": "splade"}
    else:
        if not args.ckpt:
            raise SystemExit("[ERROR] --miner concept requires --ckpt")
        ckpt = resolve_ckpt(args.ckpt, paths)
        head, info = load_checkpoint(ckpt, dev)
        head.eval()
        meta = {"miner": "concept", "ckpt": str(ckpt), "mode": info["mode"], "step": info["step"]}
        if info["mode"] == "frozen":
            tdocs = docs_to_device(docs, dev)
            with timer("g(d) for all documents"):
                D = project_rows(tdocs, head.weight.detach())
            tdocs.free()
            del tdocs
            gq = concept_vectors_from_csr(head, train_q, dev, rows)
        else:
            from splade_encoder import SpladeEncoder
            encoder = SpladeEncoder(info["backbone_path"]).to(dev)
            collection = Collection(paths.collection_tsv)
            D = encode_corpus_e2e(head, encoder, collection, doc_pids,
                                  paths.embeddings / f"{ckpt.parent.name}_{ckpt.name}", dev,
                                  max_length=args.d_max_len, tag=str(ckpt)).to(dev)
            qtexts = read_queries(paths.queries_tsv("train"))
            gq = concept_vectors_from_texts(head, encoder, [qtexts[int(q)] for q in q_ids[rows]], dev,
                                            args.q_max_len)
        free_cuda()
        with timer("exact dense search"):
            _, top = dense_search(D, gq, args.depth, q_batch=1024)
        del D

    # remove the positives, keep the first `keep` in rank order
    neg = np.full((len(rows), args.keep), -1, dtype=np.int32)
    found_pos = 0
    for j, r in enumerate(rows):
        pos = set(positives[int(q_ids[r])])
        cand = doc_pids[top[j]]
        hit = np.fromiter((p in pos for p in cand.tolist()), dtype=bool, count=len(cand))
        found_pos += bool(hit.any())
        keep = cand[~hit][: args.keep]
        neg[j, :len(keep)] = keep
    recall = found_pos / max(1, len(rows))
    log(f"the miner finds a positive in the top {args.depth} for {100 * recall:.1f}% of the queries")
    log(f"negatives per query (mean): {(neg >= 0).sum(1).mean():.1f}")

    out = paths.negatives / f"{args.out}.npz"
    np.savez_compressed(out, qids=q_ids[rows], neg_pids=neg)
    save_json({**meta, "depth": args.depth, "keep": args.keep, "n_queries": int(len(rows)),
               "miner_recall_at_depth": recall}, paths.negatives / f"{args.out}.json")
    log(f"saved {out}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
