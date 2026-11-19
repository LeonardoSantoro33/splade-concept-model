"""
13_eval_nq.py — Zero-shot evaluation OUT OF DOMAIN on NQ (BEIR) with the SPLADE vectors given by the professor.

Dataset: tuskanny/sparse-nq-splade-cocondenser (2,680,893 passages, 3,452 test queries), already encoded with
naver/splade-cocondenser-ensembledistil, Seismic binary format. There is NO text, so nothing is encoded here:

  - SPLADE baseline : exact sparse search on the professor's vectors
  - frozen / rand   : g(x) = W . SPLADE(x) applied to the professor's vectors (same exact streaming search as
                      08b: score = d . (W g(q)), the dense documents are never built)
  - e2e checkpoints : NOT possible (they need the text of the passages to re-encode them): skipped.

Expected folder (WORK_DIR/beir/nq by default):
    documents.bin  queries.bin  doc_ids.npy  queries_ids.npy  groundtruth.tsv  qrels/test.tsv

The BEIR ids (strings: 'doc123', 'test45') are mapped to integers = ROW number in the .bin files.
Results go to results/summary.csv with query_set = beir_nq (same format as 10_eval_beir.py).
The parsed vectors are cached in <dir>/csr_docs and <dir>/csr_queries (the first run reads the 3.3 GB .bin).

    sbatch --time=02:00:00 -J nq jobs/gpu.sbatch 13_eval_nq.py --splade \
        --ckpts frozen_k1024_rand frozen_k1024_r2_lr3e5
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from concept_model import load_checkpoint
from config import Paths, resolve_ckpt
from pipeline import concept_vectors_from_csr, free_cuda, report_and_save
from retrieval import TorchCSR, search_with_query_table, sparse_search
from sparse_io import CSR, read_seismic_bin
from utils import get_device, log, log_run_header, make_transformers_cpu_safe, timer

make_transformers_cpu_safe()

VOCAB = 30522          # BERT vocabulary: the professor's ids are BERT ids (checked by decoding the tokens)


# ----------------------------------------------------------------------
#  data
# ----------------------------------------------------------------------
def load_vectors(nq_dir, name):
    """documents.bin / queries.bin -> CSR in the BERT space (cached on disk)."""
    cache = nq_dir / f"csr_{name}"
    if (cache / "meta.json").exists():
        csr = CSR.load(cache)
        log(f"[{name}] loaded from cache {cache}")
    else:
        with timer(f"reading {name}.bin"):
            csr = read_seismic_bin(nq_dir / f"{name}.bin")
        if csr.nnz and int(csr.indices.max()) >= VOCAB:
            raise SystemExit(f"[ERROR] {name}.bin has component ids up to {int(csr.indices.max())} (>= {VOCAB}): "
                             f"not the BERT vocabulary, W cannot be applied")
        csr.n_cols = VOCAB
        csr.ids = np.arange(csr.n_rows, dtype=np.int64)
        csr.save(cache)
    log(f"[{name}] {csr.n_rows:,} vectors, mean {csr.nnz / max(1, csr.n_rows):.1f} non-zero entries")
    return csr


def load_qrels(nq_dir, n_docs, n_queries):
    """qrels/test.tsv (BEIR strings) -> {query_row: {doc_row: rel}}, rows = positions in the .bin files."""
    q_ids = np.load(nq_dir / "queries_ids.npy", allow_pickle=True)
    d_ids = np.load(nq_dir / "doc_ids.npy", mmap_mode="r")          # <U259, 2.7 GB: read in pieces
    if len(q_ids) != n_queries or d_ids.shape[0] != n_docs:
        raise SystemExit(f"[ERROR] ids and vectors do not match: queries {len(q_ids)} vs {n_queries}, "
                         f"documents {d_ids.shape[0]} vs {n_docs}")
    judged = {}
    with open(nq_dir / "qrels" / "test.tsv", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row or row[0] == "query-id":
                continue
            judged.setdefault(row[0], {})[row[1]] = int(float(row[2]))
    need = np.array(sorted({d for j in judged.values() for d in j}))
    d2r = {}
    for s in range(0, n_docs, 200_000):
        chunk = np.asarray(d_ids[s:s + 200_000])
        for j in np.nonzero(np.isin(chunk, need))[0].tolist():
            d2r[str(chunk[j])] = s + j
    q2r = {str(q): i for i, q in enumerate(q_ids.tolist())}
    qrels, n_unknown = {}, 0
    for q, docs in judged.items():
        if q not in q2r:
            continue
        for d, r in docs.items():
            if d not in d2r:
                n_unknown += 1
                continue
            qrels.setdefault(q2r[q], {})[d2r[d]] = r
    if n_unknown:
        log(f"WARNING: {n_unknown} qrels point to documents not in doc_ids.npy (ignored)")
    log(f"qrels: {len(qrels):,} judged queries (of {n_queries:,}), "
        f"{sum(len(v) for v in qrels.values()):,} judgments")
    return qrels


def check_groundtruth(nq_dir, rows, scores):
    """Our exact SPLADE search must reproduce the professor's exact top-10 (same vectors, same dot product)."""
    import pandas as pd
    df = pd.read_csv(nq_dir / "groundtruth.tsv", sep="\t", header=None, names=["q", "d", "rank", "score"])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df = df[df["rank"] <= 10]
    overlap, diffs = [], []
    for q, g in df.groupby("q"):
        q = int(q)
        gt = set(g["d"].astype(np.int64).tolist())
        overlap.append(len(gt & set(rows[q, :10].tolist())) / max(1, len(gt)))
        best = g.sort_values("rank")["score"].to_numpy()
        diffs.append(float(np.max(np.abs(best - scores[q, :len(best)]))))
    log(f"[check] SPLADE vs groundtruth.tsv: mean top-10 overlap {np.mean(overlap):.4f} "
        f"(queries with full overlap: {np.mean(np.array(overlap) == 1.0):.3f}), "
        f"max score difference {np.max(diffs):.2e}")
    if np.mean(overlap) < 0.99:
        log("WARNING: the exact search does NOT reproduce the professor's ground truth: check the data before "
            "trusting any number below")


# ----------------------------------------------------------------------
#  main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="folder with the NQ files (default WORK_DIR/beir/nq)")
    ap.add_argument("--ckpts", nargs="*", default=[], help="frozen / rand run names to evaluate")
    ap.add_argument("--splade", action="store_true", help="also evaluate the SPLADE baseline")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--q-batch", type=int, default=256)
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    nq_dir = Path(args.dir) if args.dir else paths.work / "beir" / "nq"
    dev = get_device()
    log_run_header("13_eval_nq.py", args, dev)
    set_name = "beir_nq"

    docs = load_vectors(nq_dir, "documents")
    q = load_vectors(nq_dir, "queries")
    qrels = load_qrels(nq_dir, docs.n_rows, q.n_rows)
    doc_rows = np.arange(docs.n_rows, dtype=np.int64)       # row -> integer doc id (= the row itself)
    qids = np.arange(q.n_rows, dtype=np.int64)
    k = min(args.k, docs.n_rows)
    extra0 = {"dataset": "nq", "n_docs": docs.n_rows}
    tdocs = TorchCSR(docs, dev)

    if args.splade:
        with timer("exact SPLADE search"):
            scores, rows = sparse_search(tdocs, q, k, verbose=False)
        check_groundtruth(nq_dir, rows, scores)
        report_and_save(paths, "splade_baseline_beir", set_name, qids, rows, scores, doc_rows, qrels, 1,
                        extra=extra0)

    for run in args.ckpts:
        ckpt = resolve_ckpt(run, paths)
        head, info = load_checkpoint(ckpt, dev)
        head.eval()
        if info["mode"] == "e2e":
            log(f"WARNING: {run} is e2e: it needs the passage text (not available here): skipped")
            continue
        name = f"cm_{ckpt.parent.name}_{ckpt.name}"
        W = head.weight.detach().float()
        gq = concept_vectors_from_csr(head, q, dev)
        nq = gq.shape[0]
        out_s = np.empty((nq, k), dtype=np.float32)
        out_i = np.empty((nq, k), dtype=np.int64)
        with timer(f"{name}: streaming search"):
            for qs in range(0, nq, args.q_batch):
                qe = min(nq, qs + args.q_batch)
                table = (W @ gq[qs:qe].T).contiguous()        # d . (W g(q)) = g(d) . g(q)
                s, i = search_with_query_table(tdocs, table, k)
                out_s[qs:qe] = s.float().cpu().numpy()
                out_i[qs:qe] = i.cpu().numpy()
        report_and_save(paths, name, set_name, qids, out_i, out_s, doc_rows, qrels, 1,
                        extra={**extra0, "k_concepts": head.k, "mode": info["mode"], "ckpt": str(ckpt),
                               "search": "streaming"})
        del W, gq, out_s, out_i
        free_cuda()
    log("Done.")


if __name__ == "__main__":
    main()
