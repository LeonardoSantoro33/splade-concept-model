"""
01_prepare_data.py — Inspect the provided data files and build the cache in BERT vocabulary space.

What it does:
  1. lists the files in FM_DIR with their sizes
  2. reads token_to_id_mapping.json and builds the Seismic id -> BERT id (30522) conversion
  3. reads queries.bin and documents.bin (Seismic format), prints statistics and an example
  4. checks doc_ids.npy / queries_ids.npy and groundtruth_100.tsv
  5. saves the converted vectors (CSR .npy) to WORK_DIR/cache: the other scripts use these

No GPU needed. Time: ~10-20 minutes for 8.8M documents. RAM: request 64G.
    sbatch -J prep jobs/cpu.sbatch 01_prepare_data.py
For a quick test (saves nothing):
    python -u 01_prepare_data.py --max-docs 100000 --no-save
"""
import argparse

import numpy as np

from config import DEFAULT_SPLADE_MODEL, FM_FILES, Paths
from sparse_io import (build_remap, ids_to_rows, load_ids, load_token_mapping, read_groundtruth,
                       read_seismic_bin, remap_columns)
from utils import get_device, human_bytes, log, log_run_header, make_transformers_cpu_safe, save_json, timer

make_transformers_cpu_safe()
from transformers import AutoTokenizer  # noqa: E402


def show_vector(csr, i, tokenizer, n=12, title=""):
    idx, vals = csr.row(i)
    order = np.argsort(-vals)[:n]
    toks = tokenizer.convert_ids_to_tokens([int(t) for t in idx[order]])
    log(f"{title} (row id {i}, {len(idx)} non-zero entries). Highest weights:")
    for t, v in zip(toks, vals[order]):
        log(f"     {t:<18} {v:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=0, help="read only the first N documents (test)")
    ap.add_argument("--no-save", action="store_true", help="do not write the cache")
    ap.add_argument("--model", default=DEFAULT_SPLADE_MODEL, help="used only for the BERT vocabulary")
    args = ap.parse_args()
    paths = Paths()
    log_run_header("01_prepare_data.py", args, get_device("cpu"))
    if not args.no_save:
        paths.ensure_work_dir()

    # 1) files present
    log(f"Provided data directory: {paths.fm}")
    for key, name in FM_FILES.items():
        p = paths.fm / name
        log(f"  {name:<28} {'OK ' + human_bytes(p.stat().st_size) if p.exists() else 'MISSING'}")
    others = sorted(x.name for x in paths.fm.iterdir() if x.name not in FM_FILES.values())
    if others:
        log(f"  other files in the directory: {others}")

    # 2) token mapping
    tokens = load_token_mapping(paths.fm_file("mapping"))
    log(f"Tokens in the mapping: {len(tokens):,}. First 10: {tokens[:10]}")
    tok = AutoTokenizer.from_pretrained(args.model)
    vocab = tok.get_vocab()
    n_cols = len(vocab)
    remap, unknown = build_remap(tokens, vocab)
    log(f"BERT vocabulary: {n_cols:,} tokens. Tokens in the mapping NOT in the BERT vocabulary: {len(unknown)}"
        + (f" (examples: {unknown[:10]})" if unknown else ""))

    report = {"n_tokens_mapping": len(tokens), "unknown_tokens": unknown[:100], "model_vocab": args.model}

    # 3a) queries
    with timer("reading queries.bin"):
        q = read_seismic_bin(paths.fm_file("queries"))
    q_ids = load_ids(paths.fm_file("query_ids"))
    log(f"queries_ids.npy: {len(q_ids):,} ids (dtype {q_ids.dtype}), first: {q_ids[:5].tolist()}")
    if len(q_ids) != q.n_rows:
        log("WARNING: number of queries differs between queries.bin and queries_ids.npy!")
    q = remap_columns(q, remap, n_cols)
    q.ids = q_ids[: q.n_rows]
    report["queries"] = q.stats()
    log(f"Query statistics: {report['queries']}")
    show_vector(q, 0, tok, title=f"Example query qid={q.ids[0]}")

    # 3b) documents
    with timer("reading documents.bin"):
        d = read_seismic_bin(paths.fm_file("documents"), max_vectors=args.max_docs or None)
    d_ids = load_ids(paths.fm_file("doc_ids"))
    log(f"doc_ids.npy: {len(d_ids):,} ids (dtype {d_ids.dtype}), first: {d_ids[:5].tolist()}")
    d = remap_columns(d, remap, n_cols)
    d.ids = d_ids[: d.n_rows]
    if d_ids.dtype.kind in "iu":
        same = np.array_equal(d_ids[: d.n_rows], np.arange(d.n_rows))
        log(f"pid == row number? {same}")
        report["pid_equals_row"] = bool(same)
    report["documents"] = d.stats()
    log(f"Document statistics: {report['documents']}")
    show_vector(d, 0, tok, title=f"Example document pid={d.ids[0]}")
    if report["documents"]["frac_integer_values"] > 0.99:
        log("NOTE: the weights look like integers (quantized). This does not affect ranking.")

    # 4) groundtruth
    gt_path = paths.fm_file("groundtruth")
    if gt_path.exists():
        gt = read_groundtruth(gt_path, q_ids, d_ids)
        log(f"groundtruth: {len(gt):,} rows, {gt['q'].nunique():,} queries, "
            f"max rank {gt['rank'].max()}. First rows:\n{gt.head(5).to_string()}")
        report["groundtruth_rows"] = int(len(gt))
        # check: does the groundtruth score equal the dot product of our vectors?
        if not args.max_docs:
            r0 = gt.iloc[0]
            qi, qv = q.row(int(r0["q"]))
            di, dv = d.row(int(r0["d"]))
            common, ia, ib = np.intersect1d(qi, di, return_indices=True)
            dot = float((qv[ia] * dv[ib]).sum())
            log(f"dot-product check: groundtruth={r0['score']:.4f}  recomputed={dot:.4f}")
            report["gt_score_check"] = [float(r0["score"]), dot]

    # 5) cache
    if args.no_save:
        log("--no-save: not writing the cache.")
    elif args.max_docs:
        log("--max-docs used: NOT saving the document cache (it would be incomplete).")
    else:
        with timer("saving cache"):
            q.save(paths.dev_queries_csr)
            d.save(paths.docs_csr)
            save_json(report, paths.cache / "prepare_report.json")
        lookup = ids_to_rows(d.ids)
        assert lookup(d.ids[:3]).tolist() == [0, 1, 2]
        log(f"Cache saved to {paths.cache}")
    log("Done.")


if __name__ == "__main__":
    main()
