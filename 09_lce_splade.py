"""
09_lce_splade.py — (OPTIONAL) LCE applied to SPLADE: "static" expansion with the
co-occurrence matrix M (Sec. 3), in the same space the Concept Model operates in.

Why it is useful even without SPLARE:
  - it is the cleanest comparison for the claim of Sec. 4: COUNTED expansion (LCE) vs.
    LEARNED expansion (Concept Model), with the same underlying SPLADE vectors;
  - once SPLARE is available, the same code (lce.py) runs on the SAE vectors.

Steps: builds M over the documents (Eq. 1, saved in WORK_DIR/lce), then tries a grid of
(normalization, alpha, pruning) on the dev queries and saves everything to results/lce_grid_<set>.csv.

    sbatch -J lce jobs/gpu.sbatch 09_lce_splade.py --demo "I have a crushing pressure in my chest and I am sweaty"
"""
import argparse
import itertools

import numpy as np
import torch

from config import Paths, get_splade_model
from lce import build_cooccurrence, expand, normalize_cooccurrence, prune
from pipeline import docs_to_device, load_docs, report_and_save, splade_query_vectors
from retrieval import csr_rows_to_dense, search_with_query_table
from utils import append_csv_row, get_device, log, log_run_header, make_transformers_cpu_safe, save_json, timer

make_transformers_cpu_safe()


def show_expansion(tokenizer, Qd_row, Vext_row, n=12):
    q = Qd_row.cpu()
    v = Vext_row.cpu()
    orig = torch.topk(q, min(n, int((q > 0).sum()))).indices.tolist()
    added = torch.where(q > 0, torch.zeros_like(v), v)
    new = torch.topk(added, min(n, int((added > 0).sum()))).indices.tolist() if (added > 0).any() else []
    o = ", ".join(f"{t} {q[i]:.2f}" for t, i in zip(tokenizer.convert_ids_to_tokens(orig), orig))
    a = ", ".join(f"{t} {v[i]:.2f}" for t, i in zip(tokenizer.convert_ids_to_tokens(new), new))
    return o, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="recompute M even if already saved")
    ap.add_argument("--max-docs", type=int, default=0, help="build M on a random sample of documents (0 = all)")
    ap.add_argument("--norms", nargs="+", default=["raw", "cond", "cosine", "ppmi"])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    ap.add_argument("--alpha-mode", choices=["relative", "absolute"], default="relative")
    ap.add_argument("--prunes", nargs="+", default=["topk:64", "tau:0.3", "none"])
    ap.add_argument("--keep-original", action="store_true", help="never prune the original query terms")
    ap.add_argument("--min-count", type=int, default=1, help="ignore co-occurrences rarer than this")
    ap.add_argument("--query-sets", nargs="+", default=["dev"], choices=["dev", "dl19", "dl20"])
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--q-batch", type=int, default=256)
    ap.add_argument("--demo", default=None, help="example sentence to expand")
    ap.add_argument("--examples", type=int, default=3, help="number of expanded dev queries to show")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    dev = get_device()
    log_run_header("09_lce_splade.py", args, dev)

    docs = load_docs(paths)
    doc_pids = np.asarray(docs.ids, dtype=np.int64)
    V = docs.n_cols

    # ------------------------------------------------------------ Eq. (1): co-occurrence matrix
    tag = f"n{args.max_docs}" if args.max_docs else "all"
    m_path, df_path = paths.lce / f"M_counts_{tag}.npy", paths.lce / f"df_{tag}.npy"
    if m_path.exists() and not args.rebuild:
        with timer("loading M from cache"):
            M = torch.from_numpy(np.load(m_path)).to(dev)
            df = torch.from_numpy(np.load(df_path)).to(dev)
    else:
        rows = None
        if args.max_docs:
            rows = np.sort(np.random.default_rng(args.seed).choice(docs.n_rows, args.max_docs, replace=False))
        with timer("building M (Eq. 1)"):
            M, df = build_cooccurrence(docs, dev, rows)
        np.save(m_path, M.cpu().numpy())
        np.save(df_path, df.cpu().numpy())
    n_docs_m = args.max_docs or docs.n_rows
    nz = float((M > 0).sum()) / M.numel()
    log(f"M: {V}x{V}, fraction of pairs with co-occurrence > 0: {nz:.3f}, max {float(M.max()):,.0f}")

    tdocs = docs_to_device(docs, dev)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(get_splade_model(paths)[0])

    # ------------------------------------------------------------ demo on a sentence
    if args.demo:
        from splade_encoder import SpladeEncoder
        model_name, scale = get_splade_model(paths)
        enc = SpladeEncoder(model_name).to(dev)
        qd = enc.encode_dense([args.demo], 256, dev) * scale
        del enc
        lines = [f"SENTENCE: {args.demo}"]
        for norm in args.norms:
            B = normalize_cooccurrence(M, df, norm, n_docs_m, args.min_count)
            vext = prune(expand(qd, B, args.alphas[len(args.alphas) // 2], args.alpha_mode), args.prunes[0], qd,
                         args.keep_original)
            o, a = show_expansion(tokenizer, qd[0], vext[0])
            lines += [f"[{norm}] original: {o}", f"[{norm}] ADDED   : {a}"]
            del B
        for ln in lines:
            log(ln)
        (paths.results / "lce_demo.txt").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------ experiment grid
    for set_name in args.query_sets:
        q, qrels, thr, texts = splade_query_vectors(set_name, paths, dev)
        qids = np.asarray(q.ids)
        configs = [("none", 0.0, "none")] + list(itertools.product(args.norms, args.alphas, args.prunes))
        grid_csv = paths.results / f"lce_grid_{set_name}.csv"
        best = None
        last_norm, B = None, None
        for norm, alpha, pr in configs:
            if norm != last_norm:
                B = None
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                B = normalize_cooccurrence(M, df, norm, n_docs_m, args.min_count) if norm != "none" else None
                last_norm = norm
            name = "splade" if norm == "none" else f"lce_{norm}_a{alpha}_{pr.replace(':', '')}"
            scores = np.empty((q.n_rows, args.k), dtype=np.float32)
            rows = np.empty((q.n_rows, args.k), dtype=np.int64)
            nnz = 0
            ex_lines = []
            with timer(f"[{set_name}] {name}"):
                for s in range(0, q.n_rows, args.q_batch):
                    e = min(q.n_rows, s + args.q_batch)
                    qd = csr_rows_to_dense(q, np.arange(s, e), dev)
                    vext = qd if B is None else prune(expand(qd, B, alpha, args.alpha_mode), pr, qd,
                                                      args.keep_original)
                    nnz += int((vext > 0).sum())
                    sc, ix = search_with_query_table(tdocs, vext.T.contiguous(), args.k)
                    scores[s:e] = sc.float().cpu().numpy()
                    rows[s:e] = ix.cpu().numpy()
                    if s == 0 and B is not None and set_name == "dev":
                        for i in range(min(args.examples, e - s)):
                            o, a = show_expansion(tokenizer, qd[i], vext[i], 8)
                            qt = texts.get(int(qids[i]), "") if texts else ""
                            ex_lines.append(f"  {qt!r}\n     orig: {o}\n     +LCE: {a}")
            for ln in ex_lines:
                log(ln)
            extra = {"norm": norm, "alpha": alpha, "alpha_mode": args.alpha_mode, "prune": pr,
                     "keep_original": args.keep_original, "avg_query_nnz": nnz / q.n_rows,
                     "M_docs": n_docs_m}
            m, _ = report_and_save(paths, name, set_name, qids, rows, scores, doc_pids, qrels, thr,
                                   extra=extra, save_run=False)
            append_csv_row(grid_csv, {"config": name, **extra, **{k: v for k, v in m.items() if k[0] in "MnR"}})
            if norm != "none" and (best is None or m["MRR@10"] > best[1]):
                best = (name, m["MRR@10"])
        log(f"[{set_name}] best LCE configuration: {best}. Full table: {grid_csv}")
        save_json({"best": best}, paths.results / f"lce_best_{set_name}.json")
    log("Done.")


if __name__ == "__main__":
    main()
