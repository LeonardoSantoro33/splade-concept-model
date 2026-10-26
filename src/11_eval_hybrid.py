"""
11_eval_hybrid.py — Hybrid retrieval: SPLADE + Concept Model, with the mixing weight chosen on validation.

    score(q, d) = (1 - lam) * SPLADE(q) . SPLADE(d)  +  lam * g(q) . g(d)

  lam = 0 -> pure SPLADE (original model), lam = 1 -> pure Concept Model.

The search is EXACT over all 8.8M passages, without ever building the dense matrix g(d):
    g(q) . g(d) = (W g(q)) . SPLADE_theta(d)
  - frozen checkpoint: SPLADE_theta = original SPLADE, so both terms use the provided document vectors;
  - e2e checkpoint:    SPLADE_theta = the fine-tuned encoder, whose sparse document vectors were already
                       saved by 08c_eval_e2e_streaming.py (WORK_DIR/embeddings/<run>_<ckpt>_sparse).
                       Run 08c on the checkpoint first.

Protocol (no tuning on the test sets):
  1) VALIDATION: the same 1000 held-out training queries used during training (cache of script 06),
     but with a full search over the corpus (not a re-ranking of 100 candidates). For every lam of the
     grid we compute MRR@10 and choose lam* = the best one.
  2) TEST: dev / dl19 / dl20 with lam*, saved in summary.csv as hyb_<run>_<ckpt>_lam<lam*>.
     The two endpoints (lam = 0 and lam = 1) are computed in the same pass and only printed in the log,
     as a sanity check: they must reproduce SPLADE (dev 0.3828) and the Concept Model alone.

    sbatch --time=02:00:00 -J hyb_e2e_k4096 jobs/gpu.sbatch 11_eval_hybrid.py \
        --ckpt e2e_k4096_r2_is0156_lr3e5_bb1e6_b32_ep1 --query-sets dev dl19 dl20

--variant (e2e checkpoints only; default "two"):
  two     (1-lam) SPLADE(q).SPLADE(d)            + lam g(q).g(d)       two indexes: original SPLADE + fine-tuned
  single  (1-lam) SPLADE_t(q).SPLADE_t(d)        + lam g(q).g(d)       ONE index (the fine-tuned one):
          = SPLADE_t(q)^T [(1-lam) I + lam W^T W] SPLADE_t(d)          same order of space as SPLADE.
          lam = 0 is the fine-tuned encoder used as a plain SPLADE. Saved as hyb1_<run>_lam<lam*>.
  now     (1-lam) SPLADE(q).SPLADE(d)            + lam SPLADE_t(q).SPLADE_t(d)   control WITHOUT W: is the gain of
          "two" due to the concepts, or just to combining two encoders? Saved as hybnoW_<run>_lam<lam*>.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from concept_model import load_checkpoint
from config import Paths, resolve_ckpt
from metrics import evaluate_run, format_metrics
from msmarco import positives_from_qrels, read_qrels, read_queries, load_query_set
from pipeline import (concept_vectors_from_csr, concept_vectors_from_texts, free_cuda, load_docs,
                      report_and_save, splade_query_vectors)
from retrieval import TorchCSR, _merge_topk, csr_rows_to_dense, gpu_fits, rows_times_table
from sparse_io import CSR, ids_to_rows
from utils import get_device, log, log_run_header, make_transformers_cpu_safe, save_json, timer

make_transformers_cpu_safe()

DEFAULT_GRID = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]


def load_e2e_shards(cache_dir, doc_pids):
    """Sparse document vectors of the fine-tuned encoder, saved by 08c. Checks they are complete and aligned."""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"[ERROR] {cache_dir} not found: run 08c_eval_e2e_streaming.py on this checkpoint first")
    meta = json.loads(meta_path.read_text())
    if int(meta["n_docs"]) != len(doc_pids):
        raise SystemExit(f"[ERROR] {cache_dir}: {meta['n_docs']} documents, expected {len(doc_pids)}")
    n_shards = (len(doc_pids) + int(meta["shard_size"]) - 1) // int(meta["shard_size"])
    shards, off = [], 0
    with timer(f"loading {n_shards} sparse shards of the fine-tuned encoder"):
        for si in range(n_shards):
            f = cache_dir / f"shard_{si:04d}"
            if not (f / "meta.json").exists():
                raise SystemExit(f"[ERROR] {f} missing: 08c did not finish encoding (rerun it, it resumes)")
            c = CSR.load(f)
            if c.ids is not None and not np.array_equal(np.asarray(c.ids), doc_pids[off:off + c.n_rows]):
                raise SystemExit(f"[ERROR] {f}: document ids not aligned with the provided documents")
            shards.append((c, off))
            off += c.n_rows
    if off != len(doc_pids):
        raise SystemExit(f"[ERROR] shards contain {off} documents, expected {len(doc_pids)}")
    return shards


class Corpus:
    """Document blocks: for each block, the original SPLADE vectors and the vectors used by the Concept Model
    (the same object for frozen, the fine-tuned ones for e2e)."""

    def __init__(self, docs, shards, dev, block=500_000, use_orig=True):
        self.dev = dev
        self.n = docs.n_rows
        need_orig = docs.nnz * 8 + docs.n_rows * 8
        need_cm = 0 if shards is None else sum(c.nnz * 8 + c.n_rows * 8 for c, _ in shards)
        self.orig = None
        if use_orig:
            if not gpu_fits(need_orig, dev, margin=0.85):
                raise SystemExit(f"[ERROR] the original documents ({need_orig / 1e9:.1f} GB) do not fit on the GPU")
            with timer("copying the original SPLADE documents to the GPU"):
                self.orig = TorchCSR(docs, dev)
        else:
            log("original SPLADE documents NOT used (single-index variant)")
        if shards is None:                                   # frozen: same vectors for both terms
            self.blocks = [(s, min(self.n, s + block)) for s in range(0, self.n, block)]
            self.cm_shards, self.cm_resident = None, True
        else:
            self.blocks = [(off, off + c.n_rows) for c, off in shards]
            self.cm_resident = gpu_fits(need_cm, dev, margin=0.6)
            log(f"fine-tuned documents: {need_cm / 1e9:.1f} GB -> "
                f"{'kept on the GPU' if self.cm_resident else 'streamed from the CPU for each query batch'}")
            self.cm_cpu = [c for c, _ in shards]
            self.cm_shards = [TorchCSR(c, dev) for c in self.cm_cpu] if self.cm_resident else None

    def scores(self, j, terms):
        """terms = [(index, table), (index, table)] with index "orig" (original SPLADE documents) or "ft"
        (fine-tuned documents; = orig for frozen). Returns the (nq, block) score matrix of each term for block j."""
        s, e = self.blocks[j]
        tmp = None
        out = []
        for index, table in terms:
            if index == "orig" or self.cm_shards is None and self.cm_resident:     # frozen: ft == orig
                out.append(rows_times_table(self.orig, s, e, table).T)
            elif self.cm_resident:
                out.append(rows_times_table(self.cm_shards[j], 0, e - s, table).T)
            else:
                if tmp is None:
                    tmp = TorchCSR(self.cm_cpu[j], self.dev)
                out.append(rows_times_table(tmp, 0, e - s, table).T)
        if tmp is not None:
            tmp.free()
        return out


@torch.no_grad()
def term_tables(variant, q_csr, sq, gq, W, qs, qe, dev):
    """Query-side tables (vocab, nb) of the two terms and the index each one is scored against.
    q_csr: original SPLADE query vectors; sq: fine-tuned SPLADE_t(q) (dense, or None); gq: g(q)."""
    def orig_q():
        return csr_rows_to_dense(q_csr, np.arange(qs, qe), dev).T.contiguous()
    def ft_q():
        return sq[qs:qe].float().T.contiguous()
    def cm_q():
        return (W @ gq[qs:qe].float().T).contiguous()
    if variant == "two":
        return [("orig", orig_q()), ("ft", cm_q())]
    if variant == "single":
        return [("ft", ft_q()), ("ft", cm_q())]
    if variant == "now":
        return [("orig", orig_q()), ("ft", ft_q())]
    raise ValueError(variant)


@torch.no_grad()
def hybrid_search(corpus, q_csr, gq, W, lambdas, k, q_batch=128, variant="two", sq=None):
    """Exact top-k of (1-lam) term1 + lam term2 for every lam in `lambdas`, in a single pass over the corpus
    (terms: see term_tables). Returns {lam: (scores (nq,k), rows (nq,k))} as numpy arrays."""
    nq = q_csr.n_rows
    k = min(k, corpus.n)
    out = {lam: (np.empty((nq, k), np.float32), np.empty((nq, k), np.int64)) for lam in lambdas}
    W = W.float()
    for b, qs in enumerate(range(0, nq, q_batch)):
        qe = min(nq, qs + q_batch)
        terms = term_tables(variant, q_csr, sq, gq, W, qs, qe, corpus.dev)
        best = {lam: (None, None) for lam in lambdas}
        for j, (s, e) in enumerate(corpus.blocks):
            S1, S2 = corpus.scores(j, terms)
            kk = min(k, e - s)
            for lam in lambdas:
                S = S1 if lam == 0 else (S2 if lam == 1 else (1.0 - lam) * S1 + lam * S2)
                sc, ix = torch.topk(S, kk, dim=1)
                bs, bi = best[lam]
                best[lam] = _merge_topk(bs, bi, sc, ix + s, k, dim=1)
            del S1, S2
        for lam in lambdas:
            bs, bi = best[lam]
            out[lam][0][qs:qe] = bs.float().cpu().numpy()
            out[lam][1][qs:qe] = bi.cpu().numpy()
        if b % 10 == 0:
            log(f"  {qe:,}/{nq:,} queries")
    return out


def cm_query_vectors(head, info, q_csr, qids, texts, encoder, dev, q_max_len):
    """g(q): from the SPLADE vectors for frozen, from the texts with the fine-tuned encoder for e2e."""
    if info["mode"] == "frozen":
        return concept_vectors_from_csr(head, q_csr, dev)
    missing = [q for q in qids.tolist() if q not in texts]
    if missing:
        raise SystemExit(f"[ERROR] {len(missing)} queries without text (e.g. {missing[:3]})")
    return concept_vectors_from_texts(head, encoder, [texts[q] for q in qids.tolist()], dev, q_max_len)


@torch.no_grad()
def ft_query_vectors(head, encoder, texts, dev, max_length, batch=128, amp=True):
    """SPLADE_t(q) (dense, vocab) and g(q) = W . SPLADE_t(q) with the fine-tuned encoder, in one pass."""
    encoder.eval()
    texts = list(texts)
    order = np.argsort([len(t) for t in texts], kind="stable")
    sq = torch.empty(len(texts), head.vocab_size, dtype=torch.float32, device=dev)
    gq = torch.empty(len(texts), head.k, dtype=torch.float32, device=dev)
    for s in range(0, len(texts), batch):
        idx = torch.from_numpy(order[s:s + batch]).to(dev)
        x = encoder.encode_dense([texts[i] for i in order[s:s + batch]], max_length, dev, amp=amp)
        sq[idx] = x.float()
        gq[idx] = head.forward_dense(x).float()
    return sq, gq


def query_vectors(variant, head, info, q_csr, qids, texts, encoder, dev, q_max_len):
    """(sq, gq): sq = SPLADE_t(q) only when the variant needs it, else None."""
    if variant == "two":
        return None, cm_query_vectors(head, info, q_csr, qids, texts, encoder, dev, q_max_len)
    missing = [q for q in qids.tolist() if q not in texts]
    if missing:
        raise SystemExit(f"[ERROR] {len(missing)} queries without text (e.g. {missing[:3]})")
    return ft_query_vectors(head, encoder, [texts[q] for q in qids.tolist()], dev, q_max_len)


TERM_NAMES = {"two": ("SPLADE alone", "Concept Model alone"),
              "single": ("fine-tuned encoder as SPLADE alone", "Concept Model alone"),
              "now": ("SPLADE alone", "fine-tuned encoder as SPLADE alone")}
PREFIX = {"two": "hyb", "single": "hyb1", "now": "hybnoW"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run name (uses .../best) or path; frozen or e2e")
    ap.add_argument("--query-sets", nargs="+", default=["dev"], choices=["dev", "dl19", "dl20"])
    ap.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_GRID, help="grid searched on validation")
    ap.add_argument("--lam", type=float, default=None, help="use this lambda and skip validation")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--val-k", type=int, default=100, help="depth of the validation search (MRR@10 needs 10)")
    ap.add_argument("--name", default=None, help="method name in the results")
    ap.add_argument("--q-batch", type=int, default=128)
    ap.add_argument("--q-max-len", type=int, default=64, help="query length for the e2e encoder")
    ap.add_argument("--block", type=int, default=500_000, help="documents per block (frozen)")
    ap.add_argument("--variant", default="two", choices=["two", "single", "now"],
                    help="two = SPLADE + CM (2 indexes); single = fine-tuned SPLADE + CM (1 index); "
                         "now = SPLADE + fine-tuned SPLADE, no W (control). single/now need an e2e checkpoint")
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    dev = get_device()
    log_run_header("11_eval_hybrid.py", args, dev)

    ckpt = resolve_ckpt(args.ckpt, paths)
    head, info = load_checkpoint(ckpt, dev)
    head.eval()
    run = f"{ckpt.parent.name}_{ckpt.name}"
    W = head.weight.detach()
    if args.variant != "two" and info["mode"] != "e2e":
        raise SystemExit(f"[ERROR] --variant {args.variant} needs an e2e checkpoint")
    pre = PREFIX[args.variant]
    name0, name1 = TERM_NAMES[args.variant]
    log(f"variant: {args.variant}  (lambda=0 -> {name0}, lambda=1 -> {name1})")

    docs = load_docs(paths, mmap=False)
    doc_pids = np.asarray(docs.ids, dtype=np.int64)
    encoder, shards = None, None
    if info["mode"] == "e2e":
        if not info.get("backbone_path"):
            raise SystemExit("[ERROR] e2e checkpoint without backbone/")
        shards = load_e2e_shards(paths.embeddings / f"{run}_sparse", doc_pids)
        from splade_encoder import SpladeEncoder
        encoder = SpladeEncoder(info["backbone_path"]).to(dev)
    corpus = Corpus(docs, shards, dev, block=args.block, use_orig=(args.variant != "single"))
    free_cuda()

    # ------------------------------------------------------------------ 1) validation: choose lambda
    if args.lam is None:
        found = sorted(paths.val_candidates.glob("seed42_n1000_c100_*.npz"))
        if not found:
            raise SystemExit(f"[ERROR] validation cache not found in {paths.val_candidates} (it is created by 06)")
        val_rows = np.load(found[0])["val_rows"]
        train_q = CSR.load(paths.train_queries_csr)
        q_val = train_q.subset(val_rows)
        qids_val = np.asarray(q_val.ids, dtype=np.int64)
        pid_ok = ids_to_rows(doc_pids)
        pos = positives_from_qrels(read_qrels(paths.qrels_tsv("train")))
        qrels_val = {}
        for q in qids_val.tolist():
            ok = [p for p, r in zip(pos[q], pid_ok(np.array(pos[q]))) if r >= 0]
            qrels_val[q] = {p: 1 for p in ok}
        texts_val = read_queries(paths.queries_tsv("train")) if info["mode"] == "e2e" else None
        sq, gq = query_vectors(args.variant, head, info, q_val, qids_val, texts_val, encoder, dev, args.q_max_len)
        lambdas = sorted(set(args.lambdas) | {0.0, 1.0})
        with timer(f"validation: full search, {len(qids_val)} queries, {len(lambdas)} values of lambda"):
            res = hybrid_search(corpus, q_val, gq, W, lambdas, args.val_k, args.q_batch, args.variant, sq)
        curve = []
        log(f"{'lambda':>7}  {'MRR@10':>7}  {'nDCG@10':>7}  {'R@100':>7}")
        for lam in lambdas:
            m, _ = evaluate_run(qids_val, doc_pids[res[lam][1]], qrels_val)
            curve.append({"lambda": lam, **m})
            log(f"{lam:7.2f}  {m['MRR@10']:7.4f}  {m['nDCG@10']:7.4f}  {m.get('R@100', float('nan')):7.4f}")
        best = max(curve, key=lambda r: r["MRR@10"])
        lam_star = float(best["lambda"])
        log(f"lambda* = {lam_star:.2f} (val MRR@10 {best['MRR@10']:.4f}; "
            f"{name0} {curve[0]['MRR@10']:.4f}, {name1} {curve[-1]['MRR@10']:.4f})")
        paths.results.mkdir(parents=True, exist_ok=True)
        save_json({"ckpt": str(ckpt), "variant": args.variant, "lambda_star": lam_star, "curve": curve},
                  paths.results / f"{pre}_{run}_validation.json")
        del gq, sq, res
        free_cuda()
    else:
        lam_star = float(args.lam)
        log(f"lambda fixed to {lam_star:.2f} (validation skipped)")

    # ------------------------------------------------------------------ 2) test sets
    name = args.name or f"{pre}_{run}_lam{lam_star:.2f}"
    test_lams = sorted({0.0, lam_star, 1.0})
    for set_name in args.query_sets:
        q_csr, qrels, thr, texts = splade_query_vectors(set_name, paths, dev)
        qids = np.asarray(q_csr.ids, dtype=np.int64)
        if texts is None and info["mode"] == "e2e":
            texts, _, _ = load_query_set(set_name, paths, need_texts=True)
        sq, gq = query_vectors(args.variant, head, info, q_csr, qids, texts, encoder, dev, args.q_max_len)
        with timer(f"hybrid search on '{set_name}' ({len(qids):,} queries)"):
            res = hybrid_search(corpus, q_csr, gq, W, test_lams, args.k, args.q_batch, args.variant, sq)
        for lam in test_lams:
            if lam in (0.0, 1.0) and lam != lam_star:
                m, _ = evaluate_run(qids, doc_pids[res[lam][1]], qrels, rel_threshold=thr)
                log(f"[sanity lambda={lam:.0f} ({name0 if lam == 0 else name1}) | "
                    f"{set_name}] {format_metrics(m)}")
        report_and_save(paths, name, set_name, qids, res[lam_star][1], res[lam_star][0], doc_pids, qrels, thr,
                        extra={"lambda": lam_star, "k_concepts": head.k, "mode": info["mode"],
                               "ckpt": str(ckpt), "search": "hybrid_exact", "variant": args.variant})
        if args.variant == "single" and lam_star != 0.0:
            # the fine-tuned encoder used as a plain SPLADE: a baseline of its own (saved for the significance test)
            report_and_save(paths, f"ftsplade_{run}", set_name, qids, res[0.0][1], res[0.0][0], doc_pids, qrels,
                            thr, extra={"lambda": 0.0, "mode": info["mode"], "ckpt": str(ckpt),
                                        "search": "exact", "variant": "single_lambda0"})
        del gq, sq, res
        free_cuda()
    log("Done.")


if __name__ == "__main__":
    main()
