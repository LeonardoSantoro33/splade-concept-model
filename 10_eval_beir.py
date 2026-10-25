"""
10_eval_beir.py — Zero-shot evaluation OUT OF DOMAIN on BEIR datasets (no training).

The models trained on MS MARCO (SPLADE, Concept Model frozen / random W / e2e) are evaluated
on small BEIR datasets (default: NFCorpus, SciFact, FiQA). Same exact search and same metrics
as 04/08; the main BEIR metric is nDCG@10.

  - SPLADE baseline : SPLADE vectors (sparse) of documents and queries, exact sparse search
  - frozen / rand   : g(x) = W . SPLADE(x) with the ORIGINAL SPLADE (encoded once per dataset)
  - e2e             : g(x) = W . SPLADE_theta(x) with the fine-tuned BERT saved in the checkpoint
  - --ft-splade     : the fine-tuned encoder of an e2e checkpoint used as a PLAIN SPLADE (W thrown away):
                      score = SPLADE_theta(q) . SPLADE_theta(d), exact sparse search, same settings as the
                      SPLADE baseline (same max lengths, same scale). Saved as ftsplade_<run>_<ckpt>.
  - --hybrid RUN:VARIANT:LAMBDA   hybrid of 11_eval_hybrid.py, with LAMBDA FIXED (the lambda* chosen on the
                      MS MARCO validation: nothing is tuned on BEIR). VARIANT:
                        two : (1-lam) SPLADE(q).SPLADE(d) + lam g(q).g(d)                 -> hyb_<run>_<ckpt>_lam<lam>
                        now : (1-lam) SPLADE(q).SPLADE(d) + lam SPLADE_t(q).SPLADE_t(d)   -> hybnoW_<run>_<ckpt>_lam<lam>
                        single : (1-lam) SPLADE_t(q).SPLADE_t(d) + lam g(q).g(d)          -> hyb1_<run>_<ckpt>_lam<lam>
                                 (single index: the original SPLADE documents are not used in the score)
                      The fine-tuned side is encoded exactly as in 08c/11 (scale 1, amp, query length 64,
                      document length 256), so the scores have the same magnitude as on MS MARCO.
                      RUN can also be a FROZEN checkpoint (backbone not retrained, so there is no
                      fine-tuned encoder): then only VARIANT two is possible, with g = W . SPLADE of the
                      ORIGINAL SPLADE (same vectors and scale as the frozen evaluation above):
                        two (frozen) : (1-lam) SPLADE(q).SPLADE(d) + lam (W.SPLADE(q)).(W.SPLADE(d))

Documents are "title + text" (standard BEIR convention). Queries: those of the test qrels.
BEIR ids are strings: they are mapped to integers (row number) internally.
Results go to results/summary.csv with query_set = beir_<dataset>.

    # 1) download only (no GPU needed, e.g. from the login node)
    python -u 10_eval_beir.py --datasets nfcorpus scifact fiqa --download-only
    # 2) evaluation
    sbatch --time=04:00:00 -J beir jobs/gpu.sbatch 10_eval_beir.py --datasets nfcorpus scifact fiqa \
        --splade --ckpts frozen_k1024_rand frozen_k1024_r2_lr3e5 e2e_k1024_r2_is03125_lr3e5_bb5e6_b32_ep1
    # 3) fine-tuned encoders as plain SPLADE
    sbatch --time=02:00:00 -J beir_ft jobs/gpu.sbatch 10_eval_beir.py --datasets nfcorpus scifact fiqa \
        --ft-splade e2e_k2048_r2_is0221_lr3e5_bb5e6_b32_ep1
    # 4) hybrids with the lambda chosen on MS MARCO
    sbatch --time=02:00:00 -J beir_hyb jobs/gpu.sbatch 10_eval_beir.py --datasets nfcorpus scifact fiqa \
        --hybrid e2e_k256_r2_is0625_lr1e4_bb1e5_b32_ep1:now:0.40 e2e_k256_r2_is0625_lr1e4_bb1e5_b32_ep1:two:0.20
    # 5) hybrid with a FROZEN checkpoint (only variant two)
    sbatch --time=02:00:00 -J beir_hybfrozen jobs/gpu.sbatch 10_eval_beir.py --datasets nfcorpus scifact fiqa \
        --hybrid frozen_k256_r2:two:0.05
"""
import argparse
import csv
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch

from concept_model import load_checkpoint
from config import Paths, get_splade_model, resolve_ckpt
from msmarco import download
from metrics import evaluate_run, format_metrics
from pipeline import concept_vectors_from_csr, concept_vectors_from_texts, free_cuda, report_and_save
from retrieval import TorchCSR, csr_rows_to_dense, dense_search, project_rows, rows_times_table, sparse_search
from sparse_io import CSR
from utils import get_device, log, log_run_header, make_transformers_cpu_safe, timer

make_transformers_cpu_safe()

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


# ----------------------------------------------------------------------
#  data
# ----------------------------------------------------------------------
def fetch_dataset(name, beir_dir):
    """Download and unzip a BEIR dataset into beir_dir/<name>/ (skipped if already there)."""
    folder = beir_dir / name
    if (folder / "corpus.jsonl").exists():
        log(f"[{name}] already present in {folder}")
        return folder
    beir_dir.mkdir(parents=True, exist_ok=True)
    zpath = download(BEIR_URL.format(name=name), beir_dir / f"{name}.zip")
    with timer(f"[{name}] unzipping"):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(beir_dir)
    if not (folder / "corpus.jsonl").exists():
        raise SystemExit(f"[ERROR] {folder}/corpus.jsonl not found after unzipping {zpath}")
    return folder


def load_beir(folder, split="test"):
    """Returns (doc_texts, doc_ids_str, query_texts, query_ids_str, qrels{int_q: {int_d: rel}}).
    Integer ids = position in doc_ids_str / query_ids_str."""
    doc_ids, doc_texts = [], []
    with open(folder / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            doc_ids.append(str(o["_id"]))
            doc_texts.append(f"{o.get('title', '') or ''} {o.get('text', '') or ''}".strip())
    d2i = {d: i for i, d in enumerate(doc_ids)}

    all_q = {}
    with open(folder / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            all_q[str(o["_id"])] = o.get("text", "") or ""

    qrels_str = {}
    with open(folder / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0] == "query-id":
                continue
            q, d, r = str(row[0]), str(row[1]), int(float(row[2]))
            qrels_str.setdefault(q, {})[d] = r

    q_ids = sorted(q for q in qrels_str if q in all_q)
    q2i = {q: i for i, q in enumerate(q_ids)}
    qrels, n_unknown = {}, 0
    for q in q_ids:
        for d, r in qrels_str[q].items():
            if d not in d2i:
                n_unknown += 1
                continue
            qrels.setdefault(q2i[q], {})[d2i[d]] = r
    if n_unknown:
        log(f"WARNING: {n_unknown} qrels point to documents not in the corpus (ignored)")
    return doc_texts, doc_ids, [all_q[q] for q in q_ids], q_ids, qrels


# ----------------------------------------------------------------------
#  hybrid helpers
# ----------------------------------------------------------------------
HYB_PREFIX = {"two": "hyb", "now": "hybnoW", "single": "hyb1"}
# fine-tuned-side score terms needed by each variant: "cm" = g(q).g(d) (Concept Model), "ft" = SPLADE_t(q).SPLADE_t(d)
TERMS_OF = {"two": ("cm",), "now": ("ft",), "single": ("ft", "cm")}


def parse_hybrids(specs):
    """'RUN:VARIANT:LAMBDA' -> list of (run, variant, lam)."""
    out = []
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3 or parts[1] not in HYB_PREFIX:
            raise SystemExit(f"[ERROR] --hybrid {spec}: expected RUN:VARIANT:LAMBDA with VARIANT in two/now/single")
        lam = float(parts[2])
        if not 0.0 <= lam <= 1.0:
            raise SystemExit(f"[ERROR] --hybrid {spec}: lambda must be in [0, 1]")
        out.append((parts[0], parts[1], lam))
    return out


@torch.no_grad()
def ft_dense_queries(enc, texts, dev, max_length, batch=128):
    """SPLADE_theta(q) as a dense (nq, vocab) float32 matrix (same settings as 11: amp, query length 64)."""
    was_training = enc.training
    enc.eval()                       # no dropout (encode_to_csr restores the previous mode)
    out = []
    for s in range(0, len(texts), batch):
        out.append(enc.encode_dense(texts[s:s + batch], max_length, dev, amp=True).float())
    enc.train(was_training)
    return torch.cat(out, 0)


@torch.no_grad()
def full_scores(tcsr, table, block=200_000):
    """(documents, queries) score matrix = (sparse documents) @ table, table = (vocab, nq)."""
    parts = [rows_times_table(tcsr, s, min(tcsr.n_rows, s + block), table)
             for s in range(0, tcsr.n_rows, block)]
    return torch.cat(parts, 0)


def topk_rows(S, k):
    """S (documents, queries) -> scores (nq, k), rows (nq, k) as numpy."""
    sc, ix = torch.topk(S, min(k, S.shape[0]), dim=0)
    return sc.T.float().cpu().numpy(), ix.T.cpu().numpy().astype(np.int64)


# ----------------------------------------------------------------------
#  main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["nfcorpus", "scifact", "fiqa"])
    ap.add_argument("--ckpts", nargs="*", default=[], help="run names (frozen, rand or e2e) to evaluate")
    ap.add_argument("--splade", action="store_true", help="also evaluate the SPLADE baseline (sparse)")
    ap.add_argument("--ft-splade", nargs="*", default=[],
                    help="e2e run names: evaluate their fine-tuned encoder as a plain SPLADE (no W)")
    ap.add_argument("--hybrid", nargs="*", default=[],
                    help="RUN:VARIANT:LAMBDA with VARIANT two, now or single and LAMBDA chosen on MS MARCO "
                         "validation (RUN frozen: only VARIANT two)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--d-max-len", type=int, default=256)
    ap.add_argument("--q-max-len-splade", type=int, default=256, help="as pipeline.splade_query_vectors")
    ap.add_argument("--q-max-len-e2e", type=int, default=64, help="as 08 / training")
    ap.add_argument("--data-dir", default=None, help="folder with the BEIR datasets (default WORK_DIR/beir)")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    beir_dir = Path(args.data_dir) if args.data_dir else paths.work / "beir"

    folders = {name: fetch_dataset(name, beir_dir) for name in args.datasets}
    if args.download_only:
        log("Download done.")
        return

    dev = get_device()
    log_run_header("10_eval_beir.py", args, dev)
    model_name, scale = get_splade_model(paths)
    log(f"original SPLADE model: {model_name} (scale {scale})")

    # load the checkpoints once (they are small); e2e encoders are loaded one at a time
    heads = []
    for run in args.ckpts:
        ckpt = resolve_ckpt(run, paths)
        head, info = load_checkpoint(ckpt, dev)
        head.eval()
        heads.append((f"cm_{ckpt.parent.name}_{ckpt.name}", ckpt, head, info))

    from splade_encoder import SpladeEncoder

    ft_ckpts = []
    for run in args.ft_splade:
        ckpt = resolve_ckpt(run, paths)
        _, info = load_checkpoint(ckpt, "cpu")
        if info["mode"] != "e2e" or not info.get("backbone_path"):
            raise SystemExit(f"[ERROR] --ft-splade {run}: not an e2e checkpoint with a backbone/ folder")
        ft_ckpts.append((f"ftsplade_{ckpt.parent.name}_{ckpt.name}", ckpt, info))

    hybrids = parse_hybrids(args.hybrid)
    hyb_ckpts = {}
    for run, variant, lam in hybrids:
        if run not in hyb_ckpts:
            ckpt = resolve_ckpt(run, paths)
            head, info = load_checkpoint(ckpt, dev)
            head.eval()
            # a frozen checkpoint (backbone not retrained) is allowed too, see the check below;
            # anything else must be an e2e checkpoint with a backbone
            if info["mode"] != "frozen" and (info["mode"] != "e2e" or not info.get("backbone_path")):
                raise SystemExit(f"[ERROR] --hybrid {run}: not an e2e checkpoint with a backbone/ folder")
            hyb_ckpts[run] = (ckpt, head, info)
    for run, variant, lam in hybrids:
        if hyb_ckpts[run][2]["mode"] == "frozen" and variant != "two":
            raise SystemExit(f"[ERROR] --hybrid {run}:{variant}:{lam}: a frozen checkpoint has no fine-tuned "
                             f"encoder, only VARIANT two is possible")

    for ds, folder in folders.items():
        set_name = f"beir_{ds}"
        doc_texts, doc_ids, q_texts, q_ids, qrels = load_beir(folder, args.split)
        log(f"[{ds}] documents: {len(doc_texts):,}  test queries: {len(q_texts):,}  "
            f"judged queries: {len(qrels):,}")
        doc_rows = np.arange(len(doc_texts), dtype=np.int64)     # row -> integer doc id
        qids = np.arange(len(q_texts), dtype=np.int64)
        k = min(args.k, len(doc_texts))

        # ---------------- original SPLADE: encode once (cached on disk) ----------------
        need_splade = args.splade or bool(hybrids) or any(info["mode"] != "e2e" for *_, info in heads)
        tdocs = q_csr = None
        if need_splade:
            tag = model_name.replace("/", "__")
            cache = beir_dir / ds / f"splade_docs_{tag}_L{args.d_max_len}"
            if (cache / "meta.json").exists():
                docs_csr = CSR.load(cache)
                log(f"[{ds}] SPLADE documents loaded from cache {cache}")
            else:
                enc = SpladeEncoder(model_name).to(dev)
                with timer(f"[{ds}] SPLADE encoding of {len(doc_texts):,} documents"):
                    docs_csr = enc.encode_to_csr(doc_texts, dev, batch_size=128, max_length=args.d_max_len,
                                                 ids=doc_rows, scale=scale)
                # atomic save: several jobs may run in parallel on the same dataset
                tmp = cache.with_name(f"{cache.name}.tmp{os.getpid()}")
                docs_csr.save(tmp)
                try:
                    tmp.rename(cache)
                except OSError:          # another job saved it first: keep theirs
                    shutil.rmtree(tmp, ignore_errors=True)
                del enc
                free_cuda()
            log(f"[{ds}] SPLADE documents: mean {docs_csr.nnz / max(1, docs_csr.n_rows):.1f} non-zero entries")
            enc = SpladeEncoder(model_name).to(dev)
            q_csr = enc.encode_to_csr(q_texts, dev, batch_size=64, max_length=args.q_max_len_splade,
                                      ids=qids, scale=scale, verbose=False)
            del enc
            free_cuda()
            tdocs = TorchCSR(docs_csr, dev)

        if args.splade:
            with timer(f"[{ds}] exact SPLADE search"):
                scores, rows = sparse_search(tdocs, q_csr, k, verbose=False)
            report_and_save(paths, f"splade_baseline_beir", set_name, qids, rows, scores, doc_rows, qrels, 1,
                            extra={"dataset": ds, "n_docs": len(doc_texts)})

        # ---------------- Concept Model checkpoints ----------------
        for name, ckpt, head, info in heads:
            if info["mode"] == "e2e":
                if not info.get("backbone_path"):
                    log(f"WARNING: {ckpt} is e2e but has no backbone folder: skipped")
                    continue
                enc = SpladeEncoder(info["backbone_path"]).to(dev)
                with timer(f"[{ds}] {name}: e2e encoding of documents and queries"):
                    D = concept_vectors_from_texts(head, enc, doc_texts, dev, args.d_max_len).half()
                    gq = concept_vectors_from_texts(head, enc, q_texts, dev, args.q_max_len_e2e)
                del enc
                free_cuda()
            else:
                D = project_rows(tdocs, head.weight.detach())
                gq = concept_vectors_from_csr(head, q_csr, dev)
            scores, rows = dense_search(D, gq, k, verbose=False)
            report_and_save(paths, name, set_name, qids, rows, scores, doc_rows, qrels, 1,
                            extra={"dataset": ds, "n_docs": len(doc_texts), "k_concepts": head.k,
                                   "mode": info["mode"], "ckpt": str(ckpt)})
            del D, gq
            free_cuda()

        # ---------------- fine-tuned encoders as plain SPLADE (no W) ----------------
        for name, ckpt, info in ft_ckpts:
            enc = SpladeEncoder(info["backbone_path"]).to(dev)
            with timer(f"[{ds}] {name}: sparse encoding of documents and queries"):
                ft_docs = enc.encode_to_csr(doc_texts, dev, batch_size=128, max_length=args.d_max_len,
                                            ids=doc_rows, scale=scale)
                ft_q = enc.encode_to_csr(q_texts, dev, batch_size=64, max_length=args.q_max_len_splade,
                                         ids=qids, scale=scale, verbose=False)
            del enc
            free_cuda()
            log(f"[{ds}] {name}: mean {ft_docs.nnz / max(1, ft_docs.n_rows):.1f} non-zero entries per document")
            t = TorchCSR(ft_docs, dev)
            with timer(f"[{ds}] {name}: exact sparse search"):
                scores, rows = sparse_search(t, ft_q, k, verbose=False)
            report_and_save(paths, name, set_name, qids, rows, scores, doc_rows, qrels, 1,
                            extra={"dataset": ds, "n_docs": len(doc_texts), "mode": "ft_splade",
                                   "ckpt": str(ckpt)})
            t.free()
            del t, ft_docs, ft_q
            free_cuda()

        # ---------------- hybrids with a fixed lambda (chosen on MS MARCO) ----------------
        if hybrids:
            q_orig = csr_rows_to_dense(q_csr, np.arange(q_csr.n_rows), dev).T.contiguous()   # (vocab, nq)
            S1 = full_scores(tdocs, q_orig)                                                   # SPLADE term
            del q_orig
            for run, (ckpt, head, info) in hyb_ckpts.items():
                if info["mode"] == "frozen":
                    # FROZEN checkpoint: the backbone is the ORIGINAL SPLADE, so g(q).g(d) uses the same
                    # vectors (and scale) as the frozen evaluation above: g(d) = W . SPLADE(d) never gets
                    # built, the score is (SPLADE docs) @ (W @ g(q)^T), as in the e2e branch below.
                    with timer(f"[{ds}] {run}: frozen Concept Model term"):
                        gq_f = concept_vectors_from_csr(head, q_csr, dev).to(dev).float()
                        table = (head.weight.detach().float() @ gq_f.T).contiguous()
                        S2 = full_scores(tdocs, table)
                        del table, gq_f
                    for r2, variant, lam in hybrids:
                        if r2 != run:
                            continue
                        name = f"{HYB_PREFIX[variant]}_{ckpt.parent.name}_{ckpt.name}_lam{lam:.2f}"
                        sc, rw = topk_rows(S2, k)
                        m, _ = evaluate_run(qids, doc_rows[rw], qrels, rel_threshold=1)
                        log(f"[sanity lambda=1 (Concept Model alone) | {set_name}] {format_metrics(m)}")
                        sc, rw = topk_rows((1.0 - lam) * S1 + lam * S2, k)
                        report_and_save(paths, name, set_name, qids, rw, sc, doc_rows, qrels, 1,
                                        extra={"dataset": ds, "n_docs": len(doc_texts), "lambda": lam,
                                               "variant": variant, "ckpt": str(ckpt), "search": "hybrid_exact",
                                               "k_concepts": head.k, "mode": "frozen"})
                    del S2
                    free_cuda()
                    continue
                enc = SpladeEncoder(info["backbone_path"]).to(dev)
                with timer(f"[{ds}] {run}: fine-tuned encoding (as in 08c/11)"):
                    ft_docs = enc.encode_to_csr(doc_texts, dev, batch_size=128, max_length=args.d_max_len,
                                                amp=True, ids=doc_rows, verbose=False)
                    sq = ft_dense_queries(enc, q_texts, dev, args.q_max_len_e2e)
                del enc
                free_cuda()
                t_ft = TorchCSR(ft_docs, dev)
                terms = {}
                for r2, variant, lam in hybrids:
                    if r2 != run:
                        continue
                    for tname in TERMS_OF[variant]:
                        if tname in terms:
                            continue
                        if tname == "cm":
                            table = (head.weight.detach().float() @ head.forward_dense(sq).float().T).contiguous()
                        else:
                            table = sq.T.contiguous()
                        terms[tname] = full_scores(t_ft, table)
                        del table
                for r2, variant, lam in hybrids:
                    if r2 != run:
                        continue
                    # (1-lam) * first + lam * second:
                    #   two: SPLADE | g     now: SPLADE | SPLADE_t     single: SPLADE_t | g
                    if variant == "single":
                        S_first, S2 = terms["ft"], terms["cm"]
                    else:
                        S_first, S2 = S1, terms["cm" if variant == "two" else "ft"]
                    name = f"{HYB_PREFIX[variant]}_{ckpt.parent.name}_{ckpt.name}_lam{lam:.2f}"
                    other = "fine-tuned encoder as SPLADE alone" if variant == "now" else "Concept Model alone"
                    sc, rw = topk_rows(S2, k)
                    m, _ = evaluate_run(qids, doc_rows[rw], qrels, rel_threshold=1)
                    log(f"[sanity lambda=1 ({other}) | {set_name}] {format_metrics(m)}")
                    sc, rw = topk_rows((1.0 - lam) * S_first + lam * S2, k)
                    report_and_save(paths, name, set_name, qids, rw, sc, doc_rows, qrels, 1,
                                    extra={"dataset": ds, "n_docs": len(doc_texts), "lambda": lam,
                                           "variant": variant, "ckpt": str(ckpt), "search": "hybrid_exact"})
                t_ft.free()
                del t_ft, ft_docs, sq, terms
                free_cuda()
            del S1
            free_cuda()

        if tdocs is not None:
            tdocs.free()
            del tdocs
        free_cuda()
    log("Done.")


if __name__ == "__main__":
    main()
