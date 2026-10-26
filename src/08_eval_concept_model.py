"""
08_eval_concept_model.py — Final evaluation of the Concept Model on the FULL corpus + interpretability.

1) Exact search of g(q).g(d) over the 8.8M passages -> MRR@10, nDCG@10, R@100, R@1000
   (same metrics as baseline 04, so the comparison in the paper is direct).
2) concepts_<name>.tsv : for each concept j, the tokens with the highest/lowest weight in W[:, j]
   ("k nameable concepts", Sec. 5.1 and 8).
3) explain_<name>.txt (--explain N): for N dev queries, which concepts drive the match and an
   example of a Concept Bottleneck-style INTERVENTION (Sec. 8): zero out the concept with the
   largest contribution and observe how the rank of the relevant passage changes.

    sbatch -J eval_r2 jobs/gpu.sbatch 08_eval_concept_model.py --ckpt frozen_k256_r2 --query-sets dev dl19 dl20 --explain 5
"""
import argparse

import numpy as np
import torch

from concept_model import load_checkpoint
from config import Paths, get_splade_model, resolve_ckpt
from msmarco import Collection, load_query_set
from pipeline import (concept_vectors_from_csr, concept_vectors_from_texts, docs_to_device, encode_corpus_e2e,
                      free_cuda, load_docs, report_and_save, splade_query_vectors)
from retrieval import dense_search, project_rows
from utils import get_device, log, log_run_header, make_transformers_cpu_safe, timer

make_transformers_cpu_safe()
from transformers import AutoTokenizer  # noqa: E402


def concept_table(W, tokenizer, n_top, act=None):
    """Rows 'concept  mean_activation  top_positive_tokens  top_negative_tokens'."""
    W = W.float().cpu()
    lines = ["concept\tmean_activation\ttop_positive_tokens\ttop_negative_tokens"]
    for j in range(W.shape[1]):
        col = W[:, j]
        pos = torch.topk(col, n_top).indices.tolist()
        neg = torch.topk(-col, n_top).indices.tolist()
        pt = " ".join(f"{t}({col[i]:.2f})" for t, i in zip(tokenizer.convert_ids_to_tokens(pos), pos))
        nt = " ".join(f"{t}({col[i]:.2f})" for t, i in zip(tokenizer.convert_ids_to_tokens(neg), neg))
        a = f"{act[j]:.4f}" if act is not None else ""
        lines.append(f"{j}\t{a}\t{pt}\t{nt}")
    return lines


def concept_label(W, tokenizer, j, n=5):
    idx = torch.topk(W[:, j].float(), n).indices.tolist()
    return "/".join(tokenizer.convert_ids_to_tokens(idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run name (uses .../best) or path")
    ap.add_argument("--query-sets", nargs="+", default=["dev"], choices=["dev", "dl19", "dl20"])
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--name", default=None, help="method name in the results")
    ap.add_argument("--explain", type=int, default=0, help="number of dev queries to explain")
    ap.add_argument("--concepts-top", type=int, default=15)
    ap.add_argument("--q-max-len", type=int, default=64)
    ap.add_argument("--d-max-len", type=int, default=256)
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    dev = get_device()
    log_run_header("08_eval_concept_model.py", args, dev)

    ckpt = resolve_ckpt(args.ckpt, paths)
    head, info = load_checkpoint(ckpt, dev)
    head.eval()
    e2e = info["mode"] == "e2e"
    name = args.name or f"cm_{ckpt.parent.name}_{ckpt.name}"
    docs = load_docs(paths, mmap=e2e)
    doc_pids = np.asarray(docs.ids, dtype=np.int64)

    encoder = None
    if e2e:
        from splade_encoder import SpladeEncoder
        encoder = SpladeEncoder(info["backbone_path"]).to(dev)
        tokenizer = encoder.tokenizer
        collection = Collection(paths.collection_tsv)
        D = encode_corpus_e2e(head, encoder, collection, doc_pids, paths.embeddings / f"{ckpt.parent.name}_{ckpt.name}",
                              dev, max_length=args.d_max_len, tag=str(ckpt)).to(dev)
    else:
        tokenizer = AutoTokenizer.from_pretrained(get_splade_model(paths)[0])
        tdocs = docs_to_device(docs, dev)
        with timer("g(d) = W . SPLADE(d) for all documents"):
            D = project_rows(tdocs, head.weight.detach())
        tdocs.free()
        del tdocs
        collection = Collection(paths.collection_tsv) if paths.collection_tsv.exists() else None
    free_cuda()

    dev_cache = {}
    for set_name in args.query_sets:
        if e2e:
            texts, qrels, thr = load_query_set(set_name, paths, need_texts=True)
            qids = np.array(sorted(texts), dtype=np.int64)
            gq = concept_vectors_from_texts(head, encoder, [texts[q] for q in qids], dev, args.q_max_len)
        else:
            q, qrels, thr, texts = splade_query_vectors(set_name, paths, dev)
            qids = np.asarray(q.ids)
            gq = concept_vectors_from_csr(head, q, dev)
        with timer(f"exact dense search on '{set_name}' ({len(qids):,} queries)"):
            scores, rows = dense_search(D, gq, args.k)
        report_and_save(paths, name, set_name, qids, rows, scores, doc_pids, qrels, thr,
                        extra={"k_concepts": head.k, "mode": info["mode"], "ckpt": str(ckpt)})
        if set_name == "dev":
            dev_cache = {"qids": qids, "gq": gq, "rows": rows, "qrels": qrels, "texts": texts}

    # ---------------------------------------------------------------- concepts
    act = dev_cache["gq"].mean(0).cpu().numpy() if dev_cache else None
    lines = concept_table(head.weight.detach(), tokenizer, args.concepts_top, act)
    out = paths.results / f"concepts_{name}.tsv"
    out.write_text("\n".join(lines) + "\n")
    log(f"concepts saved to {out}. First 5:")
    for ln in lines[1:6]:
        log("  " + ln[:200])

    # ---------------------------------------------------------------- explanations + intervention
    if args.explain and dev_cache:
        W = head.weight.detach()
        rep = []
        qids, gq, rows, qrels, texts = (dev_cache[k] for k in ["qids", "gq", "rows", "qrels", "texts"])
        shown = 0
        for i, qid in enumerate(qids.tolist()):
            rel = {p for p, r in qrels.get(qid, {}).items() if r >= 1}
            ranked = doc_pids[rows[i]]
            hit = np.flatnonzero(np.isin(ranked[:100], list(rel)))
            if not len(hit):
                continue
            r_rel = int(hit[0])
            g = gq[i]
            d_top = D[rows[i, 0]].float()
            contrib = g * d_top
            top_c = torch.topk(contrib, 5).indices.tolist()
            rep.append("=" * 80)
            rep.append(f"query {qid}: {texts.get(qid, '') if texts else ''}")
            act_top = torch.topk(g, 5).indices.tolist()
            rep.append("  most active concepts in the query: " +
                       "; ".join(f"#{j} {concept_label(W, tokenizer, j)} ({g[j]:.2f})" for j in act_top))
            txt = collection.get(int(ranked[0]))[:200] if collection is not None else ""
            rep.append(f"  top-1 retrieved document pid={ranked[0]} (score {float(g @ d_top):.3f}) {txt}")
            rep.append("  concepts driving the match: " +
                       "; ".join(f"#{j} {concept_label(W, tokenizer, j)} (+{contrib[j]:.2f})" for j in top_c))
            # intervention: zero out the concept with the largest contribution and re-rank the top 100
            j_star = top_c[0]
            g2 = g.clone()
            g2[j_star] = 0.0
            cand = D[torch.from_numpy(rows[i, :100]).to(D.device)].float()
            new_order = torch.argsort(cand @ g2, descending=True).cpu().numpy()
            new_rank_rel = int(np.flatnonzero(new_order == r_rel)[0])
            rep.append(f"  INTERVENTION: zeroing concept #{j_star} moves the relevant passage "
                       f"from rank {r_rel + 1} to {new_rank_rel + 1} (within the top 100)")
            shown += 1
            if shown >= args.explain:
                break
        out = paths.results / f"explain_{name}.txt"
        out.write_text("\n".join(rep) + "\n")
        log(f"explanations saved to {out}")
        for ln in rep[: 6 * min(2, shown)]:
            log(ln[:250])
    log("Done.")


if __name__ == "__main__":
    main()
