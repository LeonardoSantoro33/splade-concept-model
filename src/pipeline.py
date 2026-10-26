"""
pipeline.py — Functions shared by scripts 04-09 (loading data and query sets, saving results).
"""
import numpy as np
import torch

from config import get_splade_model
from metrics import evaluate_run, format_metrics, write_trec_run
from msmarco import load_query_set
from retrieval import TorchCSR, gpu_fits
from sparse_io import CSR
from utils import append_csv_row, log, now_tag, save_json, timer


def load_docs(paths, mmap=False):
    with timer("loading document SPLADE vectors (cache)"):
        docs = CSR.load(paths.docs_csr, mmap=mmap)
    log(f"documents: {docs.n_rows:,}, non-zero entries: {docs.nnz:,} "
        f"(mean {docs.nnz / max(1, docs.n_rows):.1f} per document)")
    return docs


def docs_to_device(docs, dev):
    need = docs.nnz * 8 + docs.n_rows * 8
    if not gpu_fits(need, dev):
        raise SystemExit(f"[ERROR] The documents ({need / 1e9:.1f} GB) do not fit in the free GPU memory.")
    with timer(f"copying documents to {dev.type}"):
        return TorchCSR(docs, dev)


def splade_query_vectors(set_name, paths, dev, encoder=None, max_length=256):
    """SPLADE vectors (CSR, BERT space) of a query set + qrels + threshold.
    dev       -> provided vectors (queries.bin)
    dl19/dl20 -> encoded by us with the same SPLADE model (chosen by script 03)."""
    if set_name == "dev":
        q = CSR.load(paths.dev_queries_csr)
        try:
            texts, qrels, thr = load_query_set("dev", paths, need_texts=True)
        except SystemExit:
            texts = None
            _, qrels, thr = load_query_set("dev", paths, need_texts=False)
        return q, qrels, thr, texts
    texts, qrels, thr = load_query_set(set_name, paths, need_texts=True)
    model_name, scale = get_splade_model(paths)
    if encoder is None:
        from splade_encoder import SpladeEncoder
        encoder = SpladeEncoder(model_name).to(dev)
    qids = np.array(sorted(texts), dtype=np.int64)
    q = encoder.encode_to_csr([texts[i] for i in qids], dev, batch_size=64, max_length=max_length,
                              ids=qids, scale=scale, verbose=False)
    return q, qrels, thr, texts


def report_and_save(paths, method, set_name, qids, doc_rows, scores, doc_pids, qrels, thr,
                    extra=None, save_run=True):
    """Convert rows->pids, compute the metrics, save the TREC run + json + a row in summary.csv."""
    pids = np.where(doc_rows >= 0, np.asarray(doc_pids)[np.clip(doc_rows, 0, None)], -1)
    metrics, per_query = evaluate_run(qids, pids, qrels, rel_threshold=thr)
    log(f"[{method} | {set_name}] {format_metrics(metrics)}")
    if "warning" in metrics:
        log("WARNING: " + metrics["warning"])
    tag = now_tag()
    if save_run:
        paths.runs.mkdir(parents=True, exist_ok=True)
        write_trec_run(paths.runs / f"{method}_{set_name}.trec", qids, pids, scores, tag=method)
    out = {"method": method, "query_set": set_name, "time": tag, **metrics, **(extra or {})}
    save_json({"summary": out, "per_query": per_query}, paths.results / f"{method}_{set_name}.json")
    append_csv_row(paths.summary_csv, out)
    return metrics, pids


def free_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------
#  Concept Model: g(.) representations of documents and queries
# ----------------------------------------------------------------------
@torch.no_grad()
def concept_vectors_from_csr(head, csr, dev, rows=None, batch=4096):
    """g(x) = W . x for rows of a CSR (frozen variant). Returns a float32 tensor on dev."""
    from concept_model import csr_batch_to_torch
    rows = np.arange(csr.n_rows) if rows is None else np.asarray(rows)
    out = [head.forward_bags(*csr_batch_to_torch(csr, rows[s:s + batch], dev)).float()
           for s in range(0, len(rows), batch)]
    return torch.cat(out) if out else torch.zeros(0, head.k, device=dev)


@torch.no_grad()
def concept_vectors_from_texts(head, encoder, texts, dev, max_length, batch=128, amp=True):
    """g(x) = W . SPLADE_theta(x) for texts (end-to-end variant)."""
    encoder.eval()
    texts = list(texts)
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = torch.empty(len(texts), head.k, dtype=torch.float32, device=dev)
    for s in range(0, len(texts), batch):
        idx = order[s:s + batch]
        x = encoder.encode_dense([texts[i] for i in idx], max_length, dev, amp=amp)
        out[torch.from_numpy(idx).to(dev)] = head.forward_dense(x).float()
    return out


@torch.no_grad()
def encode_corpus_e2e(head, encoder, collection, doc_pids, cache_dir, dev, max_length=256, batch=256,
                      amp=True, shard_size=500_000, tag=""):
    """g(d) for the WHOLE corpus with the fine-tuned BERT (e2e variant). Slow (1-3 hours on an A100):
    saves in pieces (shards) to cache_dir, so if the job is interrupted it resumes where it left off."""
    from pathlib import Path
    from utils import load_json, save_json
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "meta.json"
    meta = {"n_docs": int(len(doc_pids)), "k": int(head.k), "tag": tag, "max_length": max_length}
    if meta_path.exists() and load_json(meta_path) != meta:
        raise SystemExit(f"[ERROR] {cache_dir} contains embeddings from a different model: delete it or use another name")
    save_json(meta, meta_path)
    parts = []
    n = len(doc_pids)
    for si, s in enumerate(range(0, n, shard_size)):
        f = cache_dir / f"shard_{si:04d}.npy"
        e = min(n, s + shard_size)
        if f.exists():
            parts.append(np.load(f))
            continue
        with timer(f"encoding documents {s:,}-{e:,} of {n:,}"):
            texts = collection.get_many(doc_pids[s:e])
            g = concept_vectors_from_texts(head, encoder, texts, dev, max_length, batch, amp)
            arr = g.half().cpu().numpy()
        np.save(f, arr)
        parts.append(arr)
    return torch.from_numpy(np.concatenate(parts))
