"""
08c_eval_e2e_streaming.py — Official evaluation of an END-TO-END checkpoint with the streaming trick
(memory independent of k): needed for k >= 2048, where 08 runs out of GPU memory.

08 builds the dense matrix D = g(d) for all 8.8M documents (18 GB at k=1024, 36 GB at 2048,
72 GB at 4096). Here we use the identity

    g(q) . g(d) = (W g(q)) . SPLADE_theta(d)

so we only need the SPARSE SPLADE vectors of the documents, computed with the FINE-TUNED BERT of
the checkpoint (not the provided ones: in e2e the backbone has changed). Steps:

  1) encode the 8.8M passages with the fine-tuned BERT into sparse vectors (1-3 hours), saved in
     shards to WORK_DIR/embeddings/<run>_<ckpt>_sparse/: if the job is interrupted it resumes;
  2) for each query set: g(q) as in 08, table = W g(q), exact search over the sparse documents.

Scores are identical to 08 up to floating-point rounding (08 stores g(d) in fp16).
The method name in summary.csv ends with _stream (as 08b).

    sbatch --time=10:00:00 -J eval_e2e_stream_k2048 jobs/gpu.sbatch 08c_eval_e2e_streaming.py \
        --ckpt e2e_k2048_r2_is0221_lr3e5_bb5e6_b32_ep1 --query-sets dev dl19 dl20
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from concept_model import load_checkpoint
from config import Paths, resolve_ckpt
from msmarco import Collection, load_query_set
from pipeline import concept_vectors_from_texts, free_cuda, load_docs, report_and_save
from retrieval import TorchCSR, gpu_fits, search_with_query_table, _merge_topk
from sparse_io import CSR
from utils import get_device, log, log_run_header, make_transformers_cpu_safe, timer

make_transformers_cpu_safe()


@torch.no_grad()
def encode_corpus_sparse(encoder, collection, doc_pids, cache_dir, dev, max_length=256, batch=256,
                         shard_size=500_000, amp=True, tag=""):
    """SPLADE_theta(d) (sparse, BERT space) for all documents, saved in resumable shards.
    Returns the list of shard folders (in order)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {"n_docs": int(len(doc_pids)), "max_length": int(max_length), "tag": tag,
            "shard_size": int(shard_size), "amp": bool(amp)}
    meta_path = cache_dir / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise SystemExit(f"[ERROR] {cache_dir} contains vectors from a different model/setting: "
                         "delete it or change checkpoint")
    meta_path.write_text(json.dumps(meta, indent=2))
    folders = []
    n = len(doc_pids)
    for si, s in enumerate(range(0, n, shard_size)):
        e = min(n, s + shard_size)
        folder = cache_dir / f"shard_{si:04d}"
        folders.append(folder)
        if (folder / "meta.json").exists():
            continue
        with timer(f"encoding documents {s:,}-{e:,} of {n:,}"):
            texts = collection.get_many(doc_pids[s:e])
            csr = encoder.encode_to_csr(texts, dev, batch_size=batch, max_length=max_length, amp=amp,
                                        ids=np.asarray(doc_pids[s:e]), verbose=False)
        tmp = folder.with_name(f"{folder.name}.tmp{os.getpid()}")
        csr.save(tmp)
        if folder.exists():
            shutil.rmtree(folder)
        tmp.rename(folder)
        log(f"  shard {si}: mean {csr.nnz / max(1, csr.n_rows):.1f} non-zero entries per document")
    return folders


@torch.no_grad()
def streaming_search(shards, W, gq, k, dev, q_batch=256):
    """Exact search of g(q).g(d) = (W g(q)).SPLADE(d) over sparse document shards.
    shards: list of (CSR, row_offset). If all shards fit in GPU memory they are kept there,
    otherwise each shard is copied to the GPU for every query batch (slower, same result)."""
    need = sum(c.nnz * 8 + c.n_rows * 8 for c, _ in shards)
    resident = gpu_fits(need, dev, margin=0.6)
    log(f"sparse documents: {need / 1e9:.1f} GB -> "
        f"{'kept on the GPU' if resident else 'streamed from the CPU for each query batch'}")
    gpu_shards = [(TorchCSR(c, dev), off) for c, off in shards] if resident else None
    nq = gq.shape[0]
    n_docs = sum(c.n_rows for c, _ in shards)
    k = min(k, n_docs)
    out_s = np.empty((nq, k), dtype=np.float32)
    out_i = np.empty((nq, k), dtype=np.int64)
    W = W.float()
    for b, qs in enumerate(range(0, nq, q_batch)):
        qe = min(nq, qs + q_batch)
        table = (W @ gq[qs:qe].float().T).contiguous()                 # (vocab, nq_batch)
        best_s = best_i = None
        for j, (c, off) in enumerate(shards):
            t = gpu_shards[j][0] if resident else TorchCSR(c, dev)
            s, i = search_with_query_table(t, table, k)
            best_s, best_i = _merge_topk(best_s, best_i, s, i + off, k, dim=1)
            if not resident:
                t.free()
        out_s[qs:qe] = best_s.float().cpu().numpy()
        out_i[qs:qe] = best_i.cpu().numpy()
        if b % 5 == 0:
            log(f"  {qe}/{nq} queries")
    if resident:
        for t, _ in gpu_shards:
            t.free()
    return out_s, out_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="run name (uses .../best) or path; must be e2e")
    ap.add_argument("--query-sets", nargs="+", default=["dev"], choices=["dev", "dl19", "dl20"])
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--name", default=None, help="method name in the results")
    ap.add_argument("--q-batch", type=int, default=256)
    ap.add_argument("--q-max-len", type=int, default=64)
    ap.add_argument("--d-max-len", type=int, default=256)
    ap.add_argument("--encode-batch", type=int, default=256)
    ap.add_argument("--shard-size", type=int, default=500_000, help="documents per saved shard")
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    dev = get_device()
    log_run_header("08c_eval_e2e_streaming.py", args, dev)

    ckpt = resolve_ckpt(args.ckpt, paths)
    head, info = load_checkpoint(ckpt, dev)
    head.eval()
    if info["mode"] != "e2e" or not info.get("backbone_path"):
        raise SystemExit("[ERROR] this script is only for e2e checkpoints (use 08b for frozen)")
    name = args.name or f"cm_{ckpt.parent.name}_{ckpt.name}_stream"

    docs = load_docs(paths, mmap=True)          # only for the list of pids (same order as 08)
    doc_pids = np.asarray(docs.ids, dtype=np.int64)
    del docs

    from splade_encoder import SpladeEncoder
    encoder = SpladeEncoder(info["backbone_path"]).to(dev)
    collection = Collection(paths.collection_tsv)
    folders = encode_corpus_sparse(encoder, collection, doc_pids,
                                   paths.embeddings / f"{ckpt.parent.name}_{ckpt.name}_sparse", dev,
                                   max_length=args.d_max_len, batch=args.encode_batch, shard_size=args.shard_size,
                                   tag=str(ckpt))
    free_cuda()
    with timer("loading the sparse document shards"):
        shards, off = [], 0
        for f in folders:
            c = CSR.load(f)
            shards.append((c, off))
            off += c.n_rows
    nnz = sum(c.nnz for c, _ in shards)
    log(f"documents: {off:,}, non-zero entries: {nnz:,} (mean {nnz / max(1, off):.1f} per document; "
        "the provided SPLADE vectors have ~120)")

    W = head.weight.detach()
    for set_name in args.query_sets:
        texts, qrels, thr = load_query_set(set_name, paths, need_texts=True)
        qids = np.array(sorted(texts), dtype=np.int64)
        gq = concept_vectors_from_texts(head, encoder, [texts[q] for q in qids], dev, args.q_max_len)
        with timer(f"exact streaming search on '{set_name}' ({len(qids):,} queries)"):
            scores, rows = streaming_search(shards, W, gq, args.k, dev, args.q_batch)
        report_and_save(paths, name, set_name, qids, rows, scores, doc_pids, qrels, thr,
                        extra={"k_concepts": head.k, "mode": info["mode"], "ckpt": str(ckpt),
                               "search": "streaming", "avg_doc_nnz": float(nnz / max(1, off))})
    log("Done.")


if __name__ == "__main__":
    main()
