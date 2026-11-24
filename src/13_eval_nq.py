"""
13_eval_nq.py — Zero-shot evaluation OUT OF DOMAIN on NQ (BEIR) with the SPLADE vectors given by the professor.

Dataset: tuskanny/sparse-nq-splade-cocondenser (2,680,893 passages, 3,452 test queries), already encoded with
naver/splade-cocondenser-ensembledistil, Seismic binary format. Same methods, names and metrics as 10_eval_beir.py:

  --splade              SPLADE baseline: exact sparse search on the professor's vectors
  --ckpts RUN ...       frozen / rand : g(x) = W . SPLADE(x) applied to the professor's vectors
                                        (score = d . (W g(q)): the dense documents are never built, as in 08b)
                        e2e           : g(x) = W . SPLADE_theta(x) with the fine-tuned BERT (see below), as in 08c
  --ft-splade RUN ...   e2e checkpoints: the fine-tuned encoder used as a PLAIN SPLADE (W thrown away):
                        SPLADE_theta(q) . SPLADE_theta(d)                         -> ftsplade_<run>_<ckpt>
  --hybrid RUN:VARIANT:LAMBDA ...   hybrids of 11_eval_hybrid.py with LAMBDA FIXED (the lambda* chosen on the MS
                        MARCO validation: nothing is tuned on NQ). VARIANT (frozen runs: only 'two'):
                          two    : (1-lam) SPLADE(q).SPLADE(d) + lam g(q).g(d)                -> hyb_<run>_<ckpt>_lamX
                          now    : (1-lam) SPLADE(q).SPLADE(d) + lam SPLADE_t(q).SPLADE_t(d)  -> hybnoW_...
                          single : (1-lam) SPLADE_t(q).SPLADE_t(d) + lam g(q).g(d)            -> hyb1_...
                        SPLADE = the professor's vectors; SPLADE_t = the fine-tuned encoder (e2e checkpoints).

e2e, --ft-splade and e2e hybrids need the fine-tuned side of the DOCUMENTS: the professor's vectors cannot be
used (the backbone has changed), so the 2.68M passages are RE-ENCODED from the BEIR text ("title + text", as
10_eval_beir.py) in the order of the professor's files, in resumable shards saved in
embeddings/nq_<run>_<ckpt>_sparse/. The encoding is done ONCE per checkpoint (tens of minutes) and reused by every
later run that involves the same checkpoint (e2e, ftsplade, hybrids): the searches then take minutes.
Settings as 08c / 11: documents 256 tokens, queries 64 tokens, bf16 autocast, scale 1.
(On the small BEIR datasets --ft-splade used fp32, scale 0.99995 and 256-token queries: here it uses the same
encoding as the e2e / hybrid terms; NQ queries are short, so the difference is only numerical.)

Expected folder (WORK_DIR/beir/nq by default):
    documents.bin  queries.bin  doc_ids.npy  queries_ids.npy  groundtruth.tsv  qrels/test.tsv
and, only for the fine-tuned side, corpus.jsonl + queries.jsonl (from the BEIR zip WORK_DIR/beir/nq.zip,
see --download-only). BEIR ids ('doc123', 'test45') are mapped to integers = ROW number in the .bin files.
Results go to results/summary.csv with query_set = beir_nq. The parsed vectors are cached in
<dir>/csr_documents and <dir>/csr_queries (the first run reads the 3.3 GB .bin).

    # SPLADE, random W, frozen (a few minutes)
    sbatch --time=01:00:00 -J nq jobs/gpu.sbatch 13_eval_nq.py --splade --ckpts frozen_k1024_rand frozen_k1024_r2_lr3e5
    # frozen hybrid (lambda* chosen on MS MARCO)
    sbatch --time=01:00:00 -J nq_hyb jobs/gpu.sbatch 13_eval_nq.py --hybrid frozen_k1024_r2_lr3e5:two:0.45
    # fine-tuned side: 1) text (once)  2) one job per checkpoint, with everything it needs
    wget -4 -c -O $WORK_DIR/beir/nq.zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
    #   (if that server does not answer: save corpus.parquet and queries.parquet of HF BeIR/nq in beir/nq/ instead)
    python -u src/13_eval_nq.py --download-only
    sbatch --time=04:00:00 -J nq_e2e_k1024 jobs/gpu.sbatch 13_eval_nq.py \
        --ckpts e2e_k1024_r2_is03125_lr3e5_bb5e6_b32_ep1 --ft-splade e2e_k1024_r2_is03125_lr3e5_bb5e6_b32_ep1 \
        --hybrid e2e_k1024_r2_is03125_lr3e5_bb5e6_b32_ep1:two:0.75 e2e_k1024_r2_is03125_lr3e5_bb5e6_b32_ep1:now:0.50
"""
import argparse
import csv
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from concept_model import load_checkpoint
from config import Paths, resolve_ckpt
from pipeline import concept_vectors_from_csr, free_cuda, report_and_save
from retrieval import (TorchCSR, _merge_topk, csr_rows_to_dense, gpu_fits, rows_times_table,
                       search_with_query_table, sparse_search)
from sparse_io import CSR, read_seismic_bin
from utils import get_device, log, log_run_header, make_transformers_cpu_safe, timer

make_transformers_cpu_safe()

HERE = Path(__file__).resolve().parent
VOCAB = 30522          # BERT vocabulary: the professor's ids are BERT ids (checked by decoding the tokens)
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
HYB_PREFIX = {"two": "hyb", "now": "hybnoW", "single": "hyb1"}


# ----------------------------------------------------------------------
#  data: the professor's vectors
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
#  data: the text (only needed for the fine-tuned side)
# ----------------------------------------------------------------------
def ensure_text(nq_dir):
    """corpus.jsonl and queries.jsonl of BEIR NQ in nq_dir (extracted from beir/nq.zip, downloaded if needed).
    Only these two files are extracted: qrels/ and the professor's files are never touched."""
    need = ["corpus.jsonl", "queries.jsonl"]
    if all((nq_dir / f).exists() for f in need):
        return
    # alternative to the BEIR zip: the HF parquet files (BeIR/nq) saved as corpus.parquet / queries.parquet
    if all((nq_dir / f"{f}.parquet").exists() for f in ("corpus", "queries")):
        import pyarrow.parquet as pq
        for f in need:
            src = nq_dir / (f[:-6] + ".parquet")
            with timer(f"converting {src.name} -> {f}"):
                tmp = nq_dir / (f + ".tmp")
                pf = pq.ParquetFile(src)
                cols = [c for c in ("_id", "title", "text") if c in pf.schema_arrow.names]
                with open(tmp, "w", encoding="utf-8") as out:
                    for batch in pf.iter_batches(batch_size=50_000, columns=cols):
                        d = batch.to_pydict()
                        for i in range(len(d["_id"])):
                            out.write(json.dumps({c: d[c][i] for c in cols}, ensure_ascii=False) + "\n")
                tmp.rename(nq_dir / f)
        return
    zpath = nq_dir.parent / "nq.zip"
    if not zpath.exists():
        from msmarco import download
        log(f"{zpath} not found: downloading (if this hangs, use: wget -4 -c -O {zpath} {BEIR_URL.format(name='nq')})")
        download(BEIR_URL.format(name="nq"), zpath)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        for f in need:
            member = [n for n in names if n == f or n.endswith("/" + f)]
            if not member:
                raise SystemExit(f"[ERROR] {f} not found in {zpath} (contents: {names[:10]})")
            with timer(f"extracting {f}"):
                tmp = nq_dir / (f + ".tmp")
                with z.open(member[0]) as src, open(tmp, "wb") as dst:
                    while True:
                        buf = src.read(1 << 24)
                        if not buf:
                            break
                        dst.write(buf)
                tmp.rename(nq_dir / f)


def load_texts(nq_dir):
    """Texts in the ORDER OF THE PROFESSOR'S FILES: (doc_texts, query_texts). Documents are 'title + text'
    (BEIR convention, as 10_eval_beir.py). Stops if any document or query has no text."""
    ensure_text(nq_dir)
    d_ids = np.load(nq_dir / "doc_ids.npy", mmap_mode="r")
    d2r = {}
    for s in range(0, d_ids.shape[0], 200_000):
        for j, d in enumerate(np.asarray(d_ids[s:s + 200_000]).tolist()):
            d2r[d] = s + j
    doc_texts = [None] * d_ids.shape[0]
    with timer("reading corpus.jsonl"):
        with open(nq_dir / "corpus.jsonl", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                r = d2r.get(str(o["_id"]))
                if r is not None:
                    doc_texts[r] = f"{o.get('title', '') or ''} {o.get('text', '') or ''}".strip()
    missing = sum(t is None for t in doc_texts)
    if missing:
        raise SystemExit(f"[ERROR] {missing:,} of {len(doc_texts):,} documents of doc_ids.npy have no text in corpus.jsonl")
    del d2r
    q_ids = np.load(nq_dir / "queries_ids.npy", allow_pickle=True).tolist()
    all_q = {}
    with open(nq_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            all_q[str(o["_id"])] = o.get("text", "") or ""
    miss_q = [q for q in q_ids if str(q) not in all_q]
    if miss_q:
        raise SystemExit(f"[ERROR] {len(miss_q)} queries of queries_ids.npy have no text (e.g. {miss_q[:3]})")
    q_texts = [all_q[str(q)] for q in q_ids]
    log(f"texts aligned to the professor's files: {len(doc_texts):,} documents, {len(q_texts):,} queries")
    log(f"  example query {q_ids[0]}: {q_texts[0]!r}")
    return doc_texts, q_texts


class _Texts:
    """Adapter: 08c.encode_corpus_sparse wants collection.get_many(pids); here pid = row number."""
    def __init__(self, texts):
        self.texts = texts

    def get_many(self, pids):
        return [self.texts[int(p)] for p in pids]


def load_08c():
    """08c_eval_e2e_streaming.py (its name starts with a digit: loaded by path) for encode_corpus_sparse:
    the fine-tuned side of the documents is encoded exactly as on MS MARCO."""
    spec = importlib.util.spec_from_file_location("eval_e2e_streaming_08c", HERE / "08c_eval_e2e_streaming.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def ft_dense_queries(enc, texts, dev, max_length, batch=128):
    """SPLADE_theta(q) as a dense (nq, vocab) float32 matrix (as 10 / 11: amp, query length 64)."""
    was_training = enc.training
    enc.eval()                       # no dropout
    out = [enc.encode_dense(texts[s:s + batch], max_length, dev, amp=True).float()
           for s in range(0, len(texts), batch)]
    enc.train(was_training)
    return torch.cat(out, 0)


# ----------------------------------------------------------------------
#  exact searches (queries enter as dense COLUMNS "tables", shape (vocab, batch))
# ----------------------------------------------------------------------
class ShardSet:
    """Fine-tuned sparse documents saved in shards (CSR, first row), kept on the GPU if they fit."""
    def __init__(self, folders, dev):
        self.dev = dev
        self.shards, off = [], 0
        for f in folders:
            c = CSR.load(f)
            self.shards.append((c, off))
            off += c.n_rows
        self.n_rows = off
        self.nnz = sum(c.nnz for c, _ in self.shards)
        need = sum(c.nnz * 8 + c.n_rows * 8 for c, _ in self.shards)
        self.resident = gpu_fits(need, dev, margin=0.6)
        log(f"fine-tuned sparse documents: {self.n_rows:,} (mean {self.nnz / max(1, self.n_rows):.1f} non-zero "
            f"entries), {need / 1e9:.1f} GB -> {'kept on the GPU' if self.resident else 'streamed from the CPU'}")
        self.gpu = [TorchCSR(c, dev) for c, _ in self.shards] if self.resident else None

    def get(self, j):
        return self.gpu[j] if self.resident else TorchCSR(self.shards[j][0], self.dev)

    def release(self, t):
        if not self.resident:
            t.free()

    def free(self):
        if self.gpu:
            for t in self.gpu:
                t.free()
        self.gpu = None


@torch.no_grad()
def search_prof(tdocs, table_fn, nq, k, q_batch):
    """Exact search over ONE sparse document set (the professor's vectors). table_fn(qs, qe) -> (vocab, qe-qs)."""
    k = min(k, tdocs.n_rows)
    out_s = np.empty((nq, k), dtype=np.float32)
    out_i = np.empty((nq, k), dtype=np.int64)
    for qs in range(0, nq, q_batch):
        qe = min(nq, qs + q_batch)
        s, i = search_with_query_table(tdocs, table_fn(qs, qe).contiguous(), k)
        out_s[qs:qe] = s.float().cpu().numpy()
        out_i[qs:qe] = i.cpu().numpy()
    return out_s, out_i


@torch.no_grad()
def search_shards(sset, table_fn, nq, k, q_batch):
    """Exact search over the fine-tuned document shards (global row = first row of the shard + local row)."""
    k = min(k, sset.n_rows)
    out_s = np.empty((nq, k), dtype=np.float32)
    out_i = np.empty((nq, k), dtype=np.int64)
    for qs in range(0, nq, q_batch):
        qe = min(nq, qs + q_batch)
        table = table_fn(qs, qe).contiguous()
        best_s = best_i = None
        for j, (c, off) in enumerate(sset.shards):
            t = sset.get(j)
            s, i = search_with_query_table(t, table, k)
            best_s, best_i = _merge_topk(best_s, best_i, s, i + off, k, dim=1)
            sset.release(t)
        out_s[qs:qe] = best_s.float().cpu().numpy()
        out_i[qs:qe] = best_i.cpu().numpy()
    return out_s, out_i


@torch.no_grad()
def search_two(tprof, sset, table1_fn, table2_fn, lam, nq, k, q_batch):
    """Exact search of (1-lam) * [professor's vectors . table1] + lam * [fine-tuned documents . table2].
    Row r of the professor's vectors and row r of the shards are the SAME passage, so the two scores of a
    block of documents are summed before taking the top-k."""
    k = min(k, sset.n_rows)
    out_s = np.empty((nq, k), dtype=np.float32)
    out_i = np.empty((nq, k), dtype=np.int64)
    for qs in range(0, nq, q_batch):
        qe = min(nq, qs + q_batch)
        t1 = table1_fn(qs, qe).contiguous()
        t2 = table2_fn(qs, qe).contiguous()
        best_s = best_i = None
        for j, (c, off) in enumerate(sset.shards):
            t = sset.get(j)
            S = (1.0 - lam) * rows_times_table(tprof, off, off + c.n_rows, t1)
            S += lam * rows_times_table(t, 0, c.n_rows, t2)          # (documents of the shard, batch)
            v, ix = torch.topk(S, min(k, c.n_rows), dim=0)
            best_s, best_i = _merge_topk(best_s, best_i, v.T.contiguous(), ix.T + off, k, dim=1)
            sset.release(t)
            del S
        out_s[qs:qe] = best_s.float().cpu().numpy()
        out_i[qs:qe] = best_i.cpu().numpy()
    return out_s, out_i


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


# ----------------------------------------------------------------------
#  main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="folder with the NQ files (default WORK_DIR/beir/nq)")
    ap.add_argument("--ckpts", nargs="*", default=[], help="frozen / rand / e2e run names to evaluate")
    ap.add_argument("--ft-splade", nargs="*", default=[],
                    help="e2e run names: evaluate their fine-tuned encoder as a plain SPLADE (no W)")
    ap.add_argument("--hybrid", nargs="*", default=[],
                    help="RUN:VARIANT:LAMBDA, VARIANT in two/now/single (frozen runs: only two); "
                         "LAMBDA = the lambda* chosen on the MS MARCO validation")
    ap.add_argument("--splade", action="store_true", help="also evaluate the SPLADE baseline")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--q-batch", type=int, default=256)
    ap.add_argument("--hyb-q-batch", type=int, default=128, help="queries per batch for the hybrids with two terms")
    ap.add_argument("--d-max-len", type=int, default=256, help="fine-tuned side: document length (as 08c / 10)")
    ap.add_argument("--q-max-len-e2e", type=int, default=64, help="fine-tuned side: query length (as 08c / 10)")
    ap.add_argument("--encode-batch", type=int, default=256, help="fine-tuned side: documents per encoding batch")
    ap.add_argument("--shard-size", type=int, default=500_000, help="fine-tuned side: documents per saved shard")
    ap.add_argument("--download-only", action="store_true",
                    help="only extract/check the BEIR text for the fine-tuned side (no GPU needed), then stop")
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    nq_dir = Path(args.dir) if args.dir else paths.work / "beir" / "nq"

    if args.download_only:
        load_texts(nq_dir)
        log("Text ready.")
        return

    dev = get_device()
    log_run_header("13_eval_nq.py", args, dev)
    set_name = "beir_nq"
    hybrids = parse_hybrids(args.hybrid)

    docs = load_vectors(nq_dir, "documents")
    q = load_vectors(nq_dir, "queries")
    qrels = load_qrels(nq_dir, docs.n_rows, q.n_rows)
    doc_rows = np.arange(docs.n_rows, dtype=np.int64)       # row -> integer doc id (= the row itself)
    nq = q.n_rows
    qids = np.arange(nq, dtype=np.int64)
    k = min(args.k, docs.n_rows)
    extra0 = {"dataset": "nq", "n_docs": docs.n_rows}

    # checkpoints (small): loaded once
    runs = {}
    for run in list(args.ckpts) + list(args.ft_splade) + [h[0] for h in hybrids]:
        if run not in runs:
            ckpt = resolve_ckpt(run, paths)
            head, info = load_checkpoint(ckpt, dev)
            head.eval()
            runs[run] = (ckpt, head, info)

    def is_e2e(run):
        return runs[run][2]["mode"] == "e2e"

    for run in args.ft_splade:
        if not is_e2e(run) or not runs[run][2].get("backbone_path"):
            raise SystemExit(f"[ERROR] --ft-splade {run}: not an e2e checkpoint with a backbone/ folder")
    for run, variant, lam in hybrids:
        if is_e2e(run) and not runs[run][2].get("backbone_path"):
            raise SystemExit(f"[ERROR] --hybrid {run}: e2e checkpoint without a backbone/ folder")
        if not is_e2e(run) and variant != "two":
            raise SystemExit(f"[ERROR] --hybrid {run}:{variant}: frozen / random-W checkpoints only have the 'two' variant")
    for run in args.ckpts:
        if is_e2e(run) and not runs[run][2].get("backbone_path"):
            log(f"WARNING: {run} is e2e but has no backbone folder: skipped")

    prof_hybrid = any(variant in ("two", "now") for _, variant, _ in hybrids)
    need_prof = (args.splade or prof_hybrid or any(not is_e2e(r) for r in args.ckpts))
    tdocs = TorchCSR(docs, dev) if need_prof else None
    t1_cache = {}

    def prof_table(qs, qe):
        """The professor's SPLADE query vectors as dense columns (vocab, batch)."""
        return csr_rows_to_dense(q, np.arange(qs, qe), dev).T.contiguous()

    def cm_name(run):
        return f"cm_{runs[run][0].parent.name}_{runs[run][0].name}"

    def hyb_name(run, variant, lam):
        c = runs[run][0]
        return f"{HYB_PREFIX[variant]}_{c.parent.name}_{c.name}_lam{lam:.2f}"

    # ---------------- SPLADE baseline ----------------
    if args.splade:
        with timer("exact SPLADE search"):
            scores, rows = sparse_search(tdocs, q, k, verbose=False)
        check_groundtruth(nq_dir, rows, scores)
        report_and_save(paths, "splade_baseline_beir", set_name, qids, rows, scores, doc_rows, qrels, 1,
                        extra=extra0)

    # ---------------- frozen / random W (professor's vectors only) ----------------
    for run in args.ckpts:
        ckpt, head, info = runs[run]
        if is_e2e(run):
            continue
        W = head.weight.detach().float()
        gq = concept_vectors_from_csr(head, q, dev)
        with timer(f"{cm_name(run)}: streaming search"):
            out_s, out_i = search_prof(tdocs, lambda qs, qe: W @ gq[qs:qe].T, nq, k, args.q_batch)
        report_and_save(paths, cm_name(run), set_name, qids, out_i, out_s, doc_rows, qrels, 1,
                        extra={**extra0, "k_concepts": head.k, "mode": info["mode"], "ckpt": str(ckpt),
                               "search": "streaming"})
        del W, gq
        free_cuda()

    # ---------------- hybrids of frozen / random-W checkpoints (professor's vectors only) ----------------
    for run, variant, lam in hybrids:
        if is_e2e(run):
            continue
        ckpt, head, info = runs[run]
        W = head.weight.detach().float()
        gq = concept_vectors_from_csr(head, q, dev)
        name = hyb_name(run, variant, lam)
        with timer(f"{name}: exact search"):
            out_s, out_i = search_prof(
                tdocs, lambda qs, qe: (1.0 - lam) * prof_table(qs, qe) + lam * (W @ gq[qs:qe].T),
                nq, k, args.q_batch)
        report_and_save(paths, name, set_name, qids, out_i, out_s, doc_rows, qrels, 1,
                        extra={**extra0, "k_concepts": head.k, "lambda": lam, "variant": variant,
                               "ckpt": str(ckpt), "search": "hybrid_exact"})
        del W, gq
        free_cuda()

    # ---------------- fine-tuned side: e2e, ftsplade, hybrids with e2e checkpoints ----------------
    e2e_runs = []
    for run in list(args.ckpts) + list(args.ft_splade) + [h[0] for h in hybrids]:
        if is_e2e(run) and runs[run][2].get("backbone_path") and run not in e2e_runs:
            e2e_runs.append(run)
    texts = m08c = None
    for run in e2e_runs:
        ckpt, head, info = runs[run]
        if texts is None:
            texts = load_texts(nq_dir)
            m08c = load_08c()
        doc_texts, q_texts = texts
        from splade_encoder import SpladeEncoder
        enc = SpladeEncoder(info["backbone_path"]).to(dev)
        folders = m08c.encode_corpus_sparse(
            enc, _Texts(doc_texts), doc_rows, paths.embeddings / f"nq_{ckpt.parent.name}_{ckpt.name}_sparse",
            dev, max_length=args.d_max_len, batch=args.encode_batch, shard_size=args.shard_size, tag=str(ckpt))
        with timer(f"{run}: encoding the queries"):
            sq = ft_dense_queries(enc, q_texts, dev, args.q_max_len_e2e)         # SPLADE_theta(q), (nq, vocab)
            gq = head.forward_dense(sq).float()                                   # g(q) = W . SPLADE_theta(q)
        del enc
        free_cuda()
        W = head.weight.detach().float()
        sset = ShardSet(folders, dev)
        extra_ft = {**extra0, "ckpt": str(ckpt), "avg_doc_nnz": float(sset.nnz / max(1, sset.n_rows))}

        def t_cm(qs, qe):
            return W @ gq[qs:qe].T                    # g(q) . g(d) = (W g(q)) . SPLADE_theta(d)

        def t_ft(qs, qe):
            return sq[qs:qe].T                        # SPLADE_theta(q) . SPLADE_theta(d)

        if run in args.ckpts:
            name = cm_name(run)
            with timer(f"{name}: exact streaming search"):
                out_s, out_i = search_shards(sset, t_cm, nq, k, args.q_batch)
            report_and_save(paths, name, set_name, qids, out_i, out_s, doc_rows, qrels, 1,
                            extra={**extra_ft, "k_concepts": head.k, "mode": info["mode"], "search": "streaming"})
        if run in args.ft_splade:
            name = f"ftsplade_{ckpt.parent.name}_{ckpt.name}"
            with timer(f"{name}: exact sparse search"):
                out_s, out_i = search_shards(sset, t_ft, nq, k, args.q_batch)
            report_and_save(paths, name, set_name, qids, out_i, out_s, doc_rows, qrels, 1,
                            extra={**extra_ft, "mode": "ft_splade", "search": "streaming"})
        for r2, variant, lam in hybrids:
            if r2 != run:
                continue
            name = hyb_name(run, variant, lam)
            with timer(f"{name}: exact hybrid search"):
                if variant == "single":      # (1-lam) SPLADE_t . SPLADE_t + lam g . g : one index, one table
                    out_s, out_i = search_shards(
                        sset, lambda qs, qe: (1.0 - lam) * t_ft(qs, qe) + lam * t_cm(qs, qe), nq, k, args.q_batch)
                else:
                    out_s, out_i = search_two(tdocs, sset, prof_table, t_cm if variant == "two" else t_ft,
                                              lam, nq, k, args.hyb_q_batch)
            report_and_save(paths, name, set_name, qids, out_i, out_s, doc_rows, qrels, 1,
                            extra={**extra_ft, "k_concepts": head.k, "lambda": lam, "variant": variant,
                                   "search": "hybrid_exact"})
        sset.free()
        del sset, W, gq, sq
        free_cuda()
    log("Done.")


if __name__ == "__main__":
    main()
