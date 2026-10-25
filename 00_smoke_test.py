"""
00_smoke_test.py — Tests ALL the code on small SYNTHETIC data, without touching the real data.

Run it first on the cluster: if it ends with "ALL TESTS PASSED", the environment (lce) and the
code work. It runs on CPU in a few minutes and writes only to a temporary directory.

    srun -p long -c 4 --mem=16G -t 00:30:00 python -u 00_smoke_test.py

Part 1 - unit tests:
  Seismic format (write/read), CSR, exact sparse and dense search vs brute force,
  metrics (vs hand computation and vs ranx if installed), InfoNCE = Eq. (6), false-negative
  mask, THE NUMERICAL EXAMPLE OF SEC. 3.4 OF THE PAPER (LCE), a Concept Model that actually
  learns on data with known structure.
Part 2 - end-to-end test: builds a fake "MS MARCO" (texts, queries, qrels, provided files in
  Seismic format, a random mini-BERT) and runs scripts 01-09 one after the other.
"""
import gzip
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from utils import make_transformers_cpu_safe  # noqa: E402

make_transformers_cpu_safe()

PASSED = []


def ok(name):
    PASSED.append(name)
    print(f"  [OK] {name}", flush=True)


# =====================================================================
#  PART 1 — unit tests
# =====================================================================
def test_seismic_and_csr(tmp):
    from sparse_io import CSR, read_seismic_bin, write_seismic_bin
    rng = np.random.default_rng(0)
    dense = (rng.random((50, 40)) < 0.15) * rng.random((50, 40)).astype(np.float32)
    dense[7] = 0  # empty row
    csr = CSR.from_dense(dense)
    write_seismic_bin(tmp / "x.bin", csr)
    back = read_seismic_bin(tmp / "x.bin", verbose=False)
    back.n_cols = 40
    assert np.allclose(back.to_dense(), dense), "roundtrip Seismic"
    part = read_seismic_bin(tmp / "x.bin", max_vectors=10, verbose=False)
    part.n_cols = 40
    assert np.allclose(part.to_dense(), dense[:10])
    rows = np.array([3, 7, 0, 49, 3])
    assert np.allclose(csr.subset(rows).to_dense(), dense[rows]), "subset/gather"
    csr.ids = np.arange(50) + 1000
    csr.save(tmp / "csr")
    again = CSR.load(tmp / "csr", mmap=True)
    assert np.allclose(again.to_dense(), dense) and again.ids[3] == 1003
    ok("Seismic format + CSR (write, read, subsets, save)")


def test_remap():
    from sparse_io import CSR, build_remap, remap_columns
    tokens = ["apple", "zebra", "##ing", "unknowntoken"]
    vocab = {"[PAD]": 0, "zebra": 5, "apple": 9, "##ing": 2}
    remap, unknown = build_remap(tokens, vocab)
    assert remap.tolist() == [9, 5, 2, -1] and unknown == ["unknowntoken"]
    csr = CSR.from_dense(np.array([[1, 0, 2, 3], [0, 4, 0, 0]], dtype=np.float32))
    out = remap_columns(csr, remap, 10)
    d = out.to_dense()
    assert d[0, 9] == 1 and d[0, 2] == 2 and d[1, 5] == 4 and d.sum() == 7
    ok("Seismic id -> BERT id conversion")


def test_search():
    from retrieval import TorchCSR, dense_search, project_rows, sparse_search
    from sparse_io import CSR
    rng = np.random.default_rng(1)
    D = (rng.random((300, 60)) < 0.2) * rng.random((300, 60))
    Q = (rng.random((17, 60)) < 0.3) * rng.random((17, 60))
    docs, qs = CSR.from_dense(D), CSR.from_dense(Q)
    td = TorchCSR(docs, torch.device("cpu"))
    S = Q @ D.T
    s, i = sparse_search(td, qs, 25, q_batch=5, doc_chunk=40, verbose=False)
    ref = np.sort(S, axis=1)[:, ::-1][:, :25]
    assert np.allclose(s, ref, atol=1e-5), "sparse search scores"
    assert np.allclose(np.take_along_axis(S, i, 1), s, atol=1e-5), "sparse search indices"
    W = torch.randn(60, 8)
    G = project_rows(td, W, chunk=33, out_dtype=torch.float32)
    assert torch.allclose(G, torch.from_numpy(D).float() @ W, atol=1e-4), "g(d) = W.SPLADE(d)"
    gq = torch.from_numpy(Q).float() @ W
    s2, i2 = dense_search(G, gq, 10, q_batch=4, doc_chunk=70, verbose=False)
    S2 = (gq @ G.T).numpy()
    assert np.allclose(s2, np.sort(S2, 1)[:, ::-1][:, :10], atol=1e-4), "dense search"
    ok("exact sparse and dense search = brute force (also chunked)")


def test_metrics():
    from metrics import evaluate_run
    qrels = {1: {10: 1}, 2: {20: 1, 21: 1}, 3: {30: 3, 31: 1, 32: 2}}
    run = np.array([[5, 10, 6, 7], [20, 9, 8, 21], [31, 30, 4, 32]])
    m, _ = evaluate_run([1, 2, 3], run, qrels, rel_threshold=1, cutoffs_recall=(2, 4), ndcg_cut=3, mrr_cut=3)
    assert abs(m["MRR@3"] - (0.5 + 1 + 1) / 3) < 1e-9
    assert abs(m["R@2"] - (1 + 0.5 + 2 / 3) / 3) < 1e-9
    d = 1 / np.log2(np.arange(2, 6))
    nd1 = d[1] / d[0]
    nd2 = d[0] / (d[0] + d[1])
    nd3 = (1 * d[0] + 3 * d[1]) / (3 * d[0] + 2 * d[1] + 1 * d[2])
    assert abs(m["nDCG@3"] - (nd1 + nd2 + nd3) / 3) < 1e-9
    try:
        from ranx import Qrels, Run, evaluate
        qr = Qrels({str(q): {str(p): r for p, r in v.items()} for q, v in qrels.items()})
        rn = Run({str(q): {str(p): float(10 - j) for j, p in enumerate(row)} for q, row in zip([1, 2, 3], run)})
        rx = evaluate(qr, rn, ["mrr@3", "ndcg@3", "recall@2"])
        assert abs(rx["mrr@3"] - m["MRR@3"]) < 1e-6 and abs(rx["recall@2"] - m["R@2"]) < 1e-6
        assert abs(rx["ndcg@3"] - m["nDCG@3"]) < 1e-6, (rx["ndcg@3"], m["nDCG@3"])
        ok("metrics (MRR, graded nDCG, Recall) = hand computation = ranx")
    except ImportError:
        ok("metrics (MRR, graded nDCG, Recall) = hand computation (ranx not installed)")


def test_lce_paper_example():
    """Reproduces the numbers of Sec. 3.4 of the paper (6-document corpus, alpha=0.5, tau=0.5)."""
    from lce import build_cooccurrence, expand, normalize_cooccurrence, prune
    from sparse_io import CSR
    docs_feats = [[1, 5, 7], [6, 8, 10], [1, 2, 3], [4, 5, 9], [6, 8, 11], [5, 6]]  # F1..F11
    X = np.zeros((6, 11), dtype=np.float32)
    for r, fs in enumerate(docs_feats):
        X[r, [f - 1 for f in fs]] = 1.0
    M, df = build_cooccurrence(CSR.from_dense(X), torch.device("cpu"), chunk=4)
    assert M[5, 7].item() == 2 and M[0, 4].item() == 1 and M[4, 5].item() == 1  # M[F6,F8]=2 etc.
    B = normalize_cooccurrence(M, df, "raw", 6)
    assert float(torch.diagonal(B).abs().sum()) == 0.0
    vq = torch.zeros(1, 11)
    vq[0, 0], vq[0, 3] = 0.94, 0.82
    vext = expand(vq, B, 0.5, "absolute")
    expected = [0.94, 0.47, 0.47, 0.82, 0.88, 0, 0.47, 0, 0.41, 0, 0]
    assert torch.allclose(vext[0], torch.tensor(expected), atol=1e-6), vext
    pruned = prune(vext, "tau:0.5")
    assert torch.allclose(pruned[0], torch.tensor([0.94, 0, 0, 0.82, 0.88, 0, 0, 0, 0, 0, 0]), atol=1e-6)
    d6 = torch.zeros(11)
    d6[4], d6[5] = 0.85, 0.91
    assert abs(float(vq[0] @ d6)) < 1e-9 and abs(float(pruned[0] @ d6) - 0.748) < 1e-6
    for kind in ["cond", "cosine", "ppmi"]:
        Bn = normalize_cooccurrence(M, df, kind, 6)
        assert torch.isfinite(Bn).all() and (Bn >= 0).all()
    top2 = prune(vext, "topk:2")
    assert int((top2 > 0).sum()) == 2
    ok("LCE: reproduced EXACTLY the example of Sec. 3.4 (V_q_ext, pruning, score 0.748)")


def test_infonce_and_mask():
    from concept_model import ConceptHead, false_negative_mask, info_nce_loss
    torch.manual_seed(0)
    q, d = torch.randn(3, 4), torch.randn(5, 4)
    labels = torch.tensor([0, 1, 2])
    loss, _, _ = info_nce_loss(q, d, labels, temperature=0.5)
    s = (q @ d.T) / 0.5
    manual = -np.mean([float(s[i, i] - torch.logsumexp(s[i], 0)) for i in range(3)])
    assert abs(float(loss) - manual) < 1e-5, "Eq. (6)"
    mask = false_negative_mask([[100, 105], [101], [102]], np.array([100, 101, 102, 105, 100]), np.array([0, 1, 2]))
    expected = np.zeros((3, 5), dtype=bool)
    expected[0, 3] = expected[0, 4] = True
    assert (mask == expected).all(), mask
    head = ConceptHead(30, 6)
    x = torch.zeros(2, 30)
    x[0, [1, 5]] = torch.tensor([0.5, 2.0])
    x[1, 7] = 1.0
    bags = head.forward_bags(torch.tensor([1, 5, 7]), torch.tensor([0, 2]), torch.tensor([0.5, 2.0, 1.0]))
    assert torch.allclose(bags, head.forward_dense(x), atol=1e-6)
    ok("InfoNCE = Eq. (6), false-negative mask, sparse g(x) = dense")


def test_concept_model_learns():
    """Data with known structure: 12 'topics', queries and documents use different words of the
    same topic (vocabulary mismatch). The lexical dot product fails; the Concept Model trained
    with InfoNCE must learn to connect them."""
    from concept_model import ConceptHead, csr_batch_to_torch, info_nce_loss
    from sparse_io import CSR
    rng = np.random.default_rng(0)
    T, per, V = 12, 10, 12 * 10
    def vec(topic, part):
        v = np.zeros(V, dtype=np.float32)
        words = topic * per + (np.arange(5) if part == "q" else np.arange(5, 10))
        v[rng.choice(words, 3, replace=False)] = rng.random(3) + 0.5
        return v
    n = 600
    topics = rng.integers(0, T, n)
    Q = CSR.from_dense(np.stack([vec(t, "q") for t in topics]))
    D = CSR.from_dense(np.stack([vec(t, "d") for t in topics]))
    dev = torch.device("cpu")
    head = ConceptHead(V, 16, seed=0)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    tr, va = np.arange(500), np.arange(500, 600)

    def val_mrr():
        with torch.no_grad():
            gq = head.forward_bags(*csr_batch_to_torch(Q, va, dev))
            gd = head.forward_bags(*csr_batch_to_torch(D, va, dev))
            S = gq @ gd.T
            ranks = (S > S.diag()[:, None]).sum(1) + 1
            hit10 = ranks <= 10
            return float(torch.where(hit10, 1.0 / ranks, torch.zeros_like(ranks, dtype=torch.float)).mean())

    before = val_mrr()
    for _ in range(150):
        b = rng.choice(tr, 64, replace=False)
        loss, _, _ = info_nce_loss(head.forward_bags(*csr_batch_to_torch(Q, b, dev)),
                                   head.forward_bags(*csr_batch_to_torch(D, b, dev)), torch.arange(64))
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = val_mrr()
    assert after > before + 0.1, (before, after)
    ok(f"the Concept Model learns to overcome the vocabulary mismatch (MRR {before:.3f} -> {after:.3f})")


# =====================================================================
#  PART 2 — synthetic world + scripts 01..09
# =====================================================================
def build_world(root):
    from transformers import BertConfig, BertForMaskedLM, BertTokenizer
    from sparse_io import write_seismic_bin
    from splade_encoder import SpladeEncoder
    rng = np.random.default_rng(42)
    words = [f"w{i}" for i in range(120)] + ["chest", "pain", "heart", "sweat", "pressure", "crushing"]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    mdir = root / "model"
    mdir.mkdir()
    (mdir / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tok = BertTokenizer.from_pretrained(str(mdir))
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=128)
    BertForMaskedLM(cfg).save_pretrained(str(mdir))
    tok.save_pretrained(str(mdir))

    # texts: 10 topics x 12 words
    def text(topic, n):
        return " ".join(rng.choice(words[topic * 12:(topic + 1) * 12], n).tolist()
                        + rng.choice(words, 2).tolist())
    N = 400
    doc_topic = rng.integers(0, 10, N)
    collection = [text(t, 12) for t in doc_topic]
    msm = root / "msm_src"
    msm.mkdir()
    (msm / "collection.tsv").write_text("".join(f"{i}\t{t}\n" for i, t in enumerate(collection)))
    def make_queries(n, start):
        qids = np.arange(start, start + n)
        pos = rng.choice(N, n, replace=False)
        texts = {int(q): text(int(doc_topic[p]), 3) for q, p in zip(qids, pos)}
        qrels = {int(q): {int(p): 1} for q, p in zip(qids, pos)}
        return texts, qrels
    tr_t, tr_r = make_queries(200, 1000)
    dv_t, dv_r = make_queries(40, 5000)
    dl_t, dl_r = make_queries(8, 9000)
    for q in dl_r:  # TREC DL-style graded relevance
        extra = rng.choice(N, 3, replace=False)
        dl_r[q].update({int(e): int(g) for e, g in zip(extra, [3, 2, 1])})
    def wq(path, t):
        Path(path).write_text("".join(f"{q}\t{s}\n" for q, s in t.items()))
    def wr(path, r):
        Path(path).write_text("".join(f"{q}\t0\t{p}\t{v}\n" for q, d in r.items() for p, v in d.items()))
    wq(msm / "queries.train.tsv", tr_t)
    wr(msm / "qrels.train.tsv", tr_r)
    wq(msm / "queries.dev.small.tsv", dv_t)
    wr(msm / "qrels.dev.small.tsv", dv_r)
    with tarfile.open(root / "collectionandqueries.tar.gz", "w:gz") as tar:
        for f in msm.iterdir():
            tar.add(f, arcname=f.name)
    dl_dir = root / "dl"
    dl_dir.mkdir()
    with gzip.open(dl_dir / "msmarco-test2019-queries.tsv.gz", "wt") as f:
        f.write("".join(f"{q}\t{s}\n" for q, s in dl_t.items()))
    (dl_dir / "2019qrels-pass.txt").write_text("".join(f"{q} Q0 {p} {v}\n" for q, d in dl_r.items()
                                                       for p, v in d.items()))

    # "provided files": SPLADE vectors (from the mini-BERT) in the Seismic space, documents in permuted order
    enc = SpladeEncoder(str(mdir))
    dev = torch.device("cpu")
    perm = rng.permutation(N)
    dcsr = enc.encode_to_csr([collection[i] for i in perm], dev, batch_size=64, verbose=False)
    dq = sorted(dv_t)
    qcsr = enc.encode_to_csr([dv_t[q] for q in dq], dev, batch_size=64, verbose=False)
    used = np.union1d(np.unique(dcsr.indices), np.unique(qcsr.indices))
    id2tok = tok.convert_ids_to_tokens(used.tolist())
    sorted_toks = sorted(id2tok)
    mapping = {t: i for i, t in enumerate(sorted_toks)}
    bert2fm = np.full(len(vocab), -1, dtype=np.int64)
    for b, t in zip(used.tolist(), id2tok):
        bert2fm[b] = mapping[t]
    fm = root / "fm"
    fm.mkdir()
    for name, c in [("documents.bin", dcsr), ("queries.bin", qcsr)]:
        c2 = type(c)(c.indptr, bert2fm[c.indices].astype(np.int32), c.data, len(sorted_toks))
        write_seismic_bin(fm / name, c2)
    np.save(fm / "doc_ids.npy", perm.astype(np.int64))
    np.save(fm / "queries_ids.npy", np.array(dq, dtype=np.int64))
    (fm / "token_to_id_mapping.json").write_text(json.dumps(mapping))
    S = qcsr.to_dense() @ dcsr.to_dense().T
    with open(fm / "groundtruth_100.tsv", "w") as f:
        for qi in range(S.shape[0]):
            top = np.argsort(-S[qi], kind="stable")[:100]
            for r, di in enumerate(top):
                f.write(f"{qi}\t{di}\t{r}\t{S[qi, di]}\n")
    return mdir, fm, dl_dir


def run(script, *args, env):
    cmd = [sys.executable, "-u", str(HERE / script), *map(str, args)]
    print(f"\n$ {script} {' '.join(map(str, args))}", flush=True)
    p = subprocess.run(cmd, env=env, cwd=str(HERE), capture_output=True, text=True)
    tail = "\n".join(p.stdout.strip().splitlines()[-6:])
    print("    " + tail.replace("\n", "\n    "), flush=True)
    if p.returncode != 0:
        print(p.stdout[-4000:])
        print(p.stderr[-4000:])
        raise SystemExit(f"FAILED: {script}")
    return p.stdout


def end_to_end(root):
    mdir, fm, dl_dir = build_world(root)
    work = root / "work"
    env = dict(os.environ, FM_DIR=str(fm), WORK_DIR=str(work), SPLADE_MODEL=str(mdir),
               CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
    run("01_prepare_data.py", env=env)
    run("02_download_msmarco.py", "--from-tar", root / "collectionandqueries.tar.gz", env=env)
    for f in dl_dir.iterdir():
        shutil.copy(f, work / "msmarco" / f.name)
    out = run("03_check_encoder.py", "--models", mdir, "--n", 20, env=env)
    assert "mean cosine 1.0000" in out, "the model that created the data must give cosine 1"
    run("04_baseline_splade.py", "--query-sets", "dev", "dl19", "--k", 50, env=env)
    res = json.loads((work / "results" / "splade_baseline_dev.json").read_text())["summary"]
    assert res["overlap_groundtruth"] > 0.99, res
    ok(f"baseline: overlap with the provided groundtruth = {res['overlap_groundtruth']:.3f}")
    run("05_encode_train_queries.py", "--batch-size", 32, env=env)
    common = ["--k", 16, "--batch-size", 16, "--val-queries", 20, "--val-candidates", 20, "--log-every", 5]
    run("06_train_concept_model.py", "--mode", "frozen", "--epochs", 3, "--val-every", 10,
        "--run-name", "fz1", *common, env=env)
    run("07_mine_negatives.py", "--miner", "splade", "--out", "sp", "--depth", 30, "--keep", 20, env=env)
    run("07_mine_negatives.py", "--miner", "concept", "--ckpt", "fz1", "--out", "r1", "--depth", 30,
        "--keep", 20, env=env)
    run("06_train_concept_model.py", "--mode", "frozen", "--max-steps", 12, "--val-every", 6, "--n-hard", 3,
        "--negatives", "sp", "r1", "--neg-rank-min", 1, "--init-from", "fz1", "--run-name", "fz2", *common, env=env)
    run("08_eval_concept_model.py", "--ckpt", "fz2", "--query-sets", "dev", "dl19", "--k", 50, "--explain", 2,
        env=env)
    run("06_train_concept_model.py", "--mode", "e2e", "--max-steps", 4, "--val-every", 2, "--n-hard", 1,
        "--negatives", "r1", "--run-name", "e2e1", "--k", 16, "--batch-size", 4, "--val-queries", 5,
        "--val-candidates", 10, "--log-every", 2, "--grad-checkpointing", env=env)
    run("07_mine_negatives.py", "--miner", "concept", "--ckpt", "e2e1", "--out", "e2e_r1", "--depth", 20,
        "--keep", 10, env=env)
    run("08_eval_concept_model.py", "--ckpt", "e2e1", "--query-sets", "dev", "dl19", "--k", 50, "--explain", 1,
        env=env)
    run("09_lce_splade.py", "--k", 50, "--norms", "raw", "cond", "ppmi", "--alphas", 0.3, "--prunes", "topk:8",
        "tau:0.1", "--demo", "crushing pressure chest sweat", "--examples", 1, "--query-sets", "dev", "dl19", env=env)
    summary = (work / "results" / "summary.csv").read_text().splitlines()
    assert len(summary) > 10
    ok(f"end-to-end: scripts 01-09 completed, {len(summary) - 1} rows in results/summary.csv")


def main():
    print("PART 1: unit tests", flush=True)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        test_seismic_and_csr(tmp)
        test_remap()
        test_search()
        test_metrics()
        test_lce_paper_example()
        test_infonce_and_mask()
        test_concept_model_learns()
    print("\nPART 2: full pipeline on synthetic data", flush=True)
    keep = "--keep" in sys.argv
    root = Path(tempfile.mkdtemp(prefix="splade_smoke_"))
    try:
        end_to_end(root)
    finally:
        if keep:
            print(f"(synthetic files left in {root})")
        else:
            shutil.rmtree(root, ignore_errors=True)
    print(f"\nALL TESTS PASSED ({len(PASSED)})", flush=True)


if __name__ == "__main__":
    main()
