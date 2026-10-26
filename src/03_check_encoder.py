"""
03_check_encoder.py — Which SPLADE model produced the provided vectors?

To train the Concept Model we need SPLADE vectors for the TRAINING queries, which are not
part of the provided data: we compute them ourselves. But they must be produced with THE SAME
model as the documents, otherwise queries and documents "speak different languages".

Test: take the dev queries (we have their text), encode them with the candidate models and
compare them with the provided vectors in queries.bin. The same model gives cosine ~1.0.
The choice is saved to WORK_DIR/encoder_check.json and used by the other scripts.

    srun -p long -c 4 --mem=32G --gpus 1 -t 00:30:00 python -u 03_check_encoder.py
"""
import argparse

import numpy as np

from config import CANDIDATE_SPLADE_MODELS, Paths
from msmarco import read_queries
from sparse_io import CSR
from splade_encoder import SpladeEncoder, top_tokens
from utils import get_device, log, log_run_header, save_json


def compare(ours: CSR, theirs: CSR):
    """Row-by-row cosine, Jaccard of the supports, and ratio of the norms."""
    cos, jac, ratio = [], [], []
    for i in range(ours.n_rows):
        ia, va = ours.row(i)
        ib, vb = theirs.row(i)
        common, xa, xb = np.intersect1d(ia, ib, return_indices=True)
        dot = float((va[xa] * vb[xb]).sum())
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        cos.append(dot / (na * nb) if na > 0 and nb > 0 else 0.0)
        union = len(np.union1d(ia, ib))
        jac.append(len(common) / union if union else 1.0)
        ratio.append(nb / na if na > 0 else np.nan)
    return np.array(cos), np.array(jac), np.array(ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=CANDIDATE_SPLADE_MODELS)
    ap.add_argument("--n", type=int, default=300, help="number of dev queries to compare")
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()
    paths = Paths()
    dev = get_device()
    log_run_header("03_check_encoder.py", args, dev)

    fm_q = CSR.load(paths.dev_queries_csr)
    texts = read_queries(paths.queries_tsv("dev"))
    qids = np.asarray(fm_q.ids)
    rows = [i for i, q in enumerate(qids.tolist()) if q in texts][: args.n]
    if not rows:
        raise SystemExit("[ERROR] None of the provided queries has a text in queries.dev.small.tsv: "
                         "are the provided queries not the dev small set? Check the output of 01.")
    log(f"Comparing {len(rows)} queries (provided qids found among dev small: "
        f"{sum(q in texts for q in qids.tolist())}/{len(qids)})")
    theirs = fm_q.subset(rows)
    batch_texts = [texts[int(qids[i])] for i in rows]

    results = {}
    for name in args.models:
        try:
            enc = SpladeEncoder(name).to(dev)
        except Exception as e:  # noqa: BLE001
            log(f"{name}: could not be loaded ({e})")
            continue
        ours = enc.encode_to_csr(batch_texts, dev, batch_size=64, max_length=args.max_length, verbose=False)
        cos, jac, ratio = compare(ours, theirs)
        results[name] = {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
                         "jaccard_mean": float(jac.mean()), "scale_median": float(np.nanmedian(ratio))}
        log(f"{name}: mean cosine {cos.mean():.4f} (min {cos.min():.4f}), Jaccard {jac.mean():.3f}, "
            f"scale provided/ours {np.nanmedian(ratio):.3f}")
        # side-by-side example on the first query
        ia, va = ours.row(0)
        ib, vb = theirs.row(0)
        log(f"  query: {batch_texts[0]!r}")
        log(f"  ours    : {top_tokens(enc.tokenizer, ia, va, 8)}")
        log(f"  provided: {top_tokens(enc.tokenizer, ib, vb, 8)}")
        del enc

    if not results:
        raise SystemExit("[ERROR] No model could be loaded (network problem reaching Hugging Face?)")
    best = max(results, key=lambda m: results[m]["cos_mean"])
    info = {"model": best, "scale": results[best]["scale_median"], "results": results}
    save_json(info, paths.encoder_check)
    verdict = "GOOD: same model" if results[best]["cos_mean"] > 0.99 else \
              "WARNING: no model reproduces the provided vectors well (ask which SPLADE checkpoint was used)"
    log(f"Selected: {best}  ->  {verdict}")
    log(f"Saved to {paths.encoder_check}")


if __name__ == "__main__":
    main()
