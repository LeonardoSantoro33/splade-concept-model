"""
06_train_concept_model.py — Training of the SPLADE Concept Model (Sec. 5-6 of the paper).

    g(x) = W . SPLADE(x)        score(q,d) = g(q) . g(d)        loss = InfoNCE (Eq. 6)

Negatives:
  - in-batch (Sec. 6.2): ALWAYS active. With B queries in the batch, each query uses the
    positives of the other B-1 queries (and their hard negatives) as negatives. Zero cost.
  - hard negatives (Sec. 6.3): with --negatives FILE.npz --n-hard N. The files are created by
    07_mine_negatives.py (with SPLADE or with an already trained Concept Model).
    Multiple files = multiple "teachers" (Sec. 6.4, DRAGON figure): for each query one of the
    teachers is picked at random and its negatives are used.

Variants (Sec. 5.3):
  --mode frozen : SPLADE fixed, only W is trained. Uses the provided SPLADE vectors. Fast (minutes).
  --mode e2e    : W and BERT are trained. Uses the texts (collection.tsv). Slow (hours).

Validation: 1000 training queries HELD OUT (never seen in training); for each one the
100 SPLADE candidates are re-ranked with g(q).g(d) -> MRR@10. The checkpoint with the
best MRR@10 is saved to models/<run>/best, the last one to models/<run>/last.

Examples:
  # round 1: in-batch negatives only
  sbatch -J cm_r1 jobs/gpu.sbatch 06_train_concept_model.py --mode frozen --k 256 --run-name frozen_k256_r1
  # round 2: + hard negatives mined with the round-1 model, initialized from it
  sbatch -J cm_r2 jobs/gpu.sbatch 06_train_concept_model.py --mode frozen --k 256 --run-name frozen_k256_r2 \
         --negatives r1 --n-hard 7 --init-from frozen_k256_r1
"""
import argparse
import math
import time

import numpy as np
import torch

from concept_model import (ConceptHead, csr_batch_to_torch, false_negative_mask, info_nce_loss,
                           load_checkpoint, save_checkpoint)
from config import Paths, get_splade_model, resolve_ckpt, resolve_negatives
from metrics import evaluate_run
from msmarco import Collection, positives_from_qrels, read_qrels, read_queries
from pipeline import docs_to_device, load_docs
from retrieval import sparse_search
from sparse_io import CSR, ids_to_rows
from utils import (append_csv_row, get_device, log, log_run_header, now_tag, save_json, set_seed,
                   timer)


def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=["frozen", "e2e"], default="frozen")
    ap.add_argument("--k", type=int, default=256, help="number of concepts (rows of W)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=float, default=None, help="default: 10 frozen, 1 e2e")
    ap.add_argument("--max-steps", type=int, default=0, help="if > 0, overrides --epochs")
    ap.add_argument("--batch-size", type=int, default=None, help="queries per batch (default 512 frozen, 32 e2e)")
    ap.add_argument("--n-hard", type=int, default=0, help="hard negatives per query (0 = in-batch only)")
    ap.add_argument("--negatives", nargs="*", default=[], help=".npz files (or names) created by 07_mine_negatives")
    ap.add_argument("--neg-rank-min", type=int, default=0, help="discard the top N candidates (possible false negatives)")
    ap.add_argument("--neg-rank-max", type=int, default=100, help="use candidates up to rank N")
    ap.add_argument("--temperature", type=float, default=1.0, help="beta in Eq. (6)")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate of W")
    ap.add_argument("--lr-backbone", type=float, default=2e-5, help="learning rate of BERT (e2e only)")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--init-std", type=float, default=0.02, help="initial W ~ N(0, init_std^2)")
    ap.add_argument("--init-from", default=None, help="checkpoint to initialize from (later rounds)")
    ap.add_argument("--max-train-queries", type=int, default=0, help="0 = all")
    ap.add_argument("--val-queries", type=int, default=None, help="default 1000 frozen, 300 e2e")
    ap.add_argument("--val-candidates", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=None, help="default 500 frozen, 2000 e2e")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--q-max-len", type=int, default=64)
    ap.add_argument("--d-max-len", type=int, default=256)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="bf16 (e2e on GPU only)")
    ap.add_argument("--grad-checkpointing", action="store_true", help="e2e: less memory, ~30%% slower")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    e2e = args.mode == "e2e"
    args.epochs = args.epochs if args.epochs is not None else (1.0 if e2e else 10.0)
    args.batch_size = args.batch_size or (32 if e2e else 512)
    args.val_queries = args.val_queries if args.val_queries is not None else (300 if e2e else 1000)
    args.val_every = args.val_every or (2000 if e2e else 500)
    args.run_name = args.run_name or f"{args.mode}_k{args.k}_{now_tag()}"
    if args.val_queries < 1 or args.batch_size < 2:
        ap.error("--val-queries must be >= 1 and --batch-size >= 2")
    return args


class Teacher:
    """Hard negatives of a "teacher" (one .npz file), aligned with the training-query rows."""

    def __init__(self, path, qid_to_row, n_rows, rank_min, rank_max):
        z = np.load(path, allow_pickle=False)
        neg = z["neg_pids"][:, rank_min:rank_max].astype(np.int32)
        rows = qid_to_row(z["qids"])
        ok = rows >= 0
        self.pids = np.full((n_rows, neg.shape[1]), -1, dtype=np.int32)
        self.pids[rows[ok]] = neg[ok]
        self.counts = (self.pids >= 0).sum(axis=1)
        self.name = str(path)
        log(f"teacher {path}: {ok.sum():,} queries aligned, negatives per query (mean) "
            f"{self.counts[rows[ok]].mean() if ok.any() else 0:.1f}")


def main():
    args = parse_args()
    e2e = args.mode == "e2e"
    paths = Paths()
    paths.ensure_work_dir()
    dev = get_device()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    log_run_header("06_train_concept_model.py", args, dev)
    run_dir = paths.models / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), run_dir / "args.json")
    model_name, _ = get_splade_model(paths)

    # ------------------------------------------------------------------ data
    train_q = CSR.load(paths.train_queries_csr)
    q_ids = np.asarray(train_q.ids, dtype=np.int64)
    qid_to_row = ids_to_rows(q_ids)
    qrels = read_qrels(paths.qrels_tsv("train"))
    docs = load_docs(paths, mmap=e2e)            # frozen: in RAM (fast random access)
    doc_pids = np.asarray(docs.ids, dtype=np.int64)
    pid2row = ids_to_rows(doc_pids)

    # positives (only those present among the documents)
    positives = {}
    for q, plist in positives_from_qrels(qrels).items():
        ok = [p for p, r in zip(plist, pid2row(np.array(plist))) if r >= 0]
        if ok:
            positives[q] = ok
    usable = np.array([i for i, q in enumerate(q_ids.tolist()) if q in positives], dtype=np.int64)
    perm = rng.permutation(usable)
    val_rows = np.sort(perm[: args.val_queries])
    train_rows = perm[args.val_queries:]
    if args.max_train_queries and args.max_train_queries < len(train_rows):
        train_rows = train_rows[: args.max_train_queries]
    log(f"usable queries {len(usable):,} -> training {len(train_rows):,}, validation {len(val_rows):,}")
    if len(train_rows) < args.batch_size:
        raise SystemExit("[ERROR] too few training queries for one batch: reduce --batch-size or --val-queries")

    teachers = [Teacher(resolve_negatives(n, paths), qid_to_row, train_q.n_rows,
                        args.neg_rank_min, args.neg_rank_max) for n in args.negatives]
    if args.n_hard > 0 and not teachers:
        raise SystemExit("[ERROR] --n-hard > 0 requires --negatives")

    if e2e:
        qtexts = read_queries(paths.queries_tsv("train"))
        collection = Collection(paths.collection_tsv)

    # ------------------------------------------------------------------ validation candidates
    val_qids = q_ids[val_rows]
    val_qrels = {int(q): {p: 1 for p in positives[int(q)]} for q in val_qids}
    cache = paths.val_candidates / f"seed{args.seed}_n{len(val_rows)}_c{args.val_candidates}_{len(usable)}.npz"
    if cache.exists():
        z = np.load(cache)
        cand_rows = z["rows"]
        assert np.array_equal(z["val_rows"], val_rows), "inconsistent candidate cache"
        log(f"validation candidates loaded from cache {cache.name}")
    else:
        with timer("validation candidates (exact SPLADE search)"):
            tdocs = docs_to_device(docs, dev)
            _, cand_rows = sparse_search(tdocs, train_q, args.val_candidates, query_rows=val_rows, verbose=False)
            tdocs.free()
            del tdocs
        paths.val_candidates.mkdir(parents=True, exist_ok=True)
        np.savez(cache, rows=cand_rows, val_rows=val_rows)
    splade_val, _ = evaluate_run(val_qids, doc_pids[cand_rows], val_qrels)
    log(f"SPLADE reference on the candidates: MRR@10={splade_val['MRR@10']:.4f} "
        f"R@{args.val_candidates}={splade_val.get(f'R@{args.val_candidates}', float('nan')):.4f}")

    # ------------------------------------------------------------------ model
    encoder = None
    if args.init_from:
        head, info = load_checkpoint(resolve_ckpt(args.init_from, paths), dev)
        if head.k != args.k:
            raise SystemExit(f"[ERROR] the checkpoint has k={head.k}, but k={args.k} was requested")
        backbone = info.get("backbone_path")
    else:
        head = ConceptHead(train_q.n_cols, args.k, init_std=args.init_std, seed=args.seed).to(dev)
        backbone = None
    if e2e:
        from splade_encoder import SpladeEncoder
        encoder = SpladeEncoder(backbone or model_name).to(dev)
        if args.grad_checkpointing:
            encoder.mlm.gradient_checkpointing_enable()
        encoder.train()
    elif backbone:
        raise SystemExit("[ERROR] cannot initialize from an e2e checkpoint in frozen mode")
    head.train()

    groups = [{"params": [head.weight], "lr": args.lr}]
    if encoder is not None:
        groups.append({"params": [p for p in encoder.parameters() if p.requires_grad], "lr": args.lr_backbone})
    optim = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    total_steps = args.max_steps or int(math.ceil(args.epochs * steps_per_epoch))
    warmup = int(args.warmup_ratio * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    log(f"steps per epoch {steps_per_epoch:,}, total steps {total_steps:,}, warmup {warmup}")
    amp = bool(args.amp and e2e and dev.type == "cuda")

    # ------------------------------------------------------------------ encoding functions
    def encode_query_rows(rows):
        if not e2e:
            return head.forward_bags(*csr_batch_to_torch(train_q, rows, dev))
        texts = [qtexts[int(q)] for q in q_ids[rows]]
        ids, mask = encoder.tokenize(texts, args.q_max_len, dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            return head.forward_dense(encoder(ids, mask))

    def encode_doc_pids(pids):
        if not e2e:
            rows = pid2row(pids)
            assert (rows >= 0).all(), "unknown document pid"
            return head.forward_bags(*csr_batch_to_torch(docs, rows, dev))
        ids, mask = encoder.tokenize(collection.get_many(pids), args.d_max_len, dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            return head.forward_dense(encoder(ids, mask))

    def sample_negatives(rows_b):
        B = len(rows_b)
        out = np.full((B, args.n_hard), -1, dtype=np.int64)
        if args.n_hard == 0:
            return out
        avail = np.stack([t.counts[rows_b] > 0 for t in teachers], axis=1)
        choice = np.where(avail.any(axis=1), (rng.random(avail.shape) * avail).argmax(axis=1), -1)
        for ti, t in enumerate(teachers):
            sel = np.flatnonzero(choice == ti)
            if len(sel) == 0:
                continue
            cnt = t.counts[rows_b[sel]]
            j = (rng.random((len(sel), args.n_hard)) * cnt[:, None]).astype(np.int64)
            out[sel] = t.pids[rows_b[sel][:, None], j]
        none = choice < 0
        if none.any():  # no teacher has negatives for this query: random documents
            out[none] = doc_pids[rng.integers(0, len(doc_pids), (int(none.sum()), args.n_hard))]
        return out

    @torch.no_grad()
    def validate():
        head.eval()
        if encoder is not None:
            encoder.eval()
        gq = torch.cat([encode_query_rows(val_rows[s:s + 256]).float() for s in range(0, len(val_rows), 256)])
        uniq = np.unique(cand_rows)
        bs = 4096 if not e2e else 128
        gd = torch.cat([encode_doc_pids(doc_pids[uniq[s:s + bs]]).float() for s in range(0, len(uniq), bs)])
        pos = torch.from_numpy(np.searchsorted(uniq, cand_rows)).to(dev)
        S = (gq[:, None, :] * gd[pos]).sum(-1)                      # (n_val, n_candidates)
        order = torch.argsort(S, dim=1, descending=True).cpu().numpy()
        ranked = np.take_along_axis(cand_rows, order, axis=1)
        m, _ = evaluate_run(val_qids, doc_pids[ranked], val_qrels)
        head.train()
        if encoder is not None:
            encoder.train()
        return m

    # ------------------------------------------------------------------ training loop
    best_mrr, step, t0 = -1.0, 0, time.time()
    log_csv = run_dir / "train_log.csv"
    m0 = validate()
    log(f"[step 0] validation: MRR@10={m0['MRR@10']:.4f} (random W)")
    run_loss = run_acc = 0.0
    n_run = 0
    done = False
    epoch = 0
    while not done:
        order = rng.permutation(train_rows)
        for s in range(0, len(order), args.batch_size):
            rows_b = order[s:s + args.batch_size]
            if len(rows_b) < 2:
                continue
            qids_b = q_ids[rows_b]
            pos = np.array([positives[int(q)][rng.integers(len(positives[int(q)]))] for q in qids_b],
                           dtype=np.int64)
            negs = sample_negatives(rows_b)
            col_pids = np.concatenate([pos, negs.ravel()])
            labels_np = np.arange(len(rows_b))
            mask_np = false_negative_mask([positives[int(q)] for q in qids_b], col_pids, labels_np)
            uniq, inv = np.unique(col_pids, return_inverse=True)

            gq = encode_query_rows(rows_b)
            gd = encode_doc_pids(uniq)[torch.from_numpy(inv).to(dev)]
            loss, acc, _ = info_nce_loss(gq, gd, torch.from_numpy(labels_np).to(dev), args.temperature,
                                         torch.from_numpy(mask_np).to(dev))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], args.grad_clip)
            optim.step()
            sched.step()
            step += 1
            run_loss += float(loss)
            run_acc += float(acc)
            n_run += 1

            if step % args.log_every == 0 or step == total_steps:
                el = time.time() - t0
                row = {"step": step, "epoch": round(step / steps_per_epoch, 3), "loss": run_loss / n_run,
                       "acc_in_batch": run_acc / n_run, "lr": sched.get_last_lr()[0], "sec": round(el, 1)}
                log(f"step {step}/{total_steps}  ep {row['epoch']:.2f}  loss {row['loss']:.4f}  "
                    f"acc {row['acc_in_batch']:.3f}  lr {row['lr']:.2e}  ({el / step:.3f} s/step)")
                append_csv_row(log_csv, row)
                run_loss = run_acc = 0.0
                n_run = 0

            if step % args.val_every == 0 or step == total_steps:
                m = validate()
                log(f"[step {step}] validation: MRR@10={m['MRR@10']:.4f} nDCG@10={m['nDCG@10']:.4f} "
                    f"(SPLADE on the same candidates: {splade_val['MRR@10']:.4f})")
                append_csv_row(log_csv, {"step": step, "val_MRR@10": m["MRR@10"], "val_nDCG@10": m["nDCG@10"]})
                extra = {"val": m, "splade_val": splade_val, "n_train_queries": len(train_rows),
                         "teachers": [t.name for t in teachers]}
                if m["MRR@10"] > best_mrr:
                    best_mrr = m["MRR@10"]
                    save_checkpoint(run_dir / "best", head, args.mode, model_name, vars(args), step, extra,
                                    encoder)
                    log(f"  new best -> saved to {run_dir / 'best'}")
            if step >= total_steps:
                done = True
                break
        epoch += 1

    save_checkpoint(run_dir / "last", head, args.mode, model_name, vars(args), step,
                    {"best_val_MRR@10": best_mrr}, encoder)
    save_json({"best_val_MRR@10": best_mrr, "splade_val": splade_val, "steps": step,
               "minutes": (time.time() - t0) / 60}, run_dir / "summary.json")
    log(f"Done. Best validation MRR@10: {best_mrr:.4f}. Models in {run_dir}")


if __name__ == "__main__":
    main()
