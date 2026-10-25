"""
concept_model.py — The SPLADE Concept Model from the paper (Sec. 5-6).

    g(x) = W . SPLADE(x)                                   Eq. (4)
    score(q, d) = g(q) . g(d)                              Eq. (5)
    L(q, d+) = -log  exp(s(q,d+)/b) / [exp(s(q,d+)/b) + sum_i exp(s(q,d_i-)/b)]   Eq. (6)

W is a k x 30522 matrix (k "concepts"). For convenience we store it transposed:
self.weight has shape (30522, k), so that g(x) = x @ weight.

Two variants (Sec. 5.3):
  - frozen : SPLADE is fixed, only W is trained (document SPLADE vectors are the provided ones)
  - e2e    : both W and BERT/SPLADE are trained (requires the texts, much more expensive)
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils import jsonable, log


class ConceptHead(nn.Module):
    def __init__(self, vocab_size, k, init_std=0.02, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        # "W — initialized randomly" (Sec. 5.3)
        self.weight = nn.Parameter(torch.randn(vocab_size, k, generator=gen) * init_std)

    @property
    def k(self):
        return self.weight.shape[1]

    @property
    def vocab_size(self):
        return self.weight.shape[0]

    def forward_bags(self, idx, offsets, vals):
        """g(x) for sparse vectors in embedding_bag's (indices, offsets, weights) format."""
        return F.embedding_bag(idx, self.weight, offsets, mode="sum",
                               per_sample_weights=vals.to(self.weight.dtype))

    def forward_dense(self, x):
        """g(x) for dense SPLADE vectors (B, 30522) — used in the end-to-end variant."""
        return x @ self.weight.to(x.dtype)


def csr_batch_to_torch(csr, rows, device):
    """Rows of a numpy CSR -> (idx, offsets, vals) on device, ready for forward_bags."""
    offsets, idx, vals = csr.gather(rows)
    return (torch.from_numpy(idx.astype(np.int64)).to(device, non_blocking=True),
            torch.from_numpy(offsets).to(device, non_blocking=True),
            torch.from_numpy(vals.astype(np.float32)).to(device, non_blocking=True))


def info_nce_loss(q, d, labels, temperature=1.0, mask=None):
    """Eq. (6). q: (B, k), d: (C, k) with C = all documents in the batch, labels: (B,) column
    of each query's positive. All other documents in the batch act as negatives
    (in-batch negatives, Sec. 6.2, + optional hard negatives, Sec. 6.3).
    mask: (B, C) True where the document must NOT be used as a negative (e.g. it is another
    positive of the same query: it would be a false negative)."""
    logits = (q.float() @ d.float().T) / temperature
    if mask is not None:
        logits = logits.masked_fill(mask, float("-inf"))
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        acc = (logits.argmax(dim=1) == labels).float().mean()
    return loss, acc, logits


def false_negative_mask(query_positive_sets, col_pids, labels):
    """Mask (B, C): True if column j contains a known positive of query i
    (other than its label column). Vectorized, no nested Python loops."""
    B, C = len(query_positive_sets), len(col_pids)
    mask = np.zeros((B, C), dtype=bool)
    pair_q = np.concatenate([np.full(len(s), i, dtype=np.int64) for i, s in enumerate(query_positive_sets)])
    pair_p = np.concatenate([np.asarray(list(s), dtype=np.int64) for s in query_positive_sets])
    if len(pair_p) == 0:
        return mask
    order = np.argsort(col_pids, kind="stable")
    sorted_p = np.asarray(col_pids)[order]
    left = np.searchsorted(sorted_p, pair_p, "left")
    right = np.searchsorted(sorted_p, pair_p, "right")
    n = right - left
    if n.sum() == 0:
        return mask
    rep_q = np.repeat(pair_q, n)
    starts = np.repeat(left - np.concatenate([[0], np.cumsum(n)[:-1]]), n) + np.arange(n.sum())
    cols = order[starts]
    mask[rep_q, cols] = True
    mask[np.arange(B), labels] = False
    return mask


# ----------------------------------------------------------------------
#  saving / loading
# ----------------------------------------------------------------------
def save_checkpoint(folder, head, mode, base_model, args, step, extra=None, encoder=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({
        "weight": head.weight.detach().float().cpu(),
        "k": int(head.k),
        "vocab_size": int(head.vocab_size),
        "mode": mode,
        "base_model": str(base_model),
        "step": int(step),
        "args": jsonable(args),
        "extra": jsonable(extra or {}),
    }, folder / "concept_head.pt")
    if encoder is not None:
        encoder.save(folder / "backbone")


def load_checkpoint(folder, device="cpu"):
    folder = Path(folder)
    obj = torch.load(folder / "concept_head.pt", map_location="cpu", weights_only=True)
    head = ConceptHead(obj["vocab_size"], obj["k"])
    with torch.no_grad():
        head.weight.copy_(obj["weight"])
    head.to(device)
    info = {k: v for k, v in obj.items() if k != "weight"}
    backbone = folder / "backbone"
    info["backbone_path"] = str(backbone) if backbone.exists() else None
    log(f"loaded checkpoint {folder} (mode={info['mode']}, k={info['k']}, step={info['step']})")
    return head, info
