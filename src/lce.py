"""
lce.py — Latent Concept Expansion (Sec. 3 of the paper), written for ANY sparse vector.

Today we apply it to SPLADE vectors (30522 tokens); once SPLARE is available it is enough to pass
the SAE vectors: the code is identical (only the dimension of the space changes).

  Eq. (1)  M_ij = sum_doc 1[V_doc(i) > 0] * 1[V_doc(j) > 0]        (diagonal = 0)
  Eq. (2)  V_q_ext = V_q + alpha * (M . V_q)
  Eq. (3)  pruning: keep only weights > tau   (or the top-k)

Practical note (worth mentioning in the paper): on MS MARCO the raw counts in M reach millions,
so M.V_q has a scale vastly different from V_q and alpha becomes unmanageable. For this reason
we provide:
  - normalizations of M: raw (plain Eq. 1), cond = P(j|i), cosine, ppmi
  - "relative" alpha: the strongest expansion term equals alpha * max(V_q)
"""
import numpy as np
import torch

from retrieval import csr_rows_to_dense
from utils import log


@torch.no_grad()
def build_cooccurrence(csr, dev, rows=None, chunk=16384):
    """M = B^T B with B the binary documents x terms matrix (Eq. 1). Returns (M, df) on dev.
    Dense computation on GPU in blocks of documents: in float32 (TF32) the counts are EXACT
    up to 2^24 = 16.7M, more than the number of documents in MS MARCO."""
    V = csr.n_cols
    rows = np.arange(csr.n_rows) if rows is None else np.asarray(rows)
    M = torch.zeros(V, V, dtype=torch.float32, device=dev)
    step = max(1, (len(rows) // chunk) // 20)
    for b, s in enumerate(range(0, len(rows), chunk)):
        X = csr_rows_to_dense(csr, rows[s:s + chunk], dev)
        Bm = (X > 0).float()
        del X
        M.addmm_(Bm.T, Bm)
        del Bm
        if b % step == 0:
            log(f"  co-occurrences: {min(s + chunk, len(rows)):,}/{len(rows):,} documents")
    df = torch.diagonal(M).clone()
    return M, df


@torch.no_grad()
def normalize_cooccurrence(M, df, kind, n_docs, min_count=1):
    """Return the matrix B used in the expansion:  E = V_q @ B,
    with B[i, j] = how strongly query term i "recalls" term j."""
    B = M.clone()
    if min_count > 1:
        B[B < min_count] = 0.0
    dfc = df.clamp_min(1.0)
    if kind == "raw":
        pass
    elif kind == "cond":                                   # P(j | i) = M_ij / df_i
        B /= dfc[:, None]
    elif kind == "cosine":                                 # M_ij / sqrt(df_i df_j)
        s = dfc.sqrt()
        B /= s[:, None]
        B /= s[None, :]
    elif kind == "ppmi":                                   # max(0, log(N M_ij / (df_i df_j)))
        nz = B > 0                                         # in-place operations: M is large (V x V)
        B.mul_(float(n_docs))
        B.div_(dfc[:, None])
        B.div_(dfc[None, :])
        B.clamp_(min=1e-12).log_()
        B.mul_(nz)
        del nz
        B.clamp_(min=0.0)
    else:
        raise ValueError(f"unknown normalization: {kind}")
    B.fill_diagonal_(0.0)                                  # M_ii = 0 (Sec. 3.4)
    return B


@torch.no_grad()
def expand(Qd, B, alpha, alpha_mode="relative"):
    """Eq. (2). Qd: (nq, V) dense queries. Returns V_q_ext (nq, V)."""
    if alpha == 0:
        return Qd.clone()
    E = Qd @ B
    if alpha_mode == "absolute":
        return Qd + alpha * E
    qmax = Qd.max(dim=1).values
    emax = E.max(dim=1).values
    a = torch.where(emax > 0, alpha * qmax / emax.clamp_min(1e-12), torch.zeros_like(emax))
    return Qd + a[:, None] * E


@torch.no_grad()
def prune(Vext, spec, Qd=None, keep_original=False):
    """Eq. (3). spec: 'none' | 'tau:0.5' (keep weights > tau) | 'topk:64' (keep the 64 highest)."""
    if spec == "none":
        out = Vext
    elif spec.startswith("tau:"):
        tau = float(spec.split(":")[1])
        out = torch.where(Vext > tau, Vext, torch.zeros((), device=Vext.device))
    elif spec.startswith("topk:"):
        k = min(int(spec.split(":")[1]), Vext.shape[1])
        vals, idx = torch.topk(Vext, k, dim=1)
        out = torch.zeros_like(Vext).scatter_(1, idx, vals)
    else:
        raise ValueError(f"unknown pruning: {spec}")
    if keep_original and Qd is not None:
        out = torch.where(Qd > 0, Vext, out)
    return out
