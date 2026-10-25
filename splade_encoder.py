"""
splade_encoder.py — SPLADE: text -> sparse 30522-dimensional vector (one dimension per BERT token).

    w_j(x) = max_{positions i}  log(1 + ReLU(logit_{i,j}))          (Formal et al., 2021)

where logit_{i,j} is the output of BERT's MLM head for position i and token j.
We take the max first and then apply log(1+ReLU(.)): the result is IDENTICAL (the function is
monotonically increasing) but uses much less GPU memory during end-to-end training.
"""
import numpy as np
import torch
from torch import nn

from sparse_io import CSR
from utils import log, make_transformers_cpu_safe

make_transformers_cpu_safe()
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402


class SpladeEncoder(nn.Module):
    def __init__(self, name_or_path):
        super().__init__()
        self.name_or_path = str(name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        self.mlm = AutoModelForMaskedLM.from_pretrained(self.name_or_path)
        self.vocab_size = int(self.mlm.config.vocab_size)

    def forward(self, input_ids, attention_mask):
        logits = self.mlm(input_ids=input_ids, attention_mask=attention_mask).logits   # (B, L, V)
        pad = ~attention_mask.bool().unsqueeze(-1)
        pooled = logits.masked_fill(pad, float("-inf")).max(dim=1).values             # (B, V)
        return torch.log1p(torch.relu(pooled))

    def tokenize(self, texts, max_length, device):
        enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=max_length,
                             return_tensors="pt")
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def encode_dense(self, texts, max_length, device, amp=False):
        ids, mask = self.tokenize(texts, max_length, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=bool(amp and device.type == "cuda")):
            return self(ids, mask).float()

    @torch.no_grad()
    def encode_to_csr(self, texts, device, batch_size=128, max_length=256, amp=False, ids=None,
                      scale=1.0, verbose=True):
        """Encode a list of texts into a CSR (BERT space). Texts are sorted by length
        (less padding = faster) and then restored to their original order."""
        was_training = self.training
        self.eval()
        texts = list(texts)
        order = np.argsort([len(t) for t in texts], kind="stable")
        parts = []
        step = max(1, (len(texts) // batch_size) // 20)
        for b, s in enumerate(range(0, len(texts), batch_size)):
            batch_idx = order[s:s + batch_size]
            x = self.encode_dense([texts[i] for i in batch_idx], max_length, device, amp) * scale
            r, c = torch.nonzero(x > 0, as_tuple=True)
            vals = x[r, c]
            counts = torch.bincount(r, minlength=x.shape[0]).cpu().numpy()
            indptr = np.zeros(x.shape[0] + 1, dtype=np.int64)
            np.cumsum(counts, out=indptr[1:])
            parts.append(CSR(indptr, c.cpu().numpy().astype(np.int32),
                             vals.cpu().numpy().astype(np.float32), self.vocab_size))
            if verbose and b % step == 0:
                log(f"  encoded {min(s + batch_size, len(texts)):,}/{len(texts):,} texts")
        if was_training:
            self.train()
        csr_sorted = CSR.concat(parts)
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[order] = np.arange(len(order))
        out = csr_sorted.subset(inverse)
        out.ids = None if ids is None else np.asarray(ids)
        return out

    def save(self, folder):
        self.mlm.save_pretrained(folder)
        self.tokenizer.save_pretrained(folder)


def top_tokens(tokenizer, idx, vals, n=15):
    """[(token, weight)] for the highest weights of a sparse vector (BERT ids)."""
    idx = np.asarray(idx)
    vals = np.asarray(vals)
    order = np.argsort(-vals)[:n]
    toks = tokenizer.convert_ids_to_tokens([int(i) for i in idx[order]])
    return [(t, float(v)) for t, v in zip(toks, vals[order])]
