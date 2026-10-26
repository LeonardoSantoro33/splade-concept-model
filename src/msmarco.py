"""
msmarco.py — Everything related to raw-text MS MARCO: downloading the official files,
reading queries, qrels (relevance judgments) and passage texts.

The provided pre-encoded data already contains the document SPLADE vectors; here we need:
  - dev qrels -> to compute MRR@10 etc. (the "ground truth" on which passages are relevant)
  - training queries + qrels -> to train the Concept Model (Sec. 6 of the paper)
  - passage texts -> only for the end-to-end variant (which re-encodes the texts)
"""
import gzip
import hashlib
import mmap
import os
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from utils import log

BASE = "https://msmarco.z22.web.core.windows.net/msmarcoranking/"
# (url, expected md5) — md5s taken from the official ir_datasets list
DOWNLOADS = {
    "collectionandqueries": (BASE + "collectionandqueries.tar.gz", "31644046b18952c1386cd4564ba2ae69"),
    "dl19_queries": (BASE + "msmarco-test2019-queries.tsv.gz", "eda71eccbe4d251af83150abe065368c"),
    "dl20_queries": (BASE + "msmarco-test2020-queries.tsv.gz", "00a406fb0d14ed3752d70d1e4eb98600"),
    "dl19_qrels": ("https://trec.nist.gov/data/deep/2019qrels-pass.txt", "2f4be390198da108f6845c822e5ada14"),
    "dl20_qrels": ("https://trec.nist.gov/data/deep/2020qrels-pass.txt", "0355ccee7509ac0463e8278186cdd8d1"),
}
# files we need from inside collectionandqueries.tar.gz
CORE_MEMBERS = ["collection.tsv", "queries.train.tsv", "qrels.train.tsv",
                "queries.dev.small.tsv", "qrels.dev.small.tsv"]


# ----------------------------------------------------------------------
#  download
# ----------------------------------------------------------------------
def md5sum(path, block=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, expected_md5=None, retries=3):
    dest = Path(dest)
    if dest.exists() and (expected_md5 is None or md5sum(dest) == expected_md5):
        log(f"already present: {dest.name}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            log(f"downloading {url} (attempt {attempt})")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length", 0))
                done, last = 0, time.time()
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 15:
                        pct = f"{100 * done / total:.1f}%" if total else ""
                        log(f"  {done / 1e6:.0f} MB {pct}")
                        last = time.time()
            if expected_md5 and md5sum(tmp) != expected_md5:
                raise IOError("md5 does not match the expected value (corrupted file?)")
            os.replace(tmp, dest)
            log(f"ok: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
            return dest
        except Exception as e:  # noqa: BLE001 - we want to retry on any network error
            log(f"  error: {e}")
            time.sleep(5 * attempt)
    raise SystemExit(f"[ERROR] Download failed: {url}\n"
                     "Try from another node, or download it manually and use --from-tar.")


def extract_members(tar_path, out_dir, members):
    """Extract only the requested files (flattening directories) from a .tar.gz."""
    out_dir = Path(out_dir)
    wanted = set(members)
    found = set()
    with tarfile.open(tar_path, "r:*") as tar:
        for m in tar:
            name = Path(m.name).name
            if m.isfile() and name in wanted:
                log(f"extracting {name} ({m.size / 1e6:.1f} MB)")
                src = tar.extractfile(m)
                with open(out_dir / name, "wb") as f:
                    while True:
                        chunk = src.read(1 << 24)
                        if not chunk:
                            break
                        f.write(chunk)
                found.add(name)
    missing = wanted - found
    if missing:
        raise SystemExit(f"[ERROR] Missing from the tar archive: {sorted(missing)}")


# ----------------------------------------------------------------------
#  reading queries / qrels
# ----------------------------------------------------------------------
def _open_text(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def read_queries(path):
    """File 'qid<TAB>text' -> dict {qid(int): text}."""
    out = {}
    with _open_text(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            qid, text = line.split("\t", 1)
            out[int(qid)] = text
    return out


def read_qrels(path):
    """qrels in TREC format 'qid 0 pid rel' (tabs or spaces) -> dict {qid: {pid: rel}}."""
    out = {}
    with _open_text(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, pid, rel = parts[:4]
            out.setdefault(int(qid), {})[int(pid)] = int(rel)
    return out


def positives_from_qrels(qrels, rel_threshold=1):
    """{qid: [relevant pids]}, where relevant means rel >= threshold."""
    return {q: [p for p, r in d.items() if r >= rel_threshold] for q, d in qrels.items()
            if any(r >= rel_threshold for r in d.values())}


QUERY_SETS = {
    # name: (query file in msmarco/, qrels file, threshold for binary metrics)
    "dev": ("queries.dev.small.tsv", "qrels.dev.small.tsv", 1),
    "dl19": ("msmarco-test2019-queries.tsv.gz", "2019qrels-pass.txt", 2),
    "dl20": ("msmarco-test2020-queries.tsv.gz", "2020qrels-pass.txt", 2),
}


def load_query_set(name, paths, need_texts=True):
    """Return (texts {qid: text} or None, qrels, relevance_threshold).
    For TREC DL, only queries with judgments are kept (43 for DL19, 54 for DL20)."""
    qfile, rfile, thr = QUERY_SETS[name]
    qrels_path = paths.msmarco / rfile
    if not qrels_path.exists():
        raise SystemExit(f"[ERROR] Missing qrels {qrels_path}. First run 02_download_msmarco.py"
                         + (" --trec-dl" if name != "dev" else ""))
    qrels = read_qrels(qrels_path)
    texts = None
    if need_texts:
        qpath = paths.msmarco / qfile
        if not qpath.exists():
            raise SystemExit(f"[ERROR] Missing queries {qpath}. Run 02_download_msmarco.py first")
        texts = read_queries(qpath)
        texts = {q: t for q, t in texts.items() if q in qrels}
    return texts, qrels, thr


# ----------------------------------------------------------------------
#  passage texts (collection.tsv, ~3 GB): access by pid without loading it all
# ----------------------------------------------------------------------
class Collection:
    """Read collection.tsv ('pid<TAB>text', one line per passage) using an index of
    line start offsets, so get(pid) is instantaneous and does not need 10 GB of RAM."""

    def __init__(self, tsv_path, index_path=None):
        self.path = Path(tsv_path)
        if not self.path.exists():
            raise SystemExit(f"[ERROR] Missing {self.path}. Run 02_download_msmarco.py first")
        index_path = Path(index_path) if index_path else self.path.with_suffix(".offsets.npy")
        if index_path.exists() and index_path.stat().st_mtime >= self.path.stat().st_mtime:
            self.offsets = np.load(index_path)
        else:
            self.offsets = self._build_offsets()
            try:
                np.save(index_path, self.offsets)
            except OSError:
                pass
        self._f = open(self.path, "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        # in MS MARCO the pid equals the line number: check it on a few lines
        self.pid_is_line = all(self._line(i)[0] == i for i in [0, len(self) // 2, len(self) - 1])
        if not self.pid_is_line:
            log("collection.tsv: pids differ from line numbers -> building the pid->line table")
            pids = np.array([self._line(i)[0] for i in range(len(self))], dtype=np.int64)
            self._pid2line = {int(p): i for i, p in enumerate(pids)}

    def _build_offsets(self):
        log(f"indexing the lines of {self.path.name} (one-time only)...")
        starts = [np.zeros(1, dtype=np.int64)]
        size = self.path.stat().st_size
        with open(self.path, "rb") as f:
            base = 0
            while True:
                buf = f.read(1 << 26)
                if not buf:
                    break
                nl = np.flatnonzero(np.frombuffer(buf, dtype=np.uint8) == 10) + base + 1
                starts.append(nl.astype(np.int64))
                base += len(buf)
        offs = np.concatenate(starts)
        if offs[-1] >= size:      # the last '\n' ends the file: it is not a line start
            offs = offs[:-1]
        log(f"  {len(offs):,} lines")
        return offs

    def __len__(self):
        return len(self.offsets)

    def _line(self, i):
        a = int(self.offsets[i])
        b = int(self.offsets[i + 1]) if i + 1 < len(self.offsets) else len(self._mm)
        raw = self._mm[a:b].decode("utf-8").rstrip("\n")
        pid, text = raw.split("\t", 1)
        return int(pid), text

    def get(self, pid):
        pid = int(pid)
        line = pid if self.pid_is_line else self._pid2line[pid]
        p, text = self._line(line)
        assert p == pid, f"pid {pid} read as {p}"
        return text

    def get_many(self, pids):
        return [self.get(p) for p in pids]
