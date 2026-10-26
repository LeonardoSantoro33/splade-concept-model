"""
config.py — Paths and settings shared by ALL scripts.

Set the two directories with environment variables (recommended, e.g. in ~/.bashrc):

    export FM_DIR=/path/to/pre-encoded/splade/msmarco
    export WORK_DIR=/path/to/large/scratch/splade_concept

  FM_DIR   : directory with the provided pre-encoded data (READ-ONLY: never write to it)
  WORK_DIR : YOUR directory, where caches, models and results are saved.
             It needs space (~60-80 GB in total): do NOT use your home directory.
"""
import json
import os
from pathlib import Path

USER = os.environ.get("USER", "user")

# Provided pre-encoded SPLADE data (Seismic format).
FM_DIR = Path(os.environ.get("FM_DIR", "data/encoded"))

# Working directory (large). Change it if you do not have write permissions.
WORK_DIR = Path(os.environ.get("WORK_DIR", "work"))

# SPLADE model used to encode queries/texts that are NOT already in the provided files
# (training queries, TREC DL, end-to-end variant). Script 03 checks which model
# actually reproduces the provided vectors and saves the choice.
DEFAULT_SPLADE_MODEL = os.environ.get("SPLADE_MODEL", "naver/splade-cocondenser-ensembledistil")
CANDIDATE_SPLADE_MODELS = [
    "naver/splade-cocondenser-ensembledistil",
    "naver/splade-cocondenser-selfdistil",
]

# File names in the provided data directory.
FM_FILES = {
    "documents": "documents.bin",
    "queries": "queries.bin",
    "doc_ids": "doc_ids.npy",
    "query_ids": "queries_ids.npy",
    "mapping": "token_to_id_mapping.json",
    "groundtruth": "groundtruth_100.tsv",
}


class Paths:
    """All paths derived from WORK_DIR, in one place."""

    def __init__(self, work_dir=None, fm_dir=None):
        self.work = Path(work_dir) if work_dir else WORK_DIR
        self.fm = Path(fm_dir) if fm_dir else FM_DIR

        # cache (SPLADE vectors converted to the BERT vocabulary space, 30522 dims)
        self.cache = self.work / "cache"
        self.docs_csr = self.cache / "docs_bert"            # 8.8M documents (provided)
        self.dev_queries_csr = self.cache / "dev_queries_bert"  # provided queries (dev)
        self.train_queries_csr = self.cache / "train_queries_bert"  # training queries (encoded by us)
        self.val_candidates = self.cache / "val_candidates"

        # MS MARCO raw text (passages, queries, qrels)
        self.msmarco = self.work / "msmarco"

        # output
        self.negatives = self.work / "negatives"
        self.models = self.work / "models"
        self.embeddings = self.work / "embeddings"
        self.results = self.work / "results"
        self.runs = self.work / "runs"
        self.lce = self.work / "lce"
        self.encoder_check = self.work / "encoder_check.json"
        self.summary_csv = self.results / "summary.csv"

    def fm_file(self, key):
        return self.fm / FM_FILES[key]

    # --- MS MARCO files ---
    @property
    def collection_tsv(self):
        return self.msmarco / "collection.tsv"

    def queries_tsv(self, split):
        names = {
            "train": "queries.train.tsv",
            "dev": "queries.dev.small.tsv",
            "dl19": "msmarco-test2019-queries.tsv.gz",
            "dl20": "msmarco-test2020-queries.tsv.gz",
        }
        return self.msmarco / names[split]

    def qrels_tsv(self, split):
        names = {
            "train": "qrels.train.tsv",
            "dev": "qrels.dev.small.tsv",
            "dl19": "2019qrels-pass.txt",
            "dl20": "2020qrels-pass.txt",
        }
        return self.msmarco / names[split]

    def ensure_work_dir(self):
        """Create WORK_DIR (and subdirectories), with a clear message if that is not possible."""
        try:
            for d in [self.work, self.cache, self.msmarco, self.negatives, self.models,
                      self.embeddings, self.results, self.runs, self.lce]:
                d.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise SystemExit(
                f"\n[ERROR] Cannot create {self.work} (permission denied).\n"
                "Choose a directory you can write to, for example:\n"
                "    export WORK_DIR=/data/$USER/splade_concept\n"
                "or change WORK_DIR in config.py.\n")


def resolve_ckpt(name_or_path, paths):
    """'frozen_k256_r1' -> WORK_DIR/models/frozen_k256_r1/best (if it exists), otherwise the path."""
    p = Path(name_or_path)
    if p.exists():
        return p
    for cand in [paths.models / name_or_path / "best", paths.models / name_or_path / "last",
                 paths.models / name_or_path]:
        if (cand / "concept_head.pt").exists():
            return cand
    raise SystemExit(f"[ERROR] Checkpoint not found: {name_or_path} (also searched in {paths.models})")


def resolve_negatives(name_or_path, paths):
    """'r1' -> WORK_DIR/negatives/r1.npz"""
    p = Path(name_or_path)
    if p.exists():
        return p
    cand = paths.negatives / (name_or_path if name_or_path.endswith(".npz") else name_or_path + ".npz")
    if cand.exists():
        return cand
    raise SystemExit(f"[ERROR] Negatives file not found: {name_or_path} (also tried {cand})")


def get_splade_model(paths):
    """SPLADE model to use: the one chosen by 03_check_encoder.py, if available."""
    if "SPLADE_MODEL" in os.environ:
        return os.environ["SPLADE_MODEL"], 1.0
    if paths.encoder_check.exists():
        info = json.loads(paths.encoder_check.read_text())
        return info["model"], float(info.get("scale", 1.0))
    return DEFAULT_SPLADE_MODEL, 1.0
