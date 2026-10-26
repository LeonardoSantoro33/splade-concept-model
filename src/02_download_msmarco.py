"""
02_download_msmarco.py — Download the official MS MARCO passage files (queries, qrels, texts).

Downloads collectionandqueries.tar.gz (~1 GB) and extracts:
    collection.tsv          8.8M plain-text passages (needed only by the end-to-end variant)
    queries.train.tsv       ~800k training queries
    qrels.train.tsv         training relevance judgments (~530k pairs)
    queries.dev.small.tsv   6980 evaluation queries (the same as the "dev" MRR@10)
    qrels.dev.small.tsv     dev relevance judgments
With --trec-dl it also downloads TREC DL 2019 and 2020 (queries + graded qrels).

This is a network job, not a compute job: it can also be run from the login node.
    python -u 02_download_msmarco.py --trec-dl
If the download fails, fetch the tar manually and use:
    python -u 02_download_msmarco.py --from-tar /path/to/collectionandqueries.tar.gz
"""
import argparse
from pathlib import Path

from config import Paths
from msmarco import CORE_MEMBERS, DOWNLOADS, download, extract_members, read_qrels, read_queries
from utils import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trec-dl", action="store_true", help="also download TREC DL 2019/2020")
    ap.add_argument("--from-tar", default=None, help="use an already downloaded collectionandqueries.tar.gz")
    ap.add_argument("--keep-tar", action="store_true", help="do not delete the .tar.gz after extraction")
    args = ap.parse_args()
    paths = Paths()
    paths.ensure_work_dir()
    out = paths.msmarco

    if all((out / m).exists() for m in CORE_MEMBERS):
        log("The main files are already present.")
    else:
        if args.from_tar:
            tar_path = Path(args.from_tar)
        else:
            url, md5 = DOWNLOADS["collectionandqueries"]
            tar_path = download(url, out / "collectionandqueries.tar.gz", md5)
        extract_members(tar_path, out, CORE_MEMBERS)
        if not args.keep_tar and not args.from_tar:
            tar_path.unlink()
            log("tar deleted (use --keep-tar to keep it)")

    if args.trec_dl:
        for key in ["dl19_queries", "dl20_queries", "dl19_qrels", "dl20_qrels"]:
            url, md5 = DOWNLOADS[key]
            download(url, out / Path(url).name, md5)

    # summary
    log("Contents of " + str(out))
    for name, kind in [("queries.train.tsv", "q"), ("qrels.train.tsv", "r"), ("queries.dev.small.tsv", "q"),
                       ("qrels.dev.small.tsv", "r"), ("msmarco-test2019-queries.tsv.gz", "q"),
                       ("2019qrels-pass.txt", "r"), ("msmarco-test2020-queries.tsv.gz", "q"),
                       ("2020qrels-pass.txt", "r")]:
        p = out / name
        if not p.exists():
            continue
        if kind == "q":
            log(f"  {name:<34} {len(read_queries(p)):>9,} queries")
        else:
            qr = read_qrels(p)
            log(f"  {name:<34} {len(qr):>9,} queries with judgments, {sum(len(v) for v in qr.values()):,} judgments")
    if (out / "collection.tsv").exists():
        log(f"  collection.tsv                     {(out / 'collection.tsv').stat().st_size / 1e9:.2f} GB")
    log("Done.")


if __name__ == "__main__":
    main()
