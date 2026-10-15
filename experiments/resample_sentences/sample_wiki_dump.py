#!/usr/bin/env python3
"""Sample random English Wikipedia articles from the wikimedia/wikipedia dump.

Source: Hugging Face dataset wikimedia/wikipedia, config 20231101.en (41 parquet
shards, ~6.4M articles, ~1k articles per row group). Parquet is read over HTTP
range requests, so only the chosen row groups are downloaded, never the whole
dump.

Sampling is two-stage, to avoid downloading ~1k row groups for 1k rows:
draw --n-groups row groups (probability proportional to their row count, without
replacement) across all shards, then --per-group random rows from each. Fully
determined by --seed.

Output (experiments/resample_sentences/data/):
    wiki_dump_pages.jsonl   one article per line: {"id", "url", "title", "text", "shard", "row_group", "row"}
    wiki_dump_manifest.json dump config, seed, and the (shard, row_group) pairs used

Other languages / output paths: --config 20231101.<lang> --out <path.jsonl>; the manifest then goes to
<stem>_manifest.json next to the output. Also importable: sample_dump(config, out_path, n_groups, per_group, seed).

Usage:
    python experiments/resample_sentences/sample_wiki_dump.py
    python experiments/resample_sentences/sample_wiki_dump.py --config 20231101.hi --out /path/wiki_dump_pages_hi.jsonl
"""
import argparse
import json
import random
import urllib.request
from pathlib import Path

import fsspec
import pyarrow.parquet as pq

HERE = Path(__file__).parent
OUT = HERE / "data"
DATASET, CONFIG = "wikimedia/wikipedia", "20231101.en"
DEFAULT_OUT = OUT / "wiki_dump_pages.jsonl"
DEFAULT_MANIFEST = OUT / "wiki_dump_manifest.json"


def shards_api(config: str) -> str:
    return f"https://huggingface.co/api/datasets/{DATASET}/parquet/{config}/train"


def manifest_path(out_path: Path) -> Path:
    out_path = Path(out_path)
    if out_path.resolve() == DEFAULT_OUT.resolve():
        return DEFAULT_MANIFEST
    return out_path.with_name(f"{out_path.stem}_manifest.json")


def sample_dump(config: str = CONFIG, out_path: Path = DEFAULT_OUT, n_groups: int = 50, per_group: int = 20,
                seed: int = 43) -> Path:
    """Sample n_groups row groups x per_group rows from wikimedia/wikipedia <config>; write jsonl + manifest."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    fs = fsspec.filesystem("http")

    with urllib.request.urlopen(shards_api(config)) as r:
        shard_urls = json.load(r)
    groups = []  # (shard, row_group, num_rows)
    for s, url in enumerate(shard_urls):
        md = pq.ParquetFile(fs.open(url, block_size=1 << 20)).metadata
        groups += [(s, g, md.row_group(g).num_rows) for g in range(md.num_row_groups)]
    total = sum(n for *_, n in groups)
    print(f"[dump {config}] {len(shard_urls)} shards, {len(groups)} row groups, {total} articles", flush=True)

    # Weighted sampling without replacement (Efraimidis-Spirakis keys).
    chosen = sorted(groups, key=lambda x: rng.random() ** (1 / x[2]), reverse=True)[:n_groups]
    chosen.sort()

    n_pages = 0
    tmp_path = out_path.with_name(out_path.name + ".part")
    with tmp_path.open("w", encoding="utf-8") as f:
        for i, (s, g, n) in enumerate(chosen, 1):
            table = pq.ParquetFile(fs.open(shard_urls[s], block_size=8 << 20)).read_row_group(g)
            for row in sorted(rng.sample(range(n), min(per_group, n))):
                rec = {k: table.column(k)[row].as_py() for k in ("id", "url", "title", "text")}
                rec.update(shard=s, row_group=g, row=row)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_pages += 1
            print(f"  group {i}/{len(chosen)}: shard {s} rg {g} -> {n_pages} pages", flush=True)
    tmp_path.replace(out_path)

    manifest_path(out_path).write_text(json.dumps({
        "dataset": DATASET, "config": config, "seed": seed, "n_groups": n_groups,
        "per_group": per_group, "total_articles": total, "n_pages": n_pages,
        "groups": [{"shard": s, "row_group": g, "num_rows": n} for s, g, n in chosen],
    }, indent=1))
    print(f"[done] {n_pages} pages -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=CONFIG, help="wikimedia/wikipedia config, e.g. 20231101.hi")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output jsonl (manifest written next to it as <stem>_manifest.json; "
                         "wiki_dump_manifest.json for the default path)")
    ap.add_argument("--n-groups", type=int, default=50)
    ap.add_argument("--per-group", type=int, default=20)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()
    sample_dump(args.config, args.out, args.n_groups, args.per_group, args.seed)


if __name__ == "__main__":
    main()
