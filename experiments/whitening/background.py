#!/usr/bin/env python3
"""Backgrounds for whitening (Instructions.txt, "Sampling recipe") + concept pages and target token sets.

1. Wikipedia sentences: whole parquet shards of wikimedia/wikipedia 20231101.en (hf_hub_download, cached in the HF
   cache), shards in a seeded random order, downloaded only when the previous ones are exhausted. Articles of the
   downloaded shards are visited in a seeded random permutation; each contributes at most 4 random sentences
   (native_neutral.prose_paragraphs: cut at References/See also..., keep lines ending in sentence punctuation;
   fetch_en_sentences.split_sentences; native_neutral.looks_like_prose: citation/list/template-hole filters).
   Written to data/wiki_sentences.jsonl in chunks of 20k, shared by both models. Fallback: local dump pages.
2. Per model: unique token types (add_special_tokens=False) minus special/unused/reserved/byte tokens; after each
   chunk a split-half stability check (top-100 eigvecs, principal-angle sines). Sampling stops at the first chunk
   where every model has mean sine <= 0.1 and n_types >= min_types (30k gemma / 40k llama), or at 300k sentences.
3. Background stats (mu, lam, Q, n) for wiki / full / latin -> data/<model>/bg_<name>_stats.pt (+ id lists).
4. Concept pages -> data/pages/<slug>.json and targets -> data/<model>/targets/<slug>.json.

Resumable: every stage skips outputs that already exist.

Usage: python experiments/whitening/background.py [--models gemma llama]
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch

from common import (CONCEPTS, DATA, MODELS, ROOT, SEED, Embeddings, chunked_stats, excluded_ids, latin_filter, log,
                    load_env, model_dir, principal_angle_sines_top, read_json, slug, tokenizer, top_eigvecs,
                    write_json)

SENTS_PATH = DATA / "wiki_sentences.jsonl"
SENTS_META = DATA / "wiki_sentences_meta.json"
CHUNK = 20_000
MAX_SENTS = 300_000
PER_ARTICLE = 4
TOP_K = 100
SINE_TOL = 0.1
DATASET, CONFIG, N_SHARDS = "wikimedia/wikipedia", "20231101.en", 41
FALLBACK_PAGES = [ROOT / "experiments/resample_sentences/data/wiki_dump_pages.jsonl",
                  ROOT / "experiments/multilingual_hp/data/dump_ext/wiki_dump_pages_en.jsonl.part"]


# ---------------------------------------------------------------- sentence source
def article_sentences(text: str, key: str) -> list[str]:
    from fetch_en_sentences import split_sentences
    from native_neutral import looks_like_prose, prose_paragraphs
    sents = [s for s in split_sentences(prose_paragraphs(text or "", "en")) if looks_like_prose(s, "en")]
    if len(sents) > PER_ARTICLE:
        sents = random.Random(key).sample(sents, PER_ARTICLE)
    return sents


def download_shard(shard: int) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
    import os
    load_env()
    fname = f"{CONFIG}/train-{shard:05d}-of-{N_SHARDS:05d}.parquet"
    for attempt in range(8):
        try:
            t0 = time.time()
            p = hf_hub_download(DATASET, fname, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
            log(f"[wiki] shard {shard} ready ({time.time() - t0:.0f}s): {p}")
            return Path(p)
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code is not None and code not in (429,) and code < 500:
                raise
            wait = min(300, 10 * 2 ** attempt)
            log(f"[wiki] HTTP {code} on {fname}; retry in {wait}s")
            time.sleep(wait)
        except (OSError, ConnectionError) as e:
            wait = min(300, 10 * 2 ** attempt)
            log(f"[wiki] {type(e).__name__}: {e}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"download failed: {fname}")


class SentenceSource:
    """Deterministic stream of sentences; chunk c = sentences [c*CHUNK, (c+1)*CHUNK)."""

    def __init__(self):
        self.shard_order = random.Random(SEED).sample(range(N_SHARDS), N_SHARDS)
        self.n_shards_used = 0
        self.buffer: list[dict] = []  # pending sentences, in order
        self.source = "hf"

    def _load_next_shard(self):
        import pyarrow.parquet as pq
        if self.source == "fallback" or self.n_shards_used >= N_SHARDS:
            raise RuntimeError("sentence source exhausted")
        shard = self.shard_order[self.n_shards_used]
        try:
            path = download_shard(shard)
        except Exception as e:  # noqa: BLE001
            if self.n_shards_used:
                raise
            log(f"[wiki] download failed ({e}); falling back to local dump pages")
            return self._load_fallback()
        self.n_shards_used += 1
        t0 = time.time()
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        order = random.Random(f"{SEED}-perm-{shard}").sample(range(n), n)
        rank = {row: r for r, row in enumerate(order)}
        per_article: list = [None] * n
        base = 0
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=["id", "title", "text"])
            ids, titles, texts = (t.column(c).to_pylist() for c in ("id", "title", "text"))
            for j in range(len(ids)):
                row = base + j
                s = article_sentences(texts[j], f"{SEED}-{shard}-{row}")
                if s:
                    per_article[rank[row]] = (ids[j], titles[j], s)
            base += len(ids)
            del t, texts
        for item in per_article:
            if item:
                aid, title, ss = item
                self.buffer += [{"article_id": aid, "title": title, "shard": shard, "sentence": x} for x in ss]
        log(f"[wiki] shard {shard}: {n} articles -> {len(self.buffer)} buffered sentences "
            f"({time.time() - t0:.0f}s)")
        del per_article
        gc.collect()

    def _load_fallback(self):
        self.source = "fallback"
        pages = []
        for p in FALLBACK_PAGES:
            if p.exists():
                with p.open(encoding="utf-8") as f:
                    pages += [json.loads(line) for line in f if line.strip()]
        rng = random.Random(SEED)
        rng.shuffle(pages)
        for pg in pages:
            for x in article_sentences(pg.get("text"), f"{SEED}-fb-{pg['id']}"):
                self.buffer.append({"article_id": pg["id"], "title": pg["title"], "shard": "fallback", "sentence": x})
        log(f"[wiki] fallback: {len(pages)} pages -> {len(self.buffer)} sentences")

    def take(self, n: int) -> list[dict]:
        while len(self.buffer) < n:
            try:
                self._load_next_shard()
            except RuntimeError:
                break
        out, self.buffer = self.buffer[:n], self.buffer[n:]
        return out


class SentenceStore:
    """data/wiki_sentences.jsonl; regenerates the stream deterministically to append new chunks."""

    def __init__(self):
        self.chunks: list[list[str]] = []
        if SENTS_PATH.exists():
            by = {}
            with SENTS_PATH.open(encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    by.setdefault(r["chunk"], []).append(r["sentence"])
            self.chunks = [by[c] for c in sorted(by)]
        self._src = None

    def chunk(self, c: int) -> list[str] | None:
        while len(self.chunks) <= c:
            if self._src is None:
                self._src = SentenceSource()
                skip = sum(len(x) for x in self.chunks)
                if skip:
                    log(f"[wiki] regenerating stream, skipping {skip} stored sentences")
                    self._src.take(skip)
            rows = self._src.take(CHUNK)
            if not rows:
                return None
            k = len(self.chunks)
            with SENTS_PATH.open("a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps({"chunk": k, **r}, ensure_ascii=False) + "\n")
            self.chunks.append([r["sentence"] for r in rows])
            write_json(SENTS_META, {"dataset": DATASET, "config": CONFIG, "seed": SEED, "per_article": PER_ARTICLE,
                                    "chunk_size": CHUNK, "n_chunks": len(self.chunks),
                                    "n_sentences": sum(len(x) for x in self.chunks),
                                    "shard_order": self._src.shard_order[:max(1, self._src.n_shards_used)],
                                    "n_shards_used": self._src.n_shards_used, "source": self._src.source})
            log(f"[wiki] chunk {k}: {len(rows)} sentences (total {sum(len(x) for x in self.chunks)})")
        return self.chunks[c]


# ---------------------------------------------------------------- per-model stability
def chunk_types(model: str, store: SentenceStore, c: int) -> list[int] | None:
    path = model_dir(model) / "chunk_types.json"
    cache = read_json(path) if path.exists() else {}
    if str(c) in cache:
        return cache[str(c)]
    sents = store.chunk(c)
    if sents is None:
        return None
    tok = tokenizer(model)
    ex = excluded_ids(model)
    ids = set()
    for i in range(0, len(sents), 2000):
        for seq in tok(sents[i:i + 2000], add_special_tokens=False)["input_ids"]:
            ids.update(seq)
    cache[str(c)] = sorted(int(i) for i in ids if i not in ex)
    write_json(path, cache, indent=None)
    return cache[str(c)]


def stability_point(model: str, emb: Embeddings, types: list[int], c: int, n_sent: int) -> dict:
    rng = random.Random(SEED)
    perm = rng.sample(types, len(types))
    h = len(perm) // 2
    sa, sb = chunked_stats(emb, perm[:h]), chunked_stats(emb, perm[h:2 * h])
    sines = principal_angle_sines_top(sa, sb, TOP_K)
    return {"chunk": c, "n_sentences": n_sent, "n_types": len(types), "mean_sine": float(sines.mean()),
            "max_sine": float(sines.max()), **extra_diag(sa, sb)}


def extra_diag(sa: dict, sb: dict) -> dict:
    """Diagnostics beyond the stop rule: smaller subspaces, and top-100 of one half inside top-150 of the other
    (tolerates eigenvector swaps across the k=100 boundary, where the eigengap is small)."""
    from sharpen import principal_angle_sines
    out = {f"mean_sine_top{k}": float(principal_angle_sines_top(sa, sb, k).mean()) for k in (10, 50)}
    ab = principal_angle_sines(top_eigvecs(sa, TOP_K), top_eigvecs(sb, 150))
    ba = principal_angle_sines(top_eigvecs(sb, TOP_K), top_eigvecs(sa, 150))
    out["mean_sine_100_in_150"] = float((ab.mean() + ba.mean()) / 2)
    out["max_sine_100_in_150"] = float(max(ab.max(), ba.max()))
    return out


def backfill_diag(model: str, store: "SentenceStore"):
    spath = model_dir(model) / "stability.json"
    st = read_json(spath)
    if all("mean_sine_100_in_150" in p for p in st["curve"]):
        return
    emb = Embeddings(model)
    try:
        types: set[int] = set()
        for p in st["curve"]:
            types.update(chunk_types(model, store, p["chunk"]))
            if "mean_sine_100_in_150" in p:
                continue
            perm = random.Random(SEED).sample(sorted(types), len(types))
            h = len(perm) // 2
            p.update(extra_diag(chunked_stats(emb, perm[:h]), chunked_stats(emb, perm[h:2 * h])))
            write_json(spath, st)
            log(f"[{model}] diag chunk {p['chunk']}: top10={p['mean_sine_top10']:.3f} top50={p['mean_sine_top50']:.3f}"
                f" 100in150={p['mean_sine_100_in_150']:.3f} (max {p['max_sine_100_in_150']:.3f})")
    finally:
        emb.free()


def run_stability(model: str, store: SentenceStore, until_chunk: int | None) -> int:
    """Extend the stability curve of `model`. until_chunk=None: stop at this model's own criterion."""
    spath = model_dir(model) / "stability.json"
    st = read_json(spath) if spath.exists() else {"curve": [], "stop_chunk": None}
    done = {p["chunk"]: p for p in st["curve"]}
    emb = None
    types: set[int] = set()
    n_sent = 0
    max_chunks = MAX_SENTS // CHUNK
    c = 0
    try:
        while c < max_chunks:
            ct = chunk_types(model, store, c)
            if ct is None:
                log(f"[{model}] sentence source exhausted at chunk {c}")
                break
            types.update(ct)
            n_sent += len(store.chunk(c))
            if c not in done:
                if emb is None:
                    emb = Embeddings(model)
                t0 = time.time()
                p = stability_point(model, emb, sorted(types), c, n_sent)
                done[c] = p
                st["curve"] = [done[k] for k in sorted(done)]
                write_json(spath, st)
                log(f"[{model}] chunk {c}: sents={n_sent} types={p['n_types']} mean_sine={p['mean_sine']:.4f} "
                    f"max_sine={p['max_sine']:.4f} ({time.time() - t0:.0f}s)")
            p = done[c]
            ok = p["mean_sine"] <= SINE_TOL and p["n_types"] >= MODELS[model]["min_types"]
            if ok and st.get("stop_chunk") is None:
                st["stop_chunk"] = c
                write_json(spath, st)
            if until_chunk is None and ok:
                break
            if until_chunk is not None and c >= until_chunk:
                break
            c += 1
        else:
            c = max_chunks - 1
    finally:
        if emb is not None:
            emb.free()
    if st.get("stop_chunk") is None:
        log(f"[{model}] criterion not met by chunk {c}")
    return st.get("stop_chunk") if st.get("stop_chunk") is not None else c


def plot_stability(model: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = read_json(model_dir(model) / "stability.json")
    cv = st["curve"]
    x = [p["n_sentences"] / 1000 for p in cv]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(x, [p["mean_sine"] for p in cv], marker="o", label="mean sine")
    ax.plot(x, [p["max_sine"] for p in cv], marker=".", ls="--", label="max sine")
    if all("mean_sine_100_in_150" in p for p in cv):
        ax.plot(x, [p["mean_sine_100_in_150"] for p in cv], marker="s", ms=3, color="tab:purple", label="mean sine, top-100 in top-150")
        ax.plot(x, [p["mean_sine_top10"] for p in cv], marker="^", ms=3, label="mean sine, top-10")
    ax.axhline(SINE_TOL, color="gray", lw=0.8)
    ax.set_xlabel("sentences (k)")
    ax.set_ylabel(f"principal-angle sine (top {TOP_K})")
    ax2 = ax.twinx()
    ax2.plot(x, [p["n_types"] / 1000 for p in cv], color="tab:green", alpha=0.6, label="types (k)")
    ax2.set_ylabel("token types (k)", color="tab:green")
    ax.set_title(f"{model}: split-half stability of the wiki background")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(model_dir(model) / "stability.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- backgrounds
def build_backgrounds(model: str, n_chunks: int, store: SentenceStore):
    md = model_dir(model)
    types = set()
    for c in range(n_chunks):
        types.update(chunk_types(model, store, c))
    wiki = sorted(types)
    if not (md / "bg_wiki_types.json").exists() or read_json(md / "bg_wiki_types.json") != wiki:
        write_json(md / "bg_wiki_types.json", wiki, indent=None)
        (md / "bg_wiki_stats.pt").unlink(missing_ok=True)
    ex = excluded_ids(model)
    todo = [n for n in ("wiki", "full", "latin") if not (md / f"bg_{n}_stats.pt").exists()]
    if not todo:
        return
    emb = Embeddings(model)
    try:
        full = [i for i in range(emb.V) if i not in ex]
        lists = {"wiki": wiki, "full": full, "latin": latin_filter(model, full)}
        write_json(md / "bg_full_types.json", lists["full"], indent=None)
        write_json(md / "bg_latin_types.json", lists["latin"], indent=None)
        for name in todo:
            t0 = time.time()
            st = chunked_stats(emb, lists[name])
            st["n_tokens"] = len(lists[name])
            st["ids_file"] = f"bg_{name}_types.json"
            torch.save(st, md / f"bg_{name}_stats.pt")
            log(f"[{model}] bg_{name}: n={st['n']} mean(lam)={st['lam'].mean():.4g} top lam={st['lam'][-1]:.4g} "
                f"({time.time() - t0:.0f}s)")
    finally:
        emb.free()


# ---------------------------------------------------------------- concept pages and targets
PAGE_SOURCES = {
    "Harry Potter": ROOT / "experiments/multilingual_hp/data/raw/en.json",
    "Valentine's Day": ROOT / "experiments/multilingual_hp/data/concepts/Valentine_s_Day/raw/en.json",
    "Ancient Rome": None,
}


def build_pages():
    from fetch_en_sentences import _ParagraphText, page_paragraphs, split_sentences
    pages = DATA / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    for concept in CONCEPTS:
        out = pages / f"{slug(concept)}.json"
        if out.exists():
            continue
        src = PAGE_SOURCES[concept]
        if src is not None:
            raw = read_json(src)
            assert raw["lang"] == "en" and raw["title"] == concept, (raw["lang"], raw["title"])
            p = _ParagraphText()
            p.feed(raw["html"])
            title, revid, paras = raw["title"], raw["revid"], p.paragraphs
        else:
            raw_path = pages / f"{slug(concept)}_paragraphs.json"
            if raw_path.exists():
                r = read_json(raw_path)
            else:
                title, revid, paras = page_paragraphs(concept)
                r = {"title": title, "revid": revid, "paragraphs": paras}
                write_json(raw_path, r)
            title, revid, paras = r["title"], r["revid"], r["paragraphs"]
        sents = split_sentences(paras)
        write_json(out, {"concept": concept, "title": title, "revid": revid, "sentences": sents})
        log(f"[page] {concept}: '{title}' rev {revid}: {len(sents)} sentences")


def build_targets(model: str):
    md = model_dir(model) / "targets"
    md.mkdir(parents=True, exist_ok=True)
    wiki = set(read_json(model_dir(model) / "bg_wiki_types.json"))
    ex = excluded_ids(model)
    tok = tokenizer(model)
    sizes = {}
    for concept in CONCEPTS:
        out = md / f"{slug(concept)}.json"
        page = read_json(DATA / "pages" / f"{slug(concept)}.json")
        ids = set()
        for seq in tok(page["sentences"], add_special_tokens=False)["input_ids"]:
            ids.update(seq)
        pg = sorted(int(i) for i in ids if i not in ex)
        tgt = {"page": pg, "page_minus_wiki": [i for i in pg if i not in wiki]}
        write_json(out, tgt, indent=None)
        sizes[concept] = {k: len(v) for k, v in tgt.items()}
    write_json(model_dir(model) / "targets" / "sizes.json", sizes)
    log(f"[{model}] target sizes: {sizes}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma", "llama"])
    args = ap.parse_args()
    t_all = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    store = SentenceStore()
    stops = {}
    for m in args.models:
        t0 = time.time()
        stops[m] = run_stability(m, store, None)
        log(f"[{m}] own stop chunk {stops[m]} ({time.time() - t0:.0f}s)")
    final = max(stops.values())
    for m in args.models:  # both models use the same sentence set: extend curves to the common stop
        run_stability(m, store, final)
        backfill_diag(m, store)
        plot_stability(m)
    write_json(DATA / "wiki_final.json", {"n_chunks": final + 1,
                                          "n_sentences": sum(len(store.chunk(c)) for c in range(final + 1)),
                                          "own_stop_chunk": stops})
    for m in args.models:
        t0 = time.time()
        build_backgrounds(m, final + 1, store)
        log(f"[{m}] backgrounds done ({time.time() - t0:.0f}s)")
    build_pages()
    for m in args.models:
        build_targets(m)
    log(f"[done] background.py in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
