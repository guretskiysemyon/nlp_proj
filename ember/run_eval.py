#!/usr/bin/env python3
"""Run only the final evaluation (final_test) stage for a config.

Usage:
    python -m ember.run_eval --config configs/ember_gemma.yaml \
        --concepts "Harry Potter" --train-eval mc
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from ember.erasure.config import parse_args
from ember.erasure import log
from ember.erasure import pipeline
from ember.erasure import model_loader
from ember.erasure.methods import base as methods_base


def main() -> None:
    cfg = parse_args()
    log.configure()

    if cfg.features_source == "hf":
        from ember.data import ensure_features
        ensure_features(cfg.model_name, cfg.concepts, cfg.rank, cfg.seed)

    # Ensure final_test stage runs
    cfg.run_tests_after_train = True

    log.set_context(method=cfg.method)

    alpaca_bs = cfg.eval.alpaca_batch_size or 50

    hf_model, tokenizer = model_loader.load_hf_model(
        cfg.model_name, cfg.cache_dir,
        dtype=model_loader.pick_dtype("bf16"),
    )

    baselines = pipeline._compute_all_baselines(cfg, hf_model, tokenizer, alpaca_bs)
    best_embed_map = pipeline._resolve_ember_step(cfg, hf_model, tokenizer, baselines, alpaca_bs)

    method = methods_base.get(cfg.method)

    pipeline._run_final_test(method, cfg, hf_model, tokenizer, baselines, best_embed_map, alpaca_bs)


if __name__ == "__main__":
    main()
