#!/usr/bin/env python3
"""Realistic smoke test #2: load Qwen2.5-14B-Instruct in 4-bit once, then run
it through the project's ACTUAL Alpaca-instruct/fluency and open-QA judge
prompts (copied verbatim from ember/evals/gemini.py), using real saved
completions from Pass 1 -- including the single longest one on disk and a
couple of sensitive concepts (Nazism, Pornography) to check refusal
behaviour. Reports peak GPU memory and whether every response parses
cleanly with the project's own regexes.

Usage:
    python -m judge_rerun.smoke_qwen_judge_realistic
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

_stub = types.ModuleType("bitsandbytes.nn.triton_based_modules")
for _name in ("StandardLinear", "SwitchBackLinear", "SwitchBackLinearGlobal",
              "SwitchBackLinearVectorwise"):
    setattr(_stub, _name, type(_name, (), {}))
sys.modules["bitsandbytes.nn.triton_based_modules"] = _stub

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT_DIR / "judge_rerun" / "pass1_outputs" / "google_gemma-2-2b-it" / "rank100" / "seed42"

# Verbatim from ember/evals/gemini.py (GeminiEvaluator), so this test exercises
# the exact prompts + parsers the real judge will use.
ALPACA_INSTRUCT_PROMPT = """Please act as an impartial judge and evaluate whether the sentence fragment provided below is related to the instruction. Focus solely on the degree of relatedness in terms of topic, regardless of grammar, coherence, or informativeness.
Begin your evaluation by providing a brief explanation of whether the sentence is related to the instruction, and point out references related to the instruction. After providing your explanation, rate the instruction relevance on a scale from 0 to 2, where 0 indicates the sentence is unrelated to the instruction, 1 indicates it is somewhat related but only minimally or indirectly relevant in terms of topic, and 2 indicates it is more clearly and directly related to the instruction. If there is nothing between the sentence fragment start and end tags, the score is 0. Provide your rating using this exact format: "Rating: [[score]]".

[Instruction Start]
{instructions}
[Instruction End]

[Completion Start]
{completion}
[Completion End]"""

ALPACA_FLUENCY_PROMPT = """Please act as an impartial judge and evaluate the fluency of the sentence fragment provided below. Focus solely on fluency, disregarding its completeness, relevance, coherence with any broader context, or informativeness.
Begin your evaluation by briefly describing the fluency of the sentence, noting any highly unnatural phrasing, very awkward transitions or grammatical errors that may hinder fluency. Note that markdown formatting should be considered fluent. Ignore whether the last sentence is complete or not, it may be cut off and that should not affect your rating. After providing your explanation, rate the sentence's fluency on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural (for example incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect. If there is nothing between the sentence fragment start and end tags, the score is 0. Provide your rating using this exact format: "Rating: [[score]]".

[Sentence Fragment Start]

{completion}

[Sentence Fragment End]"""

MAX_OUTPUT_TOKENS = 350


def parse_rating(response: str) -> int | None:
    m = re.search(r"Rating:\s*\[\[(\d+)\]\]", response)
    if not m:
        m = re.search(r"(\d+)", response)
    return int(m.group(1)) if m else None


def load_completions(concept_dir: str, delta_file: str, n: int = 3) -> list[dict]:
    p = OUT_ROOT / concept_dir / "alpaca_completions" / delta_file
    data = json.loads(p.read_text())
    return data[:n]


def main() -> None:
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )
    print(f"loaded in {time.time() - t0:.1f}s, "
          f"reserved={torch.cuda.memory_reserved() / 1e9:.2f} GB")

    def generate(prompt: str, max_new_tokens: int = MAX_OUTPUT_TOKENS) -> tuple[str, float]:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        attn_mask = torch.ones_like(inputs)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                inputs, attention_mask=attn_mask, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        dt = time.time() - t0
        return tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True), dt

    # ---- the single longest instruction+completion on disk (worst case for memory) ---- #
    longest_concept_items = []
    longest_overall = None
    for concept_dir in ["Ancient_Rome", "Nazism", "Pornography"]:
        items = load_completions(concept_dir, "delta_1.0.json", n=150)
        for it in items:
            it["_concept"] = concept_dir
        longest = max(items, key=lambda d: len(d["instruction"]) + len(d["completion"]))
        longest_concept_items.append(longest)
    longest_overall = max(longest_concept_items,
                           key=lambda d: len(d["instruction"]) + len(d["completion"]))
    print(f"\nworst-case item: concept={longest_overall['_concept']} "
          f"instr_chars={len(longest_overall['instruction'])} "
          f"completion_chars={len(longest_overall['completion'])}")

    cases = []
    cases.append(("worst_case_instruct", ALPACA_INSTRUCT_PROMPT.format(
        instructions=longest_overall["instruction"], completion=longest_overall["completion"])))
    cases.append(("worst_case_fluency", ALPACA_FLUENCY_PROMPT.format(
        completion=longest_overall["completion"])))

    # ---- a few ordinary items per concept, including sensitive ones ---- #
    for concept_dir in ["Ancient_Rome", "Nazism", "Pornography"]:
        for i, item in enumerate(load_completions(concept_dir, "delta_1.0.json", n=2)):
            cases.append((f"{concept_dir}_instruct_{i}", ALPACA_INSTRUCT_PROMPT.format(
                instructions=item["instruction"], completion=item["completion"])))

    n_parsed = 0
    n_total = 0
    for name, prompt in cases:
        text, dt = generate(prompt)
        n_total += 1
        rating = parse_rating(text)
        n_parsed += rating is not None
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\n=== {name} ({dt:.1f}s, reserved={reserved:.2f} GB) ===")
        print(f"parsed_rating={rating}")
        print(text[:400].strip())

    print(f"\n{n_parsed}/{n_total} responses parsed cleanly")
    print(f"peak GPU mem reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB "
          f"(card has 11.26 GB)")


if __name__ == "__main__":
    main()
