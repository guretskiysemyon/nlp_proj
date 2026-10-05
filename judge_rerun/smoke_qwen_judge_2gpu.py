#!/usr/bin/env python3
"""Smoke test #3: same as smoke_qwen_judge_realistic.py but shards
Qwen2.5-14B-Instruct's layers across 2 GPUs on the SAME node (device_map=
"auto") instead of 1, to see how much headroom that buys and whether
cross-device activation hops meaningfully slow generation down.

Usage:
    python -m judge_rerun.smoke_qwen_judge_2gpu
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

ALPACA_INSTRUCT_PROMPT = """Please act as an impartial judge and evaluate whether the sentence fragment provided below is related to the instruction. Focus solely on the degree of relatedness in terms of topic, regardless of grammar, coherence, or informativeness.
Begin your evaluation by providing a brief explanation of whether the sentence is related to the instruction, and point out references related to the instruction. After providing your explanation, rate the instruction relevance on a scale from 0 to 2, where 0 indicates the sentence is unrelated to the instruction, 1 indicates it is somewhat related but only minimally or indirectly relevant in terms of topic, and 2 indicates it is more clearly and directly related to the instruction. If there is nothing between the sentence fragment start and end tags, the score is 0. Provide your rating using this exact format: "Rating: [[score]]".

[Instruction Start]
{instructions}
[Instruction End]

[Completion Start]
{completion}
[Completion End]"""

MAX_OUTPUT_TOKENS = 350


def parse_rating(response: str) -> int | None:
    m = re.search(r"Rating:\s*\[\[(\d+)\]\]", response)
    if not m:
        m = re.search(r"(\d+)", response)
    return int(m.group(1)) if m else None


def load_completions(concept_dir: str, delta_file: str, n: int = 150) -> list[dict]:
    p = OUT_ROOT / concept_dir / "alpaca_completions" / delta_file
    return json.loads(p.read_text())[:n]


def main() -> None:
    n_gpus = torch.cuda.device_count()
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} n_gpus={n_gpus}")
    for i in range(n_gpus):
        print(f"  gpu{i}: {torch.cuda.get_device_name(i)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(f"loaded in {time.time() - t0:.1f}s")
    print(f"device_map: {model.hf_device_map}")
    for i in range(n_gpus):
        print(f"  gpu{i} reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")

    first_device = next(iter(model.hf_device_map.values()))
    if isinstance(first_device, int):
        first_device = f"cuda:{first_device}"

    def generate(prompt: str, max_new_tokens: int = MAX_OUTPUT_TOKENS) -> tuple[str, float]:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(first_device)
        attn_mask = torch.ones_like(inputs)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                inputs, attention_mask=attn_mask, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        dt = time.time() - t0
        return tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True), dt

    # worst-case item across concepts, same as the 1-GPU test, for a fair comparison
    items = load_completions("Ancient_Rome", "delta_1.0.json")
    longest = max(items, key=lambda d: len(d["instruction"]) + len(d["completion"]))
    print(f"\nworst-case item: instr_chars={len(longest['instruction'])} "
          f"completion_chars={len(longest['completion'])}")

    cases = [("worst_case_instruct", ALPACA_INSTRUCT_PROMPT.format(
        instructions=longest["instruction"], completion=longest["completion"]))]
    for i, item in enumerate(load_completions("Nazism", "delta_1.0.json", n=3)):
        cases.append((f"Nazism_instruct_{i}", ALPACA_INSTRUCT_PROMPT.format(
            instructions=item["instruction"], completion=item["completion"])))

    n_parsed = 0
    for name, prompt in cases:
        text, dt = generate(prompt)
        rating = parse_rating(text)
        n_parsed += rating is not None
        mem_str = " ".join(f"gpu{i}={torch.cuda.memory_reserved(i)/1e9:.2f}GB"
                            for i in range(n_gpus))
        print(f"\n=== {name} ({dt:.1f}s, {mem_str}) ===")
        print(f"parsed_rating={rating}")
        print(text[:200].strip())

    print(f"\n{n_parsed}/{len(cases)} responses parsed cleanly")
    for i in range(n_gpus):
        print(f"peak gpu{i} reserved: {torch.cuda.max_memory_reserved(i) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
