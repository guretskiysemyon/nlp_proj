#!/usr/bin/env python3
"""One-off smoke test: does Qwen2.5-14B-Instruct, loaded 4-bit (bitsandbytes
NF4), fit on an 11 GB RTX 2080 Ti and produce sane judge-style outputs?

Not part of the Pass 2 pipeline -- just answers "does this model/quant combo
work on this GPU" before any real judge code gets written.

Usage:
    python -m judge_rerun.smoke_qwen_judge
"""
from __future__ import annotations

import os
import sys
import time
import types

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# bitsandbytes.nn unconditionally imports its Triton-based SwitchBack modules,
# which are irrelevant to plain NF4 4-bit loading but JIT-compile a small C
# extension via Triton on first import. That compile needs python3-dev
# headers, which the compute nodes on this cluster don't have (only the
# login node does) -- so it crashes before bitsandbytes is even usable.
# Stub the submodule out so the real triton_based_modules.py is never
# executed; nothing in the NF4 load/generate path touches these classes.
_stub = types.ModuleType("bitsandbytes.nn.triton_based_modules")
for _name in ("StandardLinear", "SwitchBackLinear", "SwitchBackLinearGlobal",
              "SwitchBackLinearVectorwise"):
    setattr(_stub, _name, type(_name, (), {}))
sys.modules["bitsandbytes.nn.triton_based_modules"] = _stub

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

ALPACA_PROMPT = """Below is an instruction and a response. Rate the response's \
helpfulness on a 0-2 scale (0=bad, 1=ok, 2=great). End with exactly: Rating: [[x]]

Instruction: {instruction}
Response: {completion}"""

OPEN_QA_PROMPT = """Question: {question}
Gold answer: {gold}
Attempted answer: {attempted}

Is the attempted answer correct relative to the gold answer? Answer with \
exactly one word, Yes or No."""


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
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
    )
    print(f"loaded in {time.time() - t0:.1f}s")
    print(f"GPU mem allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB  "
          f"reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    cases = [
        ("alpaca", ALPACA_PROMPT.format(
            instruction="Explain why the sky is blue.",
            completion="The sky is blue because of Rayleigh scattering: shorter "
                       "(blue) wavelengths of sunlight scatter more in the "
                       "atmosphere than longer (red) wavelengths, so blue light "
                       "reaches our eyes from all directions.")),
        ("alpaca_bad", ALPACA_PROMPT.format(
            instruction="Explain why the sky is blue.",
            completion="Banana purple seventeen because yes.")),
        ("open_qa_correct", OPEN_QA_PROMPT.format(
            question="What house did Harry Potter belong to at Hogwarts?",
            gold="Gryffindor",
            attempted="He was a Gryffindor student.")),
        ("open_qa_wrong", OPEN_QA_PROMPT.format(
            question="What house did Harry Potter belong to at Hogwarts?",
            gold="Gryffindor",
            attempted="He was in Slytherin.")),
    ]

    for name, prompt in cases:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                inputs, max_new_tokens=30, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
        print(f"\n=== {name} ({time.time() - t0:.1f}s) ===\n{text.strip()!r}")

    print(f"\npeak GPU mem reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
