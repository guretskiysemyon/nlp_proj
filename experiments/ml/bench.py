#!/usr/bin/env python
"""Throughput probe for Llama-3.1-8B-Instruct on Titan Xp: fp16 over the visible GPUs vs NF4 on one GPU.

    python -m experiments.ml.bench --mode fp16|nf4|fp32
Prints load time, greedy 32-token generation time for a batch of MC-style prompts, and one perplexity forward pass.
"""
import argparse
import sys
import time
import types

import torch

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = ("Answer the multiple-choice question with the letter of the correct option.\n"
          "Question: Which wizarding school does the main character attend?\n"
          "A. Durmstrang\nB. Hogwarts\nC. Beauxbatons\nD. Ilvermorny\nAnswer:")


def stub_bnb():
    stub = types.ModuleType("bitsandbytes.nn.triton_based_modules")
    for n in ("StandardLinear", "SwitchBackLinear", "SwitchBackLinearGlobal", "SwitchBackLinearVectorwise"):
        setattr(stub, n, type(n, (), {}))
    sys.modules["bitsandbytes.nn.triton_based_modules"] = stub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fp16", choices=["fp16", "nf4", "fp32"])
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kw = {"device_map": "auto"}
    if a.mode == "nf4":
        stub_bnb()
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"] = {"": 0}
    else:
        kw["torch_dtype"] = torch.float16 if a.mode == "fp16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval()
    print(f"load {time.time() - t0:.0f}s  mode={a.mode}  gpus={torch.cuda.device_count()}  "
          f"emb dtype={model.get_input_embeddings().weight.dtype}", flush=True)
    msgs = [[{"role": "user", "content": PROMPT}]] * a.bs
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    for rep in range(2):
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        print(f"gen bs={a.bs} 32 tok: {time.time() - t:.1f}s  ->  "
              f"{tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)!r}", flush=True)
    sents = ["The quick brown fox jumps over the lazy dog near the river bank in the early morning."] * a.bs
    e2 = tok(sents, return_tensors="pt", padding=True).to(model.device)
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        loss = model(**e2, labels=e2.input_ids).loss
    torch.cuda.synchronize()
    print(f"ppl fwd bs={a.bs}: {time.time() - t:.2f}s loss={loss.item():.3f}", flush=True)
    for i in range(torch.cuda.device_count()):
        print(f"gpu{i} max mem {torch.cuda.max_memory_allocated(i) / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
