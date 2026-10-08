"""Diagnose why the NF4 Qwen-32B judge is ~40-150x slower on TITAN Xp (sm_61).

Stage 1 (seconds): micro-benchmarks one Qwen-32B-sized Linear4bit and plain
  fp16/fp32 linears for decode (1 token) and prefill (~600 tokens) on cuda:0.
Stage 2 (--load): loads the pre-quantized NF4 copy and times single decode
  steps + a torch.profiler breakdown, for each compute dtype.
"""
import argparse, sys, time
import torch

from judge_rerun.qwen_judge import _stub_out_bnb_triton

DT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] [diag] {m}", flush=True)


def timeit(fn, n=5):
    fn(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n


def micro():
    import bitsandbytes as bnb
    log(f"torch {torch.__version__} bnb {bnb.__version__} cuda {torch.version.cuda} "
        f"dev {torch.cuda.get_device_name(0)} cap {torch.cuda.get_device_capability(0)}")
    inf, outf = 5120, 27648
    for tokens in (1, 4, 600):
        for name in ("fp32", "fp16", "bf16"):
            dt = DT[name]
            try:
                lin = bnb.nn.Linear4bit(inf, outf, bias=False, compute_dtype=dt,
                                        quant_type="nf4", compress_statistics=True,
                                        device="cpu").to("cuda:0")
                x = torch.randn(1, tokens, inf, device="cuda:0", dtype=dt)
                with torch.no_grad():
                    s = timeit(lambda: lin(x))
                log(f"Linear4bit  compute={name} tokens={tokens:4d}: {s*1e3:9.2f} ms")
                del lin
            except Exception as e:
                log(f"Linear4bit  compute={name} tokens={tokens}: FAILED {type(e).__name__}: {e}")
            if name != "bf16":
                w = torch.nn.Linear(inf, outf, bias=False, device="cuda:0", dtype=dt)
                x = torch.randn(1, tokens, inf, device="cuda:0", dtype=dt)
                with torch.no_grad():
                    s = timeit(lambda: w(x))
                log(f"plain Linear dtype={name}   tokens={tokens:4d}: {s*1e3:9.2f} ms")
                del w
            torch.cuda.empty_cache()


def model_stage(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import bitsandbytes as bnb
    _stub_out_bnb_triton()
    tok = AutoTokenizer.from_pretrained(a.quantized_dir)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.quantized_dir, device_map="auto",
                                                 torch_dtype=torch.float16)
    log(f"loaded in {(time.time()-t0)/60:.1f} min; devmap={ {str(v) for v in model.hf_device_map.values()} }")
    ids = tok("Explain what a neural network is. " * 40, return_tensors="pt").input_ids[:, :600].to("cuda:0")
    for name in a.dtypes:
        dt = DT[name]
        for m in model.modules():
            if isinstance(m, bnb.nn.Linear4bit):
                m.compute_dtype = dt; m.compute_type_is_set = True
        for p in model.parameters():
            if type(p).__name__ != "Params4bit" and p.is_floating_point():
                p.data = p.data.to(dt)
        with torch.no_grad():
            torch.cuda.synchronize(); t = time.time()
            out = model(ids, use_cache=True); torch.cuda.synchronize()
            log(f"compute={name} prefill 600 tok: {time.time()-t:.2f}s  finite={bool(torch.isfinite(out.logits).all())}")
            pkv = out.past_key_values; nxt = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(3):
                o = model(nxt, past_key_values=pkv, use_cache=True)
                pkv = o.past_key_values; nxt = o.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
            log(f"compute={name} decode step: {(time.time()-t)/3:.2f}s")
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                o = model(nxt, past_key_values=pkv, use_cache=True); torch.cuda.synchronize()
            log("profile of 1 decode step:\n" + prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=60))
            log("cpu-sorted:\n" + prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8, max_name_column_width=60))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--load", action="store_true")
    p.add_argument("--quantized-dir", default="/home/morg/NLP_2526b/semyong/.cache/qwen2.5-32b-instruct-nf4")
    p.add_argument("--p2p-patch", action="store_true")
    p.add_argument("--no-micro", action="store_true")
    p.add_argument("--dtypes", nargs="+", default=["fp16", "fp32"])
    a = p.parse_args()
    assert torch.cuda.is_available(), "no CUDA"
    _stub_out_bnb_triton()
    if a.p2p_patch:
        from judge_rerun import titan_p2p_patch; titan_p2p_patch.install(); log('p2p patch installed')
    if not a.no_micro:
        micro()
    if a.load:
        model_stage(a)
