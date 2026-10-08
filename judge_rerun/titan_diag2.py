"""Isolate the 'Command Buffer Full' launch stall seen with the full NF4 Qwen-32B on TITAN Xp.
Synthetic stack of Qwen-32B-shaped Linear4bit MLP/attn blocks (random weights, no model load)."""
import subprocess, sys, time
import torch
from judge_rerun.qwen_judge import _stub_out_bnb_triton
_stub_out_bnb_triton()
import bitsandbytes as bnb

def log(m): print(f"[{time.strftime('%H:%M:%S')}] [diag2] {m}", flush=True)
def smi(tag):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,clocks.sm,clocks.max.sm,power.draw,pstate,clocks_throttle_reasons.active,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    log(f"nvidia-smi {tag}:\n{out}")

H, I = 5120, 27648
def block(dev, dt):
    shapes = [(H, H), (H, 1024), (H, 1024), (H, H), (H, I), (H, I), (I, H)]  # q k v o gate up down
    return [bnb.nn.Linear4bit(i, o, bias=False, compute_dtype=dt, quant_type="nf4",
                              compress_statistics=True, device="cpu").to(dev) for i, o in shapes]

def step(blocks_by_dev, x):
    for dev, blocks in blocks_by_dev:
        x = x.to(dev)
        for b in blocks:
            q, k, v, o, g, u, d = b
            x = x + o(q(x))
            k(x); v(x)
            x = x + d(torch.nn.functional.silu(g(x)) * u(x))
    return x

def bench(name, layout, dt, n=3):
    x = torch.randn(1, 1, H, dtype=dt, device=layout[0][0])
    with torch.no_grad():
        step(layout, x); torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            step(layout, x)
        for d, _ in layout: torch.cuda.synchronize(d)
        log(f"{name} {dt}: {(time.time()-t)/n*1e3:.0f} ms/step  ({sum(len(b) for _, b in layout)} layers)")

smi("idle")
for dt in (torch.float32, torch.float16):
    one = [("cuda:0", [block("cuda:0", dt) for _ in range(20)])]
    bench("1 GPU, 20 layers", one, dt)
    smi("after 1-GPU")
    del one; torch.cuda.empty_cache()
dt = torch.float32
three = [(f"cuda:{i}", [block(f"cuda:{i}", dt) for _ in range(7)]) for i in range(3)]
bench("3 GPUs, 21 layers", three, dt)
smi("after 3-GPU")
