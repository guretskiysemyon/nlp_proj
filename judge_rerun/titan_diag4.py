"""Sanity + speed of real generate() on the saved NF4 Qwen-32B, TITAN Xp, with p2p patch."""
import sys, time, torch, bitsandbytes as bnb
from judge_rerun.qwen_judge import _stub_out_bnb_triton
from judge_rerun import titan_p2p_patch
_stub_out_bnb_triton()
PATCH = '--no-patch' not in sys.argv
if PATCH: titan_p2p_patch.install()
DTS = [a for a in sys.argv[1:] if a in ('fp32','bf16','fp16')] or ['bf16','fp32']
from transformers import AutoModelForCausalLM, AutoTokenizer
Q = "/home/morg/NLP_2526b/semyong/.cache/qwen2.5-32b-instruct-nf4"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] [diag4] {m}", flush=True)
tok = AutoTokenizer.from_pretrained(Q); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(Q, device_map="auto", max_memory={i: "9GiB" for i in range(3)}, torch_dtype=torch.float16)
log(f"patch={PATCH} gpu={torch.cuda.get_device_name(0)} loaded; first Linear4bit quant_state dtype: {next(m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)).weight.quant_state.dtype}")
def setdt(dt):
    for m in model.modules():
        if isinstance(m, bnb.nn.Linear4bit): m.compute_dtype = dt; m.compute_type_is_set = True
    for p in model.parameters():
        if type(p).__name__ != "Params4bit" and p.is_floating_point(): p.data = p.data.to(dt)
def gen(prompts, n, label):
    w = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    inp = tok(w, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad(): out = model.generate(**inp, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize(); dt = time.time() - t
    log(f"{label}: {dt:.1f}s for {n} new tok ({dt/n:.2f} s/step); text={tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)!r}")
for name in DTS:
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]
    setdt(dt)
    gen(["What is the capital of France? Answer in one sentence."], 20, f"{name} bs=1 short")
    gen(["What is the capital of France? Answer in one sentence.", "Write a long essay about the history of the Roman Empire and its legacy in modern Europe, covering many topics in detail."], 20, f"{name} bs=2 padded")
    gen(["Rate this sentence from 0 to 2 and explain: 'The cat sat on the mat.' " * 20] * 4, 20, f"{name} bs=4 long prompt")
