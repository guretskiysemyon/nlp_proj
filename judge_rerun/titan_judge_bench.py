"""Benchmark the Qwen judge on TITAN Xp (Pascal, sm_61) under different settings.

Why this exists: on titan nodes the production judge (qwen_judge.QwenJudge,
NF4 with bf16 compute) ran ~1190 s/batch vs ~30 s on RTX 2080 Ti, and loading
the bf16 checkpoint from NFS took 2-4 h. Pascal has no native bf16, and GP102's
native fp16 math is also very slow, so the compute dtype is the prime suspect.

What it does, in ONE process (loading is the expensive part):
  1. Fails fast if CUDA is not visible (on s-002 jobs silently ran on CPU),
     and -- unless --allow-offload -- if any layer landed on cpu/disk.
  2. GEMM micro-benchmark per dtype (fp32/fp16/bf16) on the actual card.
  3. Loads the judge once (optionally from / into a pre-quantized NF4 copy,
     which is ~18 GB instead of the 62 GB bf16 checkpoint).
  4. For each --compute-dtypes x --batch-sizes config: switches the NF4
     layers' compute dtype in place, judges the first --n-items completions
     of a reference concept/delta with both prompts, and reports s/batch,
     generated tok/s, and rating agreement with the existing judged scores
     (produced by the production setup on RTX 2080 Ti).

Results are printed and appended as JSON lines to --results.

  python -m judge_rerun.titan_judge_bench --compute-dtypes fp32 fp16 bf16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ember.evals.gemini import GeminiEvaluator
from ember.utils import _safe_concept, _safe_model_name
from judge_rerun.qwen_judge import QwenJudge, _stub_out_bnb_triton, parse_rating

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--quantized-dir", default=None,
                   help="Pre-quantized NF4 copy. Loaded from if it exists; otherwise "
                        "the model is quantized from --judge-model and saved here.")
    p.add_argument("--load-dtype", choices=DTYPES, default="fp16",
                   help="compute/torch dtype used while loading (switched per config after)")
    p.add_argument("--no-double-quant", action="store_true",
                   help="disable NF4 double quantization (only applies when quantizing)")
    p.add_argument("--attn", choices=("sdpa", "eager"), default="sdpa")
    p.add_argument("--max-gpu-mem", default=None,
                   help='per-GPU cap for device_map="auto", e.g. "11GiB"')
    p.add_argument("--allow-offload", action="store_true",
                   help="don't abort if some layers land on cpu/disk")
    p.add_argument("--compute-dtypes", nargs="+", choices=DTYPES, default=["fp32", "fp16", "bf16"])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[4])
    p.add_argument("--n-items", type=int, default=8,
                   help="completions judged per config (x2 prompts)")
    p.add_argument("--max-new-tokens", type=int, default=350)  # QwenJudge.generate_batch default
    p.add_argument("--concept", default="Harry Potter")
    p.add_argument("--delta", default="1.0")
    p.add_argument("--model-name", default="google/gemma-2-2b-it")
    p.add_argument("--rank", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-gemm", action="store_true")
    p.add_argument("--p2p-patch", action="store_true",
                   help="route cuda->cuda copies via host (broken GPU peer path on titan nodes)")
    p.add_argument("--results", default=str(ROOT_DIR / "judge_rerun" / "titan_bench_results.jsonl"))
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [titan_bench] {msg}", flush=True)


def check_cuda() -> List[str]:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available (CUDA_VISIBLE_DEVICES="
                         f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}) -- refusing to run on CPU")
    names = []
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 2**30
        names.append(f"{torch.cuda.get_device_name(i)} sm_{cap[0]}{cap[1]} {mem:.1f}GiB")
        log(f"cuda:{i} {names[-1]}")
    return names


def gemm_bench(n: int = 4096, iters: int = 10) -> Dict[str, float]:
    """TFLOPS of an n x n matmul per dtype on cuda:0."""
    out = {}
    for name, dt in DTYPES.items():
        try:
            a = torch.randn(n, n, device="cuda:0", dtype=dt)
            b = torch.randn(n, n, device="cuda:0", dtype=dt)
            a @ b
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(iters):
                a @ b
            torch.cuda.synchronize()
            out[name] = round(2 * n**3 * iters / (time.time() - t0) / 1e12, 3)
        except Exception as e:  # noqa: BLE001 -- report and keep going
            out[name] = float("nan")
            log(f"gemm {name} failed: {e}")
        log(f"gemm {name}: {out[name]} TFLOPS")
    del a, b
    torch.cuda.empty_cache()
    return out


def load_judge(args: argparse.Namespace) -> QwenJudge:
    """A QwenJudge whose model/tokenizer are loaded per --quantized-dir etc."""
    _stub_out_bnb_triton()
    dt = DTYPES[args.load_dtype]
    max_memory = ({i: args.max_gpu_mem for i in range(torch.cuda.device_count())}
                  if args.max_gpu_mem else None)
    qdir = Path(args.quantized_dir) if args.quantized_dir else None
    from_saved = qdir is not None and (qdir / "config.json").exists()
    src = str(qdir) if from_saved else args.judge_model
    kwargs: Dict[str, Any] = dict(device_map="auto", max_memory=max_memory,
                                  torch_dtype=dt, attn_implementation=args.attn)
    if not from_saved:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dt,
            bnb_4bit_use_double_quant=not args.no_double_quant,
        )
    log(f"loading {src} ({'pre-quantized' if from_saved else 'quantizing on load'})")
    t0 = time.time()
    judge = QwenJudge.__new__(QwenJudge)  # skip __init__'s fixed bf16 config
    judge.tokenizer = AutoTokenizer.from_pretrained(src)
    judge.tokenizer.padding_side = "left"
    judge.model = AutoModelForCausalLM.from_pretrained(src, **kwargs)
    log(f"loaded in {(time.time() - t0) / 60:.1f} min")

    devmap = getattr(judge.model, "hf_device_map", {})
    placement: Dict[str, int] = {}
    for dev in devmap.values():
        placement[str(dev)] = placement.get(str(dev), 0) + 1
    log(f"device placement (modules per device): {placement}")
    if any(str(d) in ("cpu", "disk") for d in devmap.values()) and not args.allow_offload:
        raise SystemExit("some layers were offloaded to cpu/disk -- that is the slow path; "
                         "pass --allow-offload to benchmark it anyway")
    for i in range(torch.cuda.device_count()):
        log(f"cuda:{i} allocated {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB")

    first = next(iter(devmap.values())) if devmap else judge.model.device
    judge._input_device = f"cuda:{first}" if isinstance(first, int) else first

    if qdir is not None and not from_saved:
        log(f"saving pre-quantized copy to {qdir}")
        judge.model.save_pretrained(qdir)
        judge.tokenizer.save_pretrained(qdir)
    return judge


def set_compute_dtype(model: torch.nn.Module, dt: torch.dtype) -> None:
    """Switch NF4 layers' compute dtype and cast the non-quantized float params
    (embeddings, norms, lm_head) to match. Buffers (rotary inv_freq, fp32) and
    the 4-bit weights/quant states are left alone."""
    import bitsandbytes as bnb

    for module in model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            module.compute_dtype = dt
            module.compute_type_is_set = True  # else forward() re-derives it from the input
            if module.bias is not None:
                module.bias.data = module.bias.data.to(dt)
    for param in model.parameters():
        if type(param).__name__ != "Params4bit" and param.is_floating_point():
            param.data = param.data.to(dt)
    torch.cuda.empty_cache()


def run_config(judge: QwenJudge, prompts: Dict[str, List[str]],
               refs: Dict[str, List[Optional[int]]], batch_size: int,
               max_new_tokens: int) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    for kind, plist in prompts.items():
        batch_times: List[float] = []
        n_gen_tokens = 0
        responses: List[str] = []
        for s in range(0, len(plist), batch_size):
            chunk = plist[s:s + batch_size]
            torch.cuda.synchronize()
            t0 = time.time()
            out = judge._generate_chunk(chunk, max_new_tokens)
            torch.cuda.synchronize()
            batch_times.append(time.time() - t0)
            n_gen_tokens += sum(len(judge.tokenizer(o, add_special_tokens=False)["input_ids"])
                                for o in out)
            responses.extend(out)
            log(f"  {kind} batch {s // batch_size + 1}: {batch_times[-1]:.1f}s")
        scores: List[Optional[int]] = []
        for r in responses:
            try:
                scores.append(parse_rating(r))
            except ValueError:
                scores.append(None)
        pairs = [(a, b) for a, b in zip(scores, refs[kind]) if a is not None and b is not None]
        res[kind] = {
            "s_per_batch": round(sum(batch_times) / len(batch_times), 2),
            "gen_tok_per_s": round(n_gen_tokens / sum(batch_times), 2),
            "parse_failed": scores.count(None),
            "agree": f"{sum(a == b for a, b in pairs)}/{len(pairs)}",
            "mean": round(sum(a for a, _ in pairs) / len(pairs), 3) if pairs else None,
            "ref_mean": round(sum(b for _, b in pairs) / len(pairs), 3) if pairs else None,
            "scores": scores,
            "ref_scores": refs[kind],
            "first_response": responses[0][:300] if responses else "",
        }
    return res


def main() -> None:
    args = parse_args()
    if args.p2p_patch:
        from judge_rerun import titan_p2p_patch
        titan_p2p_patch.install()
        log("p2p patch installed")
    gpus = check_cuda()
    gemm = {} if args.skip_gemm else gemm_bench()

    run_rel = Path(_safe_model_name(args.model_name)) / f"rank{args.rank}" / f"seed{args.seed}"
    concept_dir = _safe_concept(args.concept)
    completions = json.loads((ROOT_DIR / "judge_rerun" / "pass1_outputs" / run_rel / concept_dir
                              / "alpaca_completions" / f"delta_{args.delta}.json").read_text())
    items = completions[:args.n_items]
    judged_path = (ROOT_DIR / "judge_rerun" / "pass2_outputs" / run_rel / concept_dir
                   / "alpaca_judged__qwen2.5-32b-instruct.json")
    ref_by_idx: Dict[Any, Dict[str, Any]] = {}
    if judged_path.exists():
        ref_by_idx = {r["idx"]: r for r in json.loads(judged_path.read_text()).get(args.delta, [])}
    if not ref_by_idx:
        log(f"no reference scores for {args.concept} delta {args.delta}; agreement will be empty")
    refs = {"instruct": [ref_by_idx.get(it["idx"], {}).get("instruct_score") for it in items],
            "fluency": [ref_by_idx.get(it["idx"], {}).get("fluency_score") for it in items]}
    prompts = {
        "instruct": [GeminiEvaluator.ALPACA_INSTRUCT_PROMPT.format(
            instructions=it["instruction"], completion=it["completion"]) for it in items],
        # --fluency-prompt original, as in run_pass2_delta_search_anygpu.sbatch
        "fluency": [GeminiEvaluator.ALPACA_FLUENCY_PROMPT.format(completion=it["completion"])
                    for it in items],
    }

    load_t0 = time.time()
    judge = load_judge(args)
    load_min = round((time.time() - load_t0) / 60, 1)

    results_path = Path(args.results)
    summary = []
    for dt_name in args.compute_dtypes:
        set_compute_dtype(judge.model, DTYPES[dt_name])
        for bs in args.batch_sizes:
            log(f"config: compute={dt_name} batch_size={bs} n_items={len(items)}")
            try:
                res = run_config(judge, prompts, refs, bs, args.max_new_tokens)
            except Exception as e:  # noqa: BLE001 -- one bad config shouldn't kill the sweep
                log(f"  FAILED: {type(e).__name__}: {e}")
                res = {"error": f"{type(e).__name__}: {e}"}
            record = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "slurm_job": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
                "gpus": gpus, "gemm_tflops": gemm, "judge_model": args.judge_model,
                "quantized_dir": args.quantized_dir, "load_min": load_min,
                "attn": args.attn, "compute_dtype": dt_name, "batch_size": bs,
                "concept": args.concept, "delta": args.delta, **res,
            }
            with results_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary.append((dt_name, bs, res))

    log("=" * 70)
    log(f"node={os.environ.get('SLURMD_NODENAME')} load={load_min} min gemm={gemm}")
    log("reference: production judge on RTX 2080 Ti ~ 30 s/batch (bs=4)")
    for dt_name, bs, res in summary:
        if "error" in res:
            log(f"compute={dt_name:4s} bs={bs}: {res['error']}")
            continue
        for kind in ("instruct", "fluency"):
            r = res[kind]
            log(f"compute={dt_name:4s} bs={bs} {kind:8s}: {r['s_per_batch']:7.1f} s/batch "
                f"{r['gen_tok_per_s']:6.1f} tok/s  agree={r['agree']} "
                f"mean={r['mean']} (ref {r['ref_mean']}) parse_failed={r['parse_failed']}")
    log(f"full records appended to {results_path}")


if __name__ == "__main__":
    main()
