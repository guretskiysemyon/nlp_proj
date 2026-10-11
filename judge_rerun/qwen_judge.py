"""Local Qwen2.5-14B-Instruct judge, loaded 4-bit (bitsandbytes NF4).

Reuses the project's own judge prompts and parsers verbatim
(ember.evals.gemini.GeminiEvaluator), so a Qwen-vs-Gemini agreement check
is meaningful and nothing about the rubric drifts between judges.

Not ember.evals.model_wrap.WrappedHFModel: that class hard-codes Gemma/Llama
chat-template handling and raises on any other tokenizer. Qwen's chat
template needs no such special-casing, so this is a small, separate,
generic batched-generation helper instead of bending that class to fit.
"""
from __future__ import annotations

import json
import re
import sys
import time
import types
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# The verbatim GeminiEvaluator.ALPACA_FLUENCY_PROMPT tells the judge once,
# in passing, to ignore a cut-off final sentence. On the pristine-model
# baseline run, Qwen2.5-14B-Instruct did not reliably follow that: its own
# stated reasoning repeatedly penalized completions for "ending abruptly"
# despite the instruction, dragging mean_fluency_score down to 1.15 vs
# Gemini's archived 1.613 -- a gap the project's own adoption gate
# (docs/judge_swap_plan.md SS6, +-0.15) would reject. Every Pass 1 completion
# is generated at max_new_tokens=200, so this is a real, frequent case, not
# an edge case -- worth a Qwen-specific, more forceful restatement of the
# same rule rather than changing what is being measured.
ALPACA_FLUENCY_PROMPT_QWEN = """Please act as an impartial judge and evaluate the fluency of the sentence fragment provided below. Focus solely on fluency, disregarding its completeness, relevance, coherence with any broader context, or informativeness.

IMPORTANT: This fragment was cut off by a generation length limit, not by its writer. The final sentence may end mid-word, mid-phrase, or mid-thought. This is EXPECTED on every fragment you will see and is NOT a fluency error. Do not mention the cutoff, call it abrupt, or lower the score because of it -- not even by one point. Judge fluency only on the text that IS present, as if it simply stopped at its last word.

Begin your evaluation by briefly describing the fluency of the sentence, noting any highly unnatural phrasing, very awkward transitions or grammatical errors that may hinder fluency. Note that markdown formatting should be considered fluent. After providing your explanation, rate the sentence's fluency on a scale from 0 to 2, where 0 indicates the sentence is not fluent and highly unnatural (for example incomprehensible or repetitive), 1 indicates it is somewhat fluent but contains noticeable errors or awkward phrasing, and 2 indicates the sentence is fluent and almost perfect. If there is nothing between the sentence fragment start and end tags, the score is 0. Provide your rating using this exact format: "Rating: [[score]]".

[Sentence Fragment Start]

{completion}

[Sentence Fragment End]

Reminder: the fragment ending abruptly or mid-sentence is expected and must not lower your score."""

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

# The prompts ask for "Rating: [[score]]", but Qwen2.5-32B-Instruct mostly
# writes "Rating: [2]" or "**Rating: [2]**" instead. The project's own
# GeminiEvaluator._parse_rating only accepts the double-bracket form and then
# falls back to the FIRST number anywhere in the response, so a response that
# mentions "14% tip" or "S&W 1006" before its rating got scored 14 / 1006.
_RATING_RE = re.compile(r"Rating:?\**\s*\**\[+\s*(\d+(?:\.\d+)?)\s*\]+")


def parse_rating(response: str, max_score: int = 2) -> int:
    """Last bracketed ``Rating: [n]`` / ``[[n]]`` in ``response``, as an int.

    Raises ValueError, rather than guessing, when there is no bracketed
    rating (e.g. the response was cut off at max_new_tokens), when it is not a
    whole number (``[[1.8]]``), or when it is outside 0..``max_score``. Callers
    count those as parse failures so they show up in the summary.
    """
    matches = _RATING_RE.findall(response)
    if not matches:
        raise ValueError(f"no bracketed 'Rating: [n]' in response: {response[-200:]!r}")
    value = float(matches[-1])
    if value != int(value) or not 0 <= value <= max_score:
        raise ValueError(f"rating {matches[-1]!r} is not an integer in 0..{max_score}")
    return int(value)


def _stub_out_bnb_triton() -> None:
    """bitsandbytes.nn eagerly imports its (unused, for plain NF4) Triton
    SwitchBack modules, which JIT-compile a C extension on first import.
    That needs python3-dev headers, present on the login node but not on
    this cluster's compute nodes -- so it crashes before bitsandbytes is
    even usable. Stub the submodule out; nothing in the NF4 path touches it.
    """
    if "bitsandbytes.nn.triton_based_modules" in sys.modules:
        return
    stub = types.ModuleType("bitsandbytes.nn.triton_based_modules")
    for name in ("StandardLinear", "SwitchBackLinear", "SwitchBackLinearGlobal",
                 "SwitchBackLinearVectorwise"):
        setattr(stub, name, type(name, (), {}))
    sys.modules["bitsandbytes.nn.triton_based_modules"] = stub


class QwenJudge:
    """Loads once; call ``generate_batch`` as many times as needed."""

    def __init__(self, model_id: str = MODEL_ID, device_map: str = "auto") -> None:
        _stub_out_bnb_triton()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"  # required for batched decoder-only generation
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
        first_device = next(iter(self.model.hf_device_map.values())) \
            if hasattr(self.model, "hf_device_map") else self.model.device
        self._input_device = f"cuda:{first_device}" if isinstance(first_device, int) else first_device

    def _generate_chunk(self, batch: List[str], max_new_tokens: int) -> List[str]:
        """Generate for one chunk, halving on OOM until it fits (floor: 1)."""
        try:
            wrapped = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True,
                )
                for p in batch
            ]
            inputs = self.tokenizer(
                wrapped, return_tensors="pt", padding=True, add_special_tokens=False,
            ).to(self._input_device)
            with torch.no_grad():
                gen_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            prompt_len = inputs["input_ids"].shape[1]
            return [
                self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
                for row in gen_ids
            ]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            print(f"[QwenJudge] OOM at batch_size={len(batch)}; "
                  f"retrying as {mid} + {len(batch) - mid}", flush=True)
            return (self._generate_chunk(batch[:mid], max_new_tokens)
                    + self._generate_chunk(batch[mid:], max_new_tokens))

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 350,
                       batch_size: int = 16, checkpoint_path: Path | str | None = None
                       ) -> List[str]:
        """Return one response string per prompt, same order as input.

        Adaptively halves a chunk's batch size on CUDA OOM and retries,
        rather than losing the whole run to one over-long prompt.

        If ``checkpoint_path`` is given, saves after every chunk and resumes
        from it if it already has a prefix of completed outputs -- a slow
        judge (e.g. a 32B model) can take hours per pass, and a SLURM
        time-limit kill mid-pass previously lost everything done so far.
        """
        outputs: List[str] = []
        ckpt = Path(checkpoint_path) if checkpoint_path else None
        if ckpt is not None and ckpt.exists():
            outputs = json.loads(ckpt.read_text())
            print(f"[QwenJudge] resuming from checkpoint: {len(outputs)}/{len(prompts)} done", flush=True)

        start = len(outputs)
        t0 = time.time()
        for chunk_start in range(start, len(prompts), batch_size):
            chunk = prompts[chunk_start:chunk_start + batch_size]
            outputs.extend(self._generate_chunk(chunk, max_new_tokens))
            if ckpt is not None:
                ckpt.write_text(json.dumps(outputs, ensure_ascii=False))
            done = len(outputs) - start
            elapsed = time.time() - t0
            print(f"[QwenJudge] {len(outputs)}/{len(prompts)} done, "
                  f"{elapsed / done:.1f}s/call, ~{elapsed / done * (len(prompts) - len(outputs)) / 60:.0f} min left",
                  flush=True)
        return outputs
