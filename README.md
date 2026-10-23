# From EMBER to CINDER

Code, data and paper for **From EMBER to CINDER: Sharpen the Concept, Burn It Once in Eight Languages, Spare the Word**
(Semyon Guretskiy, Ori Petel and Mary Arav; NLP course project, Tel Aviv University, 2026).

EMBER ([Suslik et al., 2026](https://arxiv.org/abs/2606.03695)) erases a concept from a language model by decomposing
token embeddings into sparse features, selecting the concept's features with an LLM judge, and subtracting them from the
concept's tokens. CINDER keeps this framework and changes how the concept is found: it sharpens the embedding matrix
toward the concept against ordinary text, lets each feature take as many tokens as the data supports, and checks
features both bottom-up (EMBER's judge) and top-down (a lexicon generated once per concept). A single CINDER fit also
erases a concept in eight languages. We further show that EMBER's evaluation rewards edits that destroy the edited
words, and propose a collateral-aware evaluation.

**The paper is [`paper/main.pdf`](paper/main.pdf).** Its source is in [`paper/`](paper/); build it with
`sh paper/build.sh` (Tectonic) or upload the folder to Overleaf.

## Where everything is

| Paper | Code | Results |
|---|---|---|
| EMBER reproduction and the preliminary EMBER/SNMF study (App. A) | `ember/`, `configs/`, `run_*.sbatch`, `judge_rerun/` | `judge_rerun/*/README.md` |
| CINDER fits and feature selection: sharpening, global sparsity, two-way judge (§4, App. B–D) | `experiments/varwidth/` (`core.py`, `prep.py`, `select.py`), `experiments/whitening/sharpen.py` | `results/varwidth/layers/` (start with `SUMMARY.md`) |
| Collateral-aware evaluation (§5.1–5.2, §7, App. G) | `experiments/varwidth/` (`erase_v2.py`, `collateral.py`, `integrity.py`, `last_mile.py`, `nonce_edit.py`, `confirm_stats.py`) | `results/varwidth/layers/collateral/`; pre-registration in `results/varwidth/layers/PREREG.md` |
| Multilingual study on Llama-3.1-8B (§4.4, §5.3, App. F) | `experiments/ml/` | `results/ml/` (`PREREG_RESULTS.md`, `UNION_RESULTS.md`) |
| Embedding-space analyses (§6, App. E and H) | `experiments/paper_analysis/` | `results/paper_analysis/<analysis>/REPORT.md` |
| Figures | `paper/figures/src/` | `paper/figures/` |

App. J of the paper lists the same files, together with the compute and API costs.

Other folders:

- **Exploration notebooks.** The `ember_three_steps*.ipynb` and `mf_playground.ipynb` notebooks at the top level, and
  those in `low_ram_demands/` and `variable_width/`, are where we first ran EMBER step by step and tried the ideas that
  became CINDER.
- **Earlier pipelines and comparisons.** `cinder/` writes CINDER's features in EMBER's file format and ran the first
  CINDER-vs-EMBER strength searches; `Part 2 Experiments/` holds follow-up comparisons of the two methods;
  `crosslingual/`, `experiments/multilingual_fit/` and `experiments/multilingual_hp/` hold the first cross-lingual and
  multilingual experiments.
- **Design notes.** `CINDER_improvements.md` and `CINDER_collateral_evaluation.md` record the findings and decisions
  behind CINDER's components and the evaluation; `docs/` holds plans, progress notes and references; `report/` holds our
  earlier written reports.
- **Data.** `data/` holds the concept and neutral sentences and the evaluation questions, including the collateral
  questions; the multilingual data are in `experiments/ml/data/` and `experiments/ml/evaldata/`.
- `external/` holds the code of the MLP-based methods from EMBER's study as git submodules (SNMF, CRISP, WMDP, PISCES);
  `archive_gemini/` is earlier work kept for reference; `scripts/` and `logs/` hold utilities and cluster logs.

## Setup

- Python 3.12. `requirements.txt` is an exact snapshot of our environment (it matches `uv.lock`):
  `pip install -r requirements.txt` or `uv pip install -r requirements.txt`.
- Clone with `git clone --recurse-submodules` to get `external/`.
- Copy `.env.example` to `.env` and set `HF_TOKEN` (needed for the gated Llama-3.1-8B-Instruct) and `GEMINI_API_KEY`
  (feature judge, lexicons and the Alpaca coherence judge).
- The experiments ran on a SLURM cluster with NVIDIA TITAN Xp and RTX 2080 Ti GPUs; the `*.sbatch` files are the job
  scripts. Models: Gemma-2-2B-it (English) and Llama-3.1-8B-Instruct (multilingual; 4-bit transformer, fp16
  embeddings). The whole project spent about USD 15 on Gemini 3.5 Flash-Lite.
