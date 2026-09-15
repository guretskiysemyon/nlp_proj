"""Locate and apply a config's already-chosen best hyper-parameter.

This module never searches: it reads the HP that the *English* grid search
already selected and applies it once. Re-running a grid here would defeat the
purpose of the whole experiment (the edit must be the one that was tuned in
English, so that any cross-lingual difference is attributable to the language
and not to a fresh tuning run).

Config -> method mapping (see ``ember/erasure/config.py``'s ``METHODS`` and
``ember/erasure/io.py``'s ``method_dir_name``)::

    pristine    no HP, no apply() -- just the loaded model
    ember       method "ember",  results/ember/...
    snmf        method "snmf" with ember_step.enabled=False -> results/snmf/...
    snmf_ember  method "snmf" with ember_step.enabled=True  -> results/snmf_ef/...

There is no separate ``snmf_ember`` method class; ``_ef`` is only a directory
suffix. For that config the SNMF HP comes from ``results/snmf_ef/.../hps.csv``
(or ``top_hps_valid.csv``) while ``delta_embed`` comes from the *ember* run's
``best_embed.csv`` and is merged into the HP dict before ``apply()`` -- exactly
what ``pipeline._run_method_grid`` does with
``if not method_is_ember and "delta_embed" not in hp``.

``--results-root`` is honored by rebasing the path that
:func:`ember.erasure.io.train_dir` builds onto the requested root, so the
``ember`` path helpers stay the single source of truth for the layout while the
tree itself can live anywhere (it is gitignored, so a fresh checkout has none
of it -- see §0 of the spec).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ember.erasure import io as ember_io
from ember.erasure import pipeline as ember_pipeline
from ember.erasure.config import RunConfig, load_yaml
from ember.erasure.methods import base as methods_base

CONFIGS: tuple[str, ...] = ("pristine", "ember", "snmf", "snmf_ember")

#: config -> (method name, YAML basename stem, ember_step.enabled)
_CONFIG_SPEC: Dict[str, tuple[str, str, bool]] = {
    "ember":      ("ember", "ember",      False),
    "snmf":       ("snmf",  "snmf",       False),
    "snmf_ember": ("snmf",  "snmf_ember", True),
}

CONFIG_ROOT = ember_io.ROOT_DIR


def check_config(config: str) -> str:
    if config not in CONFIGS:
        raise ValueError(f"Unknown config {config!r}; expected one of {CONFIGS}")
    return config


def order_configs(configs: List[str]) -> List[str]:
    """Return ``configs`` with ``pristine`` first.

    The per-language baselines must be computed on an unedited model, so when
    the pristine config is being run anyway its model load is reused for them.
    """
    return sorted(configs, key=lambda c: (c != "pristine", CONFIGS.index(c)))


def _model_yaml_suffix(model_name: str) -> str:
    return "llama" if "llama" in model_name.lower() else "gemma"


def yaml_path_for(config: str, model_name: str,
                  configs_dir: Optional[Path] = None) -> Path:
    """The YAML the original English grid search used for this config."""
    check_config(config)
    if config == "pristine":
        raise ValueError("config 'pristine' has no YAML: nothing is applied")
    _, stem, _ = _CONFIG_SPEC[config]
    root = configs_dir or (CONFIG_ROOT / "configs")
    return root / f"{stem}_{_model_yaml_suffix(model_name)}.yaml"


# ========================================================================== #
# Path resolution (results-root aware)                                        #
# ========================================================================== #

def _rebase(path: Path, results_root: Optional[Path]) -> Path:
    """Move ``path`` from the canonical ``results/`` tree onto ``results_root``."""
    if results_root is None:
        return path
    root = Path(results_root).expanduser()
    try:
        rel = Path(path).relative_to(ember_io.RESULTS_DIR)
    except ValueError:
        return path
    return root / rel


def train_dir_for(config: str, model_name: str, rank: int, seed: int,
                  train_eval: str, results_root: Optional[Path] = None) -> Path:
    """``<results_root>/<method[_ef]>/<model_safe>/rank<R>/seed<S>/train_<mode>``."""
    method, _, embed_step = _CONFIG_SPEC[check_config(config)]
    return _rebase(
        ember_io.train_dir(method, model_name, rank, seed, train_eval,
                           embed_step_enabled=embed_step),
        results_root,
    )


def ember_best_embed_csv(model_name: str, rank: int, seed: int, train_eval: str,
                         results_root: Optional[Path] = None) -> Path:
    """The ember run's ``best_embed.csv`` -- the source of ``delta_embed``."""
    return _rebase(
        ember_io.train_dir("ember", model_name, rank, seed, train_eval)
        / "best_embed.csv",
        results_root,
    )


@dataclass
class HPLocation:
    """Where a config's already-chosen HP lives, and whether it is actually there."""
    config: str
    method: str
    embed_step_enabled: bool
    yaml_path: Optional[Path]
    hp_csv: Optional[Path]
    valid_csv: Optional[Path]
    best_embed_csv: Optional[Path]
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def locate(config: str, concept: str, *, model_name: str, rank: int, seed: int,
           train_eval: str, results_root: Optional[Path] = None,
           configs_dir: Optional[Path] = None) -> HPLocation:
    """Resolve (and validate the presence of) every input needed to apply ``config``."""
    check_config(config)
    if config == "pristine":
        return HPLocation(config="pristine", method="", embed_step_enabled=False,
                          yaml_path=None, hp_csv=None, valid_csv=None,
                          best_embed_csv=None)

    method, _, embed_step = _CONFIG_SPEC[config]
    yaml_path = yaml_path_for(config, model_name, configs_dir)
    tdir = train_dir_for(config, model_name, rank, seed, train_eval, results_root)
    cdir = ember_io.concept_dir(tdir, concept)
    hp_csv = cdir / "hps.csv"
    valid_csv = cdir / "top_hps_valid.csv"
    best_embed = (ember_best_embed_csv(model_name, rank, seed, train_eval, results_root)
                  if embed_step else None)

    problems: List[str] = []
    if not yaml_path.exists():
        problems.append(f"missing grid YAML: {yaml_path}")
    if not hp_csv.exists() and not valid_csv.exists():
        problems.append(
            f"no HP cache for this concept: neither {hp_csv} nor {valid_csv} exists"
        )
    elif not hp_csv.exists():
        problems.append(f"missing {hp_csv} (falling back to {valid_csv.name})")
    if best_embed is not None:
        if not best_embed.exists():
            problems.append(f"missing ember best_embed.csv: {best_embed}")
        elif concept not in ember_io.load_best_embed_map(best_embed):
            problems.append(
                f"{best_embed} has no row for concept {concept!r}; "
                f"snmf_ember cannot resolve delta_embed"
            )
    return HPLocation(config=config, method=method, embed_step_enabled=embed_step,
                      yaml_path=yaml_path, hp_csv=hp_csv, valid_csv=valid_csv,
                      best_embed_csv=best_embed, problems=problems)


# ========================================================================== #
# RunConfig + HP resolution                                                   #
# ========================================================================== #

def load_run_config(config: str, concept: str, *, model_name: str, rank: int,
                    seed: int, train_eval: str,
                    configs_dir: Optional[Path] = None) -> RunConfig:
    """Load the same YAML the original grid used, narrowed to one concept.

    ``method.apply`` reads grid-config fields beyond the HP dict itself (e.g.
    ``cfg.snmf.w_mode``, ``cfg.selection.ratio_thresh``), so the config must be
    the run's own -- inventing a fresh one risks applying the right HP numbers
    through the wrong machinery.
    """
    cfg = load_yaml(yaml_path_for(config, model_name, configs_dir))
    cfg.concepts = [concept]
    cfg.train_eval = train_eval
    cfg.model_name = model_name
    cfg.rank = rank
    cfg.seed = seed
    _, _, embed_step = _CONFIG_SPEC[check_config(config)]
    cfg.ember_step.enabled = embed_step
    cfg.validate()
    if cfg.method != _CONFIG_SPEC[config][0]:
        raise ValueError(
            f"{yaml_path_for(config, model_name, configs_dir)} declares "
            f"method={cfg.method!r} but config {config!r} expects "
            f"{_CONFIG_SPEC[config][0]!r}"
        )
    if cfg.ember_step.enabled != embed_step:
        raise ValueError(f"config {config!r} expects ember_step.enabled={embed_step}")
    return cfg


def resolve_hp(config: str, concept: str, *, model_name: str, rank: int, seed: int,
               train_eval: str, results_root: Optional[Path] = None,
               configs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Return the HP dict to hand to ``method.apply`` for this config + concept.

    Selection is delegated to :func:`ember.erasure.pipeline._pick_best_hp_for_test`
    (``top_hps_valid.csv`` by ``harmonic_alpaca`` when present, else the
    method's own ``pick_best_hp_row`` over ``hps.csv``), so the HP chosen here
    is the same row the English final-test stage would have chosen.
    """
    loc = locate(config, concept, model_name=model_name, rank=rank, seed=seed,
                 train_eval=train_eval, results_root=results_root,
                 configs_dir=configs_dir)
    if loc.hp_csv is None:
        raise ValueError(f"config {config!r} has no HP to resolve")
    if not loc.hp_csv.exists() and not (loc.valid_csv and loc.valid_csv.exists()):
        raise FileNotFoundError(
            "; ".join(loc.problems) or f"no HP cache at {loc.hp_csv}"
        )

    method = methods_base.get(loc.method)
    best_row = ember_pipeline._pick_best_hp_for_test(method, loc.hp_csv, loc.valid_csv)
    if best_row is None:
        raise ValueError(
            f"no usable HP row for concept {concept!r} in {loc.hp_csv} "
            f"(the grid search may not have completed for this concept)"
        )

    hp: Dict[str, Any] = {c: best_row[c] for c in method.hp_columns() if c in best_row}

    if loc.method != "ember" and "delta_embed" not in hp:
        hp["delta_embed"] = _delta_embed_for(loc, concept)
    if loc.method != "ember":
        # An _ef run's own hps.csv already carries the ember delta, but a NaN
        # cell (or a plain-snmf row) must still resolve to a concrete number.
        if pd.isna(hp.get("delta_embed", float("nan"))):
            hp["delta_embed"] = _delta_embed_for(loc, concept)
    return hp


def _delta_embed_for(loc: HPLocation, concept: str) -> float:
    """``delta_embed`` for a non-ember config: the ember run's best, or 0.0."""
    if not loc.embed_step_enabled:
        return 0.0  # plain snmf never applies an embedding edit
    if loc.best_embed_csv is None or not loc.best_embed_csv.exists():
        raise FileNotFoundError(
            f"config {loc.config!r} needs the ember delta but "
            f"{loc.best_embed_csv} does not exist"
        )
    mapping = ember_io.load_best_embed_map(loc.best_embed_csv)
    if concept not in mapping:
        raise KeyError(
            f"{loc.best_embed_csv} has no delta_embed for concept {concept!r}"
        )
    return float(mapping[concept])


# ========================================================================== #
# Application                                                                 #
# ========================================================================== #

def apply_config(hf_model: Any, tokenizer: Any, config: str, concept: str, *,
                 model_name: str, rank: int, seed: int, train_eval: str,
                 results_root: Optional[Path] = None,
                 configs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Apply ``config``'s best HP to ``hf_model`` **once**, in place.

    Returns ``{"config", "hp", "info"}``. ``pristine`` is a no-op.

    No snapshot/restore cycling is needed (unlike the grid search): exactly one
    HP is applied per model load, and the model is discarded afterwards.
    """
    check_config(config)
    if config == "pristine":
        return {"config": config, "hp": {}, "info": {}}

    cfg = load_run_config(config, concept, model_name=model_name, rank=rank,
                          seed=seed, train_eval=train_eval, configs_dir=configs_dir)
    hp = resolve_hp(config, concept, model_name=model_name, rank=rank, seed=seed,
                    train_eval=train_eval, results_root=results_root,
                    configs_dir=configs_dir)

    method = methods_base.get(cfg.method)
    print(f"[cl-hp] config={config} method={cfg.method} concept={concept!r} hp={hp}")
    # on_concept_start loads the method's per-concept state (SNMF's
    # ConceptContext); apply() depends on it having run.
    method.on_concept_start(hf_model, concept, cfg)
    try:
        info = method.apply(hf_model, tokenizer, hp, concept, cfg)
    finally:
        method.on_concept_end(hf_model, concept, cfg)

    eval_model = method.get_model_to_eval()
    if eval_model is not None and eval_model is not hf_model:
        raise RuntimeError(
            f"method {cfg.method!r} returned a separate model to evaluate; "
            f"crosslingual.run evaluates the model it edits in place"
        )
    return {"config": config, "hp": hp, "info": info or {}}


__all__ = [
    "CONFIGS", "check_config", "order_configs",
    "yaml_path_for", "train_dir_for", "ember_best_embed_csv",
    "HPLocation", "locate", "load_run_config", "resolve_hp", "apply_config",
]
