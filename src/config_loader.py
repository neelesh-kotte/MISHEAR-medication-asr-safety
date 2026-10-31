"""Load and validate MISHEAR YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


CONFIG_FILENAMES = {
    "asr": "asr_config.yaml",
    "dataset": "dataset_config.yaml",
    "experiment": "experiment_config.yaml",
    "pipeline": "pipeline_config.yaml",
    "corpus": "corpus_config.yaml",
    "evaluation": "evaluation_config.yaml",
    "paper_artifacts": "paper_artifacts.yaml",
}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping from disk, reporting missing/invalid files clearly."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {source}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML configuration loading requires PyYAML; install it from requirements.txt") from exc
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Configuration root must be a YAML mapping: {source}")
    config_kind = source.stem.split(".", 1)[0].replace("_config", "")
    validate_config(value, config_kind)
    return value


def load_project_configs(config_dir: str | Path = "config") -> dict[str, dict[str, Any]]:
    """Load the four project configuration files by their conventional names."""
    root = Path(config_dir)
    return {name: load_yaml_config(root / filename) for name, filename in CONFIG_FILENAMES.items()}


def validate_config(config: Mapping[str, Any], kind: str | None = None) -> None:
    """Validate structural invariants shared by CLI, tests, and pipeline code."""
    if kind is None:
        return
    if kind == "asr":
        recognizers = config.get("recognizers")
        if not isinstance(recognizers, list) or not recognizers:
            raise ValueError("asr_config.yaml requires a non-empty 'recognizers' list")
        for index, item in enumerate(recognizers):
            if not isinstance(item, Mapping) or not item.get("backend"):
                raise ValueError(f"ASR recognizer {index} must be a mapping with a backend")
    elif kind == "dataset":
        if "random_seed" not in config:
            raise ValueError("dataset_config.yaml requires random_seed")
        for field in ("pilot_pair_target", "main_pair_planning_target"):
            interval = config.get(field)
            if interval is not None and (not isinstance(interval, (list, tuple)) or len(interval) != 2
                                         or int(interval[0]) > int(interval[1])):
                raise ValueError(f"{field} must be a two-value increasing interval")
    elif kind == "experiment":
        fractions = config.get("ingredient_split_fraction")
        if fractions is not None:
            if not isinstance(fractions, Mapping) or not {"train", "validation", "test"}.issubset(fractions):
                raise ValueError("ingredient_split_fraction must define train, validation, and test")
            if any(float(fractions[key]) < 0 for key in ("train", "validation", "test")):
                raise ValueError("split fractions cannot be negative")
            if abs(sum(float(fractions[key]) for key in ("train", "validation", "test")) - 1.0) > 1e-6:
                raise ValueError("ingredient split fractions must sum to 1")
        budget = config.get("false_alert_budget_per_100_mentions")
        if budget is not None and float(budget) < 0:
            raise ValueError("false-alert budget cannot be negative")
    elif kind == "pipeline":
        if config.get("vocabulary_csv") and not isinstance(config["vocabulary_csv"], str):
            raise ValueError("vocabulary_csv must be a path string")
        asr = config.get("asr")
        if asr is not None and (not isinstance(asr, Mapping) or not asr.get("backend")
                                and not asr.get("engine")):
            raise ValueError("pipeline asr config must specify a backend")
        budget = config.get("false_alert_budget")
        if budget is not None and not 0 <= float(budget) <= 1:
            raise ValueError("false_alert_budget must be a fraction in [0, 1]")
        conditions = config.get("dsp_conditions")
        if conditions is not None and not isinstance(conditions, list):
            raise ValueError("dsp_conditions must be a list")
    elif kind == "corpus":
        datasets = config.get("datasets")
        if not isinstance(datasets, Mapping) or not datasets:
            raise ValueError("corpus_config.yaml requires a non-empty datasets mapping")
        for name, specification in datasets.items():
            if not isinstance(specification, Mapping):
                raise ValueError(f"Dataset configuration {name!r} must be a mapping")
            if not specification.get("manifest") and not specification.get("root"):
                raise ValueError(f"Dataset configuration {name!r} requires manifest or root")
        thresholds = config.get("stopping_rules", {})
        if not isinstance(thresholds, Mapping):
            raise ValueError("stopping_rules must be a mapping")
    elif kind == "evaluation":
        if config.get("evaluation_input_jsonl") and not isinstance(config["evaluation_input_jsonl"], str):
            raise ValueError("evaluation_input_jsonl must be a path")
        scores = config.get("score_fields")
        if scores is not None and (not isinstance(scores, Mapping) or not scores):
            raise ValueError("score_fields must map system names to score column names")
        budgets = config.get("review_budgets")
        if budgets is not None and any(not 0 < float(value) <= 1 for value in budgets):
            raise ValueError("review_budgets must be fractions in (0, 1]")
    elif kind == "paper_artifacts":
        evaluation_sets = config.get("evaluation_sets", {})
        if not isinstance(evaluation_sets, Mapping):
            raise ValueError("paper_artifacts evaluation_sets must be a mapping")
        for name, specification in evaluation_sets.items():
            if not isinstance(specification, Mapping) or not specification.get("predictions_jsonl"):
                raise ValueError(f"Paper artifact evaluation set {name!r} requires predictions_jsonl")
            scores = specification.get("score_fields")
            if not isinstance(scores, Mapping) or not scores:
                raise ValueError(f"Paper artifact evaluation set {name!r} requires score_fields")
        ablation = config.get("ablation")
        if ablation is not None and (not isinstance(ablation, Mapping)
                or not ablation.get("training_csv") or not ablation.get("development_csv")):
            raise ValueError("paper_artifacts ablation requires training_csv and development_csv")
    else:
        raise ValueError(f"Unknown config kind {kind!r}")
