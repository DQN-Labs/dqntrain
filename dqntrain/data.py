"""The fixed 8-step data pipeline."""
from __future__ import annotations

from .errors import DataColumnError

REQUIRED_COLUMNS = {
    "SFT": [("messages",), ("text",)],
    "DPO": [("prompt", "chosen", "rejected")],
    "GRPO": [("prompt",)],
    "Eval": [],
    "Custom": [],
    "Merge": [],
}


def _load_dataset(config):
    from datasets import load_dataset

    kwargs = {"path": config.name, "split": config.split}
    if config.subset is not None:
        kwargs["name"] = config.subset
    if config.revision is not None:
        kwargs["revision"] = config.revision
    return load_dataset(**kwargs)


def prepare_dataset(config):
    """Steps 1-6: load, shuffle, limit, filter, rename, map."""
    dataset = _load_dataset(config)
    dataset = dataset.shuffle(seed=config.seed)
    if config.limit is not None:
        dataset = dataset.select(range(min(config.limit, len(dataset))))
    if config.filter is not None:
        dataset = dataset.filter(config.filter)
    if config.columns:
        renames = {source: expected for expected, source in config.columns.items()}
        dataset = dataset.rename_columns(renames)
    if config.map is not None:
        dataset = dataset.map(config.map)
    return dataset


def check_columns(dataset, stage_kind):
    """Step 7: fail fast when required columns are missing.

    ``REQUIRED_COLUMNS`` holds alternatives: each tuple is satisfied when all
    its names are present, and any satisfied tuple passes the check.
    """
    options = REQUIRED_COLUMNS.get(stage_kind, [])
    found = list(dataset.column_names)
    if not options:
        return found
    if any(all(name in found for name in option) for option in options):
        return found
    required = [list(option) for option in options]
    missing = sorted({name for option in options for name in option if name not in found})
    raise DataColumnError(stage_kind, required, missing, found)


def load_stage_data(data_config, stage_kind):
    """Run all 8 steps and return ``{"train": dataset, "eval": dataset or None}``."""
    train = prepare_dataset(data_config)
    check_columns(train, stage_kind)
    eval_dataset = prepare_dataset(data_config.eval) if data_config.eval is not None else None
    return {"train": train, "eval": eval_dataset}
