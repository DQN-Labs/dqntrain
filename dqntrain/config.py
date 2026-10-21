"""Friendly configuration helpers and the alias table."""
from __future__ import annotations

ALIASES = {
    "epochs": "num_train_epochs",
    "lr": "learning_rate",
    "batch": "per_device_train_batch_size",
    "grad_accum": "gradient_accumulation_steps",
    "generations": "num_generations",
    "alpha": "lora_alpha",
    "four_bit": "load_in_4bit",
}

_UNSET = object()


def apply_aliases(values):
    """Map short names to real ones. Unknown keys pass through untouched."""
    out = {}
    for key, value in values.items():
        real = ALIASES.get(key, key)
        if real != key and real in values:
            raise ValueError(f"pass either {key!r} or {real!r}, not both")
        out[real] = value
    return out


class Data:
    """Dataset description.

    ``columns`` maps the expected column name to the source name, so
    ``columns={"prompt": "question"}`` renames ``question`` to ``prompt``.
    ``eval`` may be a dict or a ``Data`` describing the eval split.
    """

    def __init__(self, name, split="train", subset=None, revision=None, seed=3407,
                 limit=None, columns=None, filter=None, map=None, eval=None):
        self.name = name
        self.split = split
        self.subset = subset
        self.revision = revision
        self.seed = seed
        self.limit = limit
        self.columns = dict(columns) if columns else {}
        self.filter = filter
        self.map = map
        self.eval = as_data(eval) if eval is not None else None

    def __repr__(self):
        return f"Data({self.name!r}, split={self.split!r}, limit={self.limit!r})"


def as_data(value):
    if isinstance(value, Data):
        return value
    if isinstance(value, str):
        return Data(value)
    if isinstance(value, dict):
        return Data(**value)
    raise TypeError(f"expected Data, dict, or str, got {type(value).__name__}")


class LoRA:
    """LoRA options forwarded to ``peft.get_peft_model``.

    Real defaults are ``r=16, lora_alpha=16, lora_dropout=0.0,
    target_modules="all-linear"`` plus transformers gradient checkpointing
    (``use_gradient_checkpointing=True``).
    """

    def __init__(self, r=_UNSET, alpha=_UNSET, dropout=_UNSET, target_modules=_UNSET, **kwargs):
        values = apply_aliases(kwargs)
        for key, value, default in (
            ("r", r, 16),
            ("lora_alpha", alpha, 16),
            ("lora_dropout", dropout, 0.0),
            ("target_modules", target_modules, "all-linear"),
        ):
            if value is not _UNSET:
                if key in values:
                    raise ValueError("pass either 'alpha' or 'lora_alpha', not both")
                values[key] = value
            elif key not in values:
                values[key] = default
        values.setdefault("use_gradient_checkpointing", True)
        self.values = values

    def __repr__(self):
        return f"LoRA({self.values})"


class Train:
    """Training options forwarded to the TRL config after aliasing."""

    def __init__(self, **kwargs):
        self.values = apply_aliases(kwargs)

    def __repr__(self):
        return f"Train({self.values})"


class Load:
    """Model loading options forwarded to the transformers backend.

    Defaults are ``max_seq_length=2048`` and ``load_in_4bit=False`` (bf16);
    pass ``four_bit=True`` or ``load_in_4bit=True`` only when bitsandbytes
    works on the machine.
    """

    def __init__(self, **kwargs):
        values = apply_aliases(kwargs)
        values.setdefault("max_seq_length", 2048)
        values.setdefault("load_in_4bit", False)
        self.values = values

    def __repr__(self):
        return f"Load({self.values})"
