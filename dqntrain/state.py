"""Manifest, lock, and config hashing."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from inspect import getsource
from types import CodeType

MANIFEST_NAME = "dqntrain.json"
MANIFEST_VERSION = 1
LOCK_TTL = timedelta(minutes=15)
_MISSING = object()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha12(value):
    text = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(text):
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def _code_fingerprint(code):
    consts = []
    for const in code.co_consts:
        if isinstance(const, CodeType):
            consts.append(_code_fingerprint(const))
        elif isinstance(const, (set, frozenset)):
            consts.append(sorted(repr(item) for item in const))
        else:
            consts.append(repr(const))
    parts = [repr(code.co_code), list(code.co_names), list(code.co_varnames), consts]
    return repr(parts)


def callable_hash(fn):
    """Return ``(hash, warning)``; warning is set when source is unavailable."""
    try:
        return sha12(getsource(fn)), None
    except (OSError, TypeError):
        code = getattr(fn, "__code__", None)
        if code is None:
            return sha12(repr(fn)), f"{getattr(fn, '__name__', '<callable>')} has no source or bytecode; hashed repr"
        warning = f"{getattr(fn, '__name__', '<callable>')} has no source; hashed bytecode"
        return sha12(_code_fingerprint(code)), warning


def callback_hash(callback):
    """Return ``(hash, warning)`` for a callback instance or plain callable."""
    if callable(callback) and hasattr(callback, "__code__"):
        return callable_hash(callback)
    cls = type(callback)
    parts = {}
    try:
        parts["source"] = getsource(cls)
        warning = None
    except (OSError, TypeError):
        parts["class"] = f"{cls.__module__}.{cls.__qualname__}"
        parts["methods"] = {
            key: callable_hash(value)[0]
            for key, value in vars(cls).items()
            if callable(value)
        }
        warning = f"{cls.__name__} has no source; hashed class path and methods"
    attrs = {}
    for key, value in vars(callback).items():
        attrs[key] = callable_hash(value)[0] if callable(value) else repr(value)
    parts["attrs"] = attrs
    return sha12(parts), warning


def _data_dict(data, chash, warnings):
    if data is None:
        return None
    def maybe_hash(fn):
        if fn is None:
            return None
        value, warning = chash(fn)
        if warning:
            warnings.append(warning)
        return value
    return {
        "name": data.name,
        "split": data.split,
        "subset": data.subset,
        "revision": data.revision,
        "seed": data.seed,
        "limit": data.limit,
        "columns": data.columns,
        "filter": maybe_hash(data.filter),
        "map": maybe_hash(data.map),
        "eval": _data_dict(data.eval, chash, warnings),
    }


def stage_config(stage, base, load_values, prev_hash):
    """Canonical config object for one stage, plus hashing warnings."""
    warnings = []

    def chash(fn):
        return callable_hash(fn)

    callbacks = []
    for fn in getattr(stage, "callbacks", []) or []:
        value, warning = callback_hash(fn)
        if warning:
            warnings.append(warning)
        callbacks.append(value)

    config = {
        "class": stage.kind,
        "name": stage.name,
        "data": _data_dict(getattr(stage, "data", None), chash, warnings),
        "lora": stage.lora.values if getattr(stage, "lora", None) is not None else None,
        "train": stage.train.values if getattr(stage, "train", None) is not None else {},
        "load": load_values,
        "base": base,
        "save_every": getattr(stage, "save_every", None),
        "format": getattr(stage, "format", None),
        "callbacks": callbacks,
        "prev_hash": prev_hash,
    }
    if stage.kind == "Merge":
        config["push_to"] = stage.push_to
        config["dtype"] = stage.dtype
    elif stage.kind == "Eval":
        config["tasks"] = list(stage.tasks)
        config["limit"] = stage.limit
    elif stage.kind == "GRPO":
        rewards = []
        for fn in stage.reward_funcs:
            value, warning = chash(fn)
            if warning:
                warnings.append(warning)
            rewards.append(value)
        config["reward_funcs"] = rewards
        config["reward_weights"] = stage.reward_weights
    elif stage.kind == "Custom":
        value, warning = chash(stage.fn)
        if warning:
            warnings.append(warning)
        config["fn"] = value
    return config, warnings


def empty_manifest():
    return {"version": MANIFEST_VERSION, "stages": [], "lock": {}}


def find_stage(manifest, name):
    for record in manifest.get("stages", []):
        if record.get("name") == name:
            return record
    return None


def lock_is_fresh(lock):
    heartbeat = (lock or {}).get("heartbeat")
    if not heartbeat:
        return False
    try:
        return datetime.now(timezone.utc) - parse_time(heartbeat) < LOCK_TTL
    except (TypeError, ValueError):
        return False


def config_diff(old, new):
    changes = {}

    def walk(path, left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                walk(f"{path}.{key}" if path else str(key), left.get(key, _MISSING), right.get(key, _MISSING))
        elif left != right:
            changes[path or "<config>"] = (None if left is _MISSING else left, None if right is _MISSING else right)

    walk("", old or {}, new or {})
    return changes
