"""Stage classes and their run functions."""
from __future__ import annotations

import json
import os

from .config import LoRA, Train, as_data
from .data import load_stage_data

FORMATS = ("chatml", "text", "template")
TRAINING_KINDS = ("SFT", "DPO", "GRPO")


def _fast_language_model():
    from unsloth import FastLanguageModel

    return FastLanguageModel


def _load_adapter(model, adapter_dir):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)


def _trainer_cls(kind):
    if kind == "SFT":
        from trl import SFTTrainer

        return SFTTrainer
    if kind == "DPO":
        from trl import DPOTrainer

        return DPOTrainer
    from trl import GRPOTrainer

    return GRPOTrainer


def _config_cls(kind):
    if kind == "SFT":
        from trl import SFTConfig

        return SFTConfig
    if kind == "DPO":
        from trl import DPOConfig

        return DPOConfig
    from trl import GRPOConfig

    return GRPOConfig


class SFT:
    kind = "SFT"

    def __init__(self, name, data, lora=None, train=None, save_every=None, format="chatml"):
        if format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, got {format!r}")
        self.name = name
        self.data = as_data(data)
        self.lora = lora if lora is not None else LoRA()
        self.train = train if train is not None else Train()
        self.save_every = save_every
        self.format = format


class DPO:
    kind = "DPO"

    def __init__(self, name, data, lora=None, train=None, save_every=None):
        self.name = name
        self.data = as_data(data)
        self.lora = lora if lora is not None else LoRA()
        self.train = train if train is not None else Train()
        self.save_every = save_every


class GRPO:
    kind = "GRPO"

    def __init__(self, name, data, reward_funcs, reward_weights=None, lora=None, train=None, save_every=None):
        self.name = name
        self.data = as_data(data)
        self.reward_funcs = list(reward_funcs)
        self.reward_weights = reward_weights
        self.lora = lora if lora is not None else LoRA()
        self.train = train if train is not None else Train()
        self.save_every = save_every


class Merge:
    kind = "Merge"

    def __init__(self, name, push_to=None, dtype="bf16"):
        self.name = name
        self.push_to = push_to
        self.dtype = dtype


class Eval:
    kind = "Eval"

    def __init__(self, name, tasks, limit=None):
        self.name = name
        self.tasks = tasks
        self.limit = limit


class Custom:
    kind = "Custom"

    def __init__(self, name, fn, data=None, save_every=None):
        self.name = name
        self.fn = fn
        self.data = as_data(data) if data is not None else None
        self.save_every = save_every


def run_stage(stage, ctx):
    if stage.kind == "Merge":
        return _run_merge(stage, ctx)
    if stage.kind == "Eval":
        return _run_eval(stage, ctx)
    if stage.kind == "Custom":
        return _run_custom(stage, ctx)
    return _run_trainer(stage, ctx)


def _is_adapter_dir(path):
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def _load_initial_model(ctx):
    fast = _fast_language_model()
    source = ctx.resume_dir or ctx.prev_final_dir
    if source and (ctx.resume_dir or _is_adapter_dir(source)):
        try:
            model, tokenizer = fast.from_pretrained(model_name=source, **ctx.load_values)
        except Exception:
            model, tokenizer = fast.from_pretrained(model_name=ctx.base, **ctx.load_values)
            model = _load_adapter(model, source)
        ctx.adapter_loaded = True
        return model, tokenizer
    return fast.from_pretrained(model_name=source or ctx.base, **ctx.load_values)


def _prepare_model(stage, ctx):
    if ctx.model is None:
        ctx.model, ctx.tokenizer = _load_initial_model(ctx)
    if stage.kind in TRAINING_KINDS and not ctx.adapter_loaded:
        ctx.model = _fast_language_model().get_peft_model(ctx.model, **stage.lora.values)
        ctx.adapter_loaded = True
    return ctx.model, ctx.tokenizer


def _save_tokenizer(tokenizer, folder):
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(folder)


def _last_metrics(state):
    for record in reversed(getattr(state, "log_history", None) or []):
        if "loss" in record or "reward" in record:
            return dict(record)
    return {}


def _checkpoint_callback(ctx):
    from transformers import TrainerCallback

    class _CkptCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            step = int(state.global_step)
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            ctx.on_checkpoint(step, checkpoint_dir, _last_metrics(state))

    return _CkptCallback()


def _apply_format(stage, data, tokenizer):
    if stage.kind != "SFT" or stage.format != "template":
        return data

    def to_text(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    return {
        "train": data["train"].map(to_text),
        "eval": data["eval"].map(to_text) if data["eval"] is not None else None,
    }


def _train_config(stage, ctx):
    values = dict(stage.train.values)
    values["output_dir"] = os.path.join(ctx.run_dir, "output")
    if stage.save_every:
        values["save_strategy"] = "steps"
        values["save_steps"] = stage.save_every
    if not ctx.pipe.full_state:
        values.setdefault("save_only_model", True)
    if stage.kind == "GRPO" and stage.reward_weights is not None:
        values.setdefault("reward_weights", stage.reward_weights)
    return _config_cls(stage.kind)(**values)


def _build_trainer(trainer_cls, kwargs, tokenizer):
    kwargs = dict(kwargs)
    kwargs["processing_class"] = tokenizer
    try:
        return trainer_cls(**kwargs)
    except TypeError as error:
        if "processing_class" not in str(error):
            raise
        kwargs.pop("processing_class")
        kwargs["tokenizer"] = tokenizer
        return trainer_cls(**kwargs)


def _run_trainer(stage, ctx):
    data = load_stage_data(stage.data, stage.kind)
    model, tokenizer = _prepare_model(stage, ctx)
    data = _apply_format(stage, data, tokenizer)
    kwargs = {
        "model": model,
        "args": _train_config(stage, ctx),
        "train_dataset": data["train"],
        "eval_dataset": data["eval"],
        "callbacks": [_checkpoint_callback(ctx)],
    }
    if stage.kind == "GRPO":
        kwargs["reward_funcs"] = stage.reward_funcs
    trainer = _build_trainer(_trainer_cls(stage.kind), kwargs, tokenizer)
    trainer.train(resume_from_checkpoint=ctx.resume_dir)
    final_dir = os.path.join(ctx.run_dir, "final")
    trainer.save_model(final_dir)
    _save_tokenizer(tokenizer, final_dir)
    ctx.finish(final_dir)
    return model, tokenizer


def _run_merge(stage, ctx):
    model, tokenizer = _prepare_model(stage, ctx)
    final_dir = os.path.join(ctx.run_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    merged = False
    if hasattr(model, "save_pretrained_merged"):
        save_method = "merged_4bit" if stage.dtype in ("4bit", "int4") else "merged_16bit"
        try:
            model.save_pretrained_merged(final_dir, tokenizer, save_method=save_method)
            merged = True
        except RuntimeError:
            merged = False
    if not merged:
        merged_model = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
        merged_model.save_pretrained(final_dir)
        _save_tokenizer(tokenizer, final_dir)
    if stage.push_to:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(stage.push_to, private=True, exist_ok=True)
        api.upload_folder(repo_id=stage.push_to, folder_path=final_dir)
    ctx.finish(final_dir)
    return model, tokenizer


def _run_eval(stage, ctx):
    try:
        from lm_eval import simple_evaluate
    except ImportError as error:
        raise RuntimeError("Eval stages need lm-eval: pip install lm-eval") from error
    model_args = {"pretrained": ctx.base}
    if ctx.prev_final_dir and not _is_adapter_dir(ctx.prev_final_dir):
        model_args["pretrained"] = ctx.prev_final_dir
    elif ctx.prev_final_dir:
        model_args["peft"] = ctx.prev_final_dir
    results = simple_evaluate(model="hf", model_args=model_args, tasks=list(stage.tasks), limit=stage.limit)
    final_dir = os.path.join(ctx.run_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    with open(os.path.join(final_dir, "results.json"), "w") as handle:
        json.dump(results, handle, indent=2, default=str)
    ctx.finish(final_dir)
    return None, None


class _CustomContext:
    """Small context handed to Custom stage functions."""

    def __init__(self, ctx):
        self._ctx = ctx

    @property
    def resume_dir(self):
        return self._ctx.resume_dir

    @property
    def run_dir(self):
        return self._ctx.run_dir

    def save(self, step, checkpoint_dir=None):
        step = int(step)
        target = checkpoint_dir or os.path.join(self._ctx.run_dir, f"checkpoint-{step:06d}")
        os.makedirs(target, exist_ok=True)
        return self._ctx.on_checkpoint(step, target, {})

    def log(self, record):
        self._ctx.log(record)


def _run_custom(stage, ctx):
    data = load_stage_data(stage.data, "Custom") if stage.data is not None else None
    model, tokenizer = _prepare_model(stage, ctx)
    result = stage.fn(model, tokenizer, data, _CustomContext(ctx))
    if result is not None:
        ctx.model = result
    final_dir = os.path.join(ctx.run_dir, "final")
    if hasattr(ctx.model, "save_pretrained"):
        try:
            ctx.model.save_pretrained(final_dir)
            _save_tokenizer(tokenizer, final_dir)
            ctx.finish(final_dir)
            return ctx.model, tokenizer
        except TypeError:
            pass
    ctx.finish(None)
    return ctx.model, tokenizer
