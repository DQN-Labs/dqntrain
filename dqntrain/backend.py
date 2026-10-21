"""Model backend built on transformers + PEFT, with no Unsloth dependency."""
from __future__ import annotations

import os

_DTYPES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "fp32": "float32",
    "float32": "float32",
}


def _torch_dtype(dtype):
    import torch

    if dtype is None:
        return torch.bfloat16
    if not isinstance(dtype, str):
        return dtype
    try:
        return getattr(torch, _DTYPES[dtype.lower()])
    except KeyError:
        raise ValueError(f"unknown dtype {dtype!r}; use bf16, fp16, fp32, or a torch dtype") from None


def _model_classes():
    from transformers import AutoModelForCausalLM

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None
    return AutoModelForCausalLM, AutoModelForImageTextToText


def _load_tokenizer(name):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(name)
        return getattr(processor, "tokenizer", processor)


def _quantization_config(compute_dtype=None):
    import torch

    try:
        import bitsandbytes
        from transformers import BitsAndBytesConfig
    except Exception as error:
        raise RuntimeError(
            "load_in_4bit=True needs a working bitsandbytes install; "
            "pass four_bit=False to train in bf16"
        ) from error
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype or torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_model(name, dtype="bf16", max_seq_length=2048, load_in_4bit=False, **kwargs):
    """Load ``name`` plus tokenizer; returns ``(model, tokenizer)``.

    Tries ``AutoModelForCausalLM`` first and falls back to
    ``AutoModelForImageTextToText`` (then uses the text path) for
    vision-language checkpoints.
    """
    tokenizer = _load_tokenizer(name)
    if max_seq_length:
        tokenizer.model_max_length = int(max_seq_length)
    values = {"dtype": _torch_dtype(dtype), **kwargs}
    if load_in_4bit:
        values["quantization_config"] = _quantization_config(values["dtype"])
    causal, vision = _model_classes()
    loader = causal.__name__
    try:
        model = causal.from_pretrained(name, **values)
    except TypeError as error:
        if "dtype" not in str(error):
            raise
        values["torch_dtype"] = values.pop("dtype")
        model = causal.from_pretrained(name, **values)
    except Exception as error:
        if vision is None:
            raise
        loader = vision.__name__
        try:
            model = vision.from_pretrained(name, **values)
        except Exception:
            raise error
    print(f"[dqntrain] loaded {name} with {loader} ({type(model).__name__})")
    return model, tokenizer


def apply_lora(model, lora_cfg):
    """Wrap the model with PEFT LoRA and enable transformers gradient checkpointing."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    values = dict(lora_cfg)
    gradient_checkpointing = values.pop("use_gradient_checkpointing", None)
    values.setdefault("task_type", "CAUSAL_LM")
    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(gradient_checkpointing)
        )
    model = get_peft_model(model, LoraConfig(**values))
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if getattr(model, "config", None) is not None and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model


def generate(model, tokenizer, prompts, **gen_kwargs):
    """Run ``model.generate`` on text prompts and return the new text only."""
    if isinstance(prompts, str):
        prompts = [prompts]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_window = getattr(tokenizer, "model_max_length", None)
    max_new = gen_kwargs.get("max_new_tokens")
    tokenize_args = {"return_tensors": "pt", "padding": True}
    if isinstance(token_window, int) and 0 < token_window < 100000:
        tokenize_args["truncation"] = True
        tokenize_args["max_length"] = token_window
        if isinstance(max_new, int):
            tokenize_args["max_length"] = max(1, token_window - max_new)
    enc = tokenizer(list(prompts), **tokenize_args)

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    enc = enc.to(device)
    gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)

    was_training = model.training
    model.eval()
    use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    if use_cache is not None:
        model.config.use_cache = True
    try:
        output = model.generate(**enc, **gen_kwargs)
    finally:
        if use_cache is not None:
            model.config.use_cache = use_cache
        model.train(was_training)
    new_tokens = output[:, enc["input_ids"].shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def save_adapter(model, path):
    """Save PEFT adapter files (weights + config) to ``path``."""
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    return path


def merge_and_save(model, path, dtype="bf16"):
    """Merge LoRA into the base weights and save a full model to ``path``."""
    if str(dtype).lower() in ("4bit", "int4"):
        raise ValueError("4-bit merge needs bitsandbytes; use bf16, fp16, or fp32")
    os.makedirs(path, exist_ok=True)
    merged = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
    merged.to(_torch_dtype(dtype))
    if getattr(merged, "config", None) is not None:
        merged.config.use_cache = True
    merged.save_pretrained(path)
    return path


def _load_adapter(model, adapter_dir):
    from peft import PeftModel, prepare_model_for_kbit_training

    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(model)
    return PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
