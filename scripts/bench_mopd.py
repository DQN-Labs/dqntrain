"""Benchmark one MOPD-style rollout step with two PEFT adapters.

Run on the target machine (no vLLM):

    python scripts/bench_mopd.py --model Qwen/Qwen3.8-27B \
        --teacher-adapter /path/to/teacher --rollouts 16 --max-new-tokens 512

Without --teacher-adapter the script adds a second, freshly initialized LoRA
adapter named "teacher" so adapter switching is still exercised.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

import dqntrain.backend as backend

PROMPTS = [
    "Solve: what is 17 * 23? Show your work and put the answer in \\boxed{}.",
    "Solve: what is 144 / 12? Show your work and put the answer in \\boxed{}.",
    "Solve: what is 9 + 41 - 7? Show your work and put the answer in \\boxed{}.",
    "Solve: what is 15% of 240? Show your work and put the answer in \\boxed{}.",
    "Solve: what is 3^4? Show your work and put the answer in \\boxed{}.",
    "Solve: what is 2/3 + 1/6? Show your work and put the answer in \\boxed{}.",
    "Solve: a train goes 60 mph for 2.5 hours. How far does it go? Use \\boxed{}.",
    "Solve: what is the sum of the first 20 positive integers? Use \\boxed{}.",
    "Solve: what is 81 squared? Show your work and put the answer in \\boxed{}.",
    "Solve: what is 7! / 5!? Show your work and put the answer in \\boxed{}.",
    "Solve: if 5x + 7 = 42, what is x? Put the answer in \\boxed{}.",
    "Solve: what is 12.5 * 8? Show your work and put the answer in \\boxed{}.",
    "Solve: what is 1000 - 379? Show your work and put the answer in \\boxed{}.",
    "Solve: what is the gcd of 84 and 126? Put the answer in \\boxed{}.",
    "Solve: what is 0.25 as a fraction? Put the answer in \\boxed{}.",
    "Solve: what is 45 minutes in seconds? Put the answer in \\boxed{}.",
]


def lora_config():
    return {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "target_modules": "all-linear",
        "use_gradient_checkpointing": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--teacher-adapter", default=None)
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    model, tokenizer = backend.load_model(
        args.model, dtype=args.dtype, max_seq_length=args.max_seq_length
    )
    model = backend.apply_lora(model, lora_config())
    if args.teacher_adapter:
        model.load_adapter(args.teacher_adapter, adapter_name="teacher", is_trainable=False)
    else:
        from peft import LoraConfig

        model.add_adapter("teacher", LoraConfig(r=32, lora_alpha=64, target_modules="all-linear"))
    model.eval()

    prompts = (PROMPTS * ((args.rollouts + len(PROMPTS) - 1) // len(PROMPTS)))[: args.rollouts]
    generation = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
    }

    if args.warmup:
        backend.generate(model, tokenizer, prompts[:1], max_new_tokens=8, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    model.set_adapter("default")
    outputs = backend.generate(model, tokenizer, prompts, **generation)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    switch_started = time.perf_counter()
    model.set_adapter("teacher")
    backend.generate(model, tokenizer, prompts[:1], max_new_tokens=8, do_sample=False)
    with model.disable_adapter():
        backend.generate(model, tokenizer, prompts[:1], max_new_tokens=8, do_sample=False)
    model.set_adapter("default")
    switch_elapsed = time.perf_counter() - switch_started

    report = {
        "model": args.model,
        "teacher_adapter": args.teacher_adapter,
        "dtype": args.dtype,
        "rollouts": len(prompts),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "mopd_step_seconds": round(elapsed, 3),
        "seconds_per_rollout": round(elapsed / len(prompts), 3),
        "adapter_switch_check_seconds": round(switch_elapsed, 3),
        "sample_completion_chars": [len(text) for text in outputs[:4]],
    }
    print("MOPD_STEP_METRICS " + json.dumps(report))
    return report


if __name__ == "__main__":
    main()
