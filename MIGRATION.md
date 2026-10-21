# Migration: Unsloth backend → transformers + PEFT + TRL

Unsloth cannot be installed reliably on the ROCm MI300X machines, so the
training backend is now plain transformers + PEFT + TRL. The pipeline layer
(stages, manifest, resume, locks, config hashing, data pipeline, rewards) is
unchanged, and the manifest format did not change.

## Backend contract

`dqntrain/backend.py` exposes five functions used by every stage:

| Function | Implementation |
|---|---|
| `load_model(name, dtype="bf16", max_seq_length=2048, load_in_4bit=False, **kwargs)` | `AutoTokenizer` + `AutoModelForCausalLM.from_pretrained(..., dtype=...)`; falls back to `AutoModelForImageTextToText.from_pretrained` (text path) and prints which loader/class worked |
| `apply_lora(model, lora_cfg)` | `peft.get_peft_model` with `LoraConfig`, then transformers `gradient_checkpointing_enable()` and `enable_input_require_grads()` when requested; k-bit models go through `prepare_model_for_kbit_training` |
| `generate(model, tokenizer, prompts, **gen_kwargs)` | left-padded `model.generate`; sets `model.eval()`, restores `model.train(was_training)` and the previous `use_cache` flag |
| `save_adapter(model, path)` | `model.save_pretrained` (adapter files) |
| `merge_and_save(model, path, dtype="bf16")` | `merge_and_unload()` then `save_pretrained` after casting to bf16/fp16/fp32 |

`_load_adapter(model, adapter_dir)` is a private helper for resume: it wraps the
base model with `PeftModel.from_pretrained(adapter_dir, is_trainable=True)`.

## Unsloth calls replaced

| Removed Unsloth call | Replacement |
|---|---|
| `from unsloth import FastLanguageModel` / `_fast_language_model()` | lazy `transformers` imports inside `backend.py`; no Unsloth import anywhere |
| `FastLanguageModel.from_pretrained(model_name=..., max_seq_length=..., load_in_4bit=...)` | `backend.load_model(name, dtype, max_seq_length, load_in_4bit)` |
| `FastLanguageModel.get_peft_model(model, **LoRA.values)` | `backend.apply_lora(model, lora.values)` |
| `"use_gradient_checkpointing": "unsloth"` | `"use_gradient_checkpointing": True` → transformers `gradient_checkpointing_enable()` |
| Unsloth adapter loading via `FastLanguageModel.from_pretrained(adapter_dir)` | `backend._load_adapter` → `peft.PeftModel.from_pretrained(..., is_trainable=True)` |
| `model.save_pretrained_merged(folder, tokenizer, save_method="merged_16bit")` | `backend.merge_and_save(model, folder, dtype)` |
| Unsloth `RuntimeError` merge fallback (`merge_and_unload` + `save_pretrained`) | no longer needed; merge is the primary path |
| Unsloth-patched generation in callbacks | stock `model.generate`; Custom stages can call `backend.generate` |
| `env_report()` requiring Unsloth | Unsloth is optional in `env_info()` and reports `None` when absent |

The `Merge` stage still accepts `dtype="bf16" | "fp16" | "fp32"`. `"4bit"` now
raises `ValueError`: it requires bitsandbytes quantization and is not part of
the transformers merge path.

## Behavior changes

- `Load()` defaults to `load_in_4bit=False` (bf16 LoRA). `four_bit=True` only
  works when bitsandbytes works on the machine; the backend raises a clear
  `RuntimeError` otherwise.
- `LoRA()` defaults to `use_gradient_checkpointing=True`; there is no Unsloth
  variant.
- Config hashes change once because the LoRA default value changed. The
  manifest format is untouched. Existing dqntrain runs raise
  `ConfigChangedError`; call `pipe.rerun("<stage>")` to accept the new backend
  and retrain from that stage.
- `env_info()` still records `unsloth` (as `None`); every checkpoint's
  `meta.json` records `torch`, `transformers`, `trl`, `peft`, and `dqntrain`
  versions, and `env_report()` runs without Unsloth installed.
- Multi-adapter MOPD is supported directly through PEFT:
  `add_adapter` / `load_adapter` / `set_adapter` / `disable_adapter` /
  `delete_adapter`. `tests/test_backend.py` exercises switching, the disabled
  adapter context, and deletion.

## Qwen3.8 (vision-language) checkpoints

`load_model` tries `AutoModelForCausalLM` first. If the checkpoint is
vision-language and that loader fails, it retries with
`AutoModelForImageTextToText`, then uses the text path. The loader and the
concrete model class are printed once:

```
[dqntrain] loaded Qwen/Qwen3.8-27B with AutoModelForImageTextToText (<model class>)
```

No images are needed for the text path; prompts are tokenized with the
tokenizer (falling back to `AutoProcessor.tokenizer` if the checkpoint has no
standalone tokenizer).

## MOPD step time

`scripts/bench_mopd.py` reproduces the MOPD shape: two adapters on one model
(student `default`, frozen `teacher`), 16 prompts, 512 new tokens, one rollout
pass, plus a separate adapter-switch/`disable_adapter` check. It logs JSON:

```bash
python scripts/bench_mopd.py --model Qwen/Qwen3.8-27B \
    --teacher-adapter /path/to/teacher --rollouts 16 --max-new-tokens 512
# MOPD_STEP_METRICS {"mopd_step_seconds": ..., "seconds_per_rollout": ...}
```

**Measured on MI300X: pending.** The migration was developed on macOS without a
ROCm GPU, so no honest MI300X number can be reported from this environment.
Run the command above on the target box and paste its `MOPD_STEP_METRICS` line
here. Do not tune generation until that number exists; HF `generate` for
rollouts is the expected bottleneck, and vLLM is out of scope.

Local validation (CPU, tiny random Llama, not representative of MI300X):
real `SFTTrainer` and `GRPOTrainer` runs complete end to end, and kill/resume
continues from the last committed checkpoint.

## Verification

- `pytest`: 46 passed, 1 skipped (network-gated real-TRL test), with
  `LocalStore` and no network.
- `tests/test_backend.py` loads a tiny Llama from a local folder, applies LoRA,
  generates, saves the adapter, merges and reloads the merged model, switches
  two adapters, and checks the vision fallback and the missing-bitsandbytes
  error.
- Real CPU smokes with transformers 5.17 / trl 1.13 / peft 0.21:
  - SFT: LoRA training, two checkpoints, kill-and-resume from `ckpt-000001`.
  - GRPO (`loss_type="dr_grpo"`): two steps, kill-and-resume from
    `ckpt-000001`.
- Import without Unsloth: `import dqntrain` works and `tests/test_env.py`
  asserts Unsloth was not imported.

## Notes and risks

- Reward functions still run in-process. A reward that executes model output
  (for example a unit-test sandbox) must do so in a subprocess with a timeout;
  see the warning in the rewards docs. GRPO passes reward callables straight to
  TRL, so a sandboxed reward is unchanged by this migration.
- transformers 5.x changed some TRL config fields (for example
  `max_prompt_length` was removed from `GRPOConfig` in trl 1.13). `Train`
  remains a passthrough; remove stale fields from configs.
- peft 0.21 requires transformers 5.x; pin the trio
  `transformers==5.17.0 trl==1.13.0 peft==0.21.2` on the MI300X image.
- 27B bf16 LoRA fits in 192 GB; quantization is not required.
