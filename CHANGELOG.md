# Changelog

## 1.0.0

Initial release.

- Transformers + PEFT + TRL backend with no Unsloth dependency; optional 4-bit
  loading when bitsandbytes works.
- `Sequential` pipelines with automatic resume from a Hugging Face repo (or a
  `LocalStore` folder), config hashing, locking, and rerun support.
- Stages: `SFT`, `DPO`, `GRPO`, `Merge`, `Eval`, `Custom`.
- Config helpers: `Data`, `LoRA`, `Train`, `Load`, with friendly aliases and
  pass-through of unknown keyword arguments to PEFT and TRL.
- Fixed 8-step data pipeline with fail-fast column checks.
- Built-in GRPO reward helpers with source-based hashing.
- Checkpoint protocol: files first, manifest second; full trainer state by
  default; `meta.json` per checkpoint; per-stage JSONL logs.
- Optional `callbacks=` on `SFT`, `DPO`, and `GRPO`; callbacks run at each save
  before the checkpoint is uploaded and are included in the stage hash.
- `env_report()` for environment diagnostics used in checkpoint metadata.
