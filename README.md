# dqntrain

Simple, resumable training pipelines for Unsloth + TRL. A pipeline is a short
list of stage objects; the Hugging Face repo is the only memory, so a wiped
notebook resumes exactly where it stopped.

## Documentation

Full documentation lives in [`docs/`](docs/index.md) and builds with MkDocs:

- [Quickstart](docs/quickstart.md)
- [Stages](docs/stages.md)
- [Configuration](docs/config.md)
- [Resume and state](docs/resume.md)
- [Rewards](docs/rewards.md)
- [API reference](docs/api.md)

[`examples/quick_test.ipynb`](examples/quick_test.ipynb) runs every stage type on Qwen3-0.6B 4-bit in ~10 minutes:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DQN-Labs/dqntrain/blob/main/examples/quick_test.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https%3A%2F%2Fraw.githubusercontent.com%2FDQN-Labs%2Fdqntrain%2Fmain%2Fexamples%2Fquick_test.ipynb)

```bash
pip install mkdocs-material
mkdocs serve     # local preview at http://127.0.0.1:8000
```

## Install

```bash
pip install dqntrain
```

`dqntrain` targets CUDA/ROCm machines where Unsloth works. The core pipeline
logic is dependency-light and runs anywhere.

## Quick start

```python
import dqntrain as dq

sft = dq.SFT("sft-code",
    data="org/code-instruct",
    train=dq.Train(epochs=2, lr=2e-4, batch=4),
    save_every=200,
)
pipe = dq.Sequential("nexus-v1", repo="you/nexus-ckpt",
    base="unsloth/Qwen3-4B", stages=[sft])
pipe.run()
```

Run it again after a crash, a reboot, or a fresh machine: `run()` downloads the
last checkpoint the manifest lists and continues. `pipe.summary()` shows what
`run()` will do without loading a model.

## Stages

| Class | Arguments | Notes |
|---|---|---|
| `SFT` | `data`, `lora=None`, `train=None`, `save_every=None`, `format="chatml"` | `SFTTrainer`. `format` is `"chatml"`, `"text"`, or `"template"` (tokenizer chat template) |
| `DPO` | `data`, `lora=None`, `train=None`, `save_every=None` | `DPOTrainer`. `beta` lives in `train` |
| `GRPO` | `data`, `reward_funcs`, `reward_weights=None`, `lora=None`, `train=None`, `save_every=None` | `GRPOTrainer`. `reward_funcs` is a list of callables |
| `Merge` | `push_to=None`, `dtype="bf16"` | Merges the LoRA into the base, dequantizes, optionally pushes to a separate repo |
| `Eval` | `tasks`, `limit=None` | lm-eval tasks on the current model; results saved as JSON |
| `Custom` | `fn`, `data=None`, `save_every=None` | `fn(model, tokenizer, data, ctx)` runs your training and returns the model |

Every stage takes `name` first. Names must be unique. `Sequential` accepts
`stages=[...]` and `pipe.add(*stages)`; both append in order.

`lora=None` does **not** mean full fine-tuning: SFT, DPO, and GRPO default to
LoRA with `LoRA()` defaults when `lora` is omitted.

### Custom stages

```python
def my_stage(model, tokenizer, data, ctx):
    trainer = MyTrainer(model=model, train_dataset=data["train"])
    trainer.train(resume_from_checkpoint=ctx.resume_dir)  # None on a fresh start
    ctx.log({"note": "trained"})
    ctx.save(trainer.state.global_step)                   # uploads a checkpoint
    return model

pipe.add(dq.Custom("my-stage", fn=my_stage, data="org/data"))
```

`ctx` exposes `resume_dir`, `run_dir`, `save(step, checkpoint_dir=None)`, and
`log(dict)`. `save` uploads the given folder (default
`<run_dir>/checkpoint-<step>`) through the same files-first, manifest-second
protocol as built-in stages.

## Config helpers

| Helper | Defaults | Forwarded to |
|---|---|---|
| `Data(name, split="train", subset=None, revision=None, seed=3407, limit=None, columns=None, filter=None, map=None, eval=None)` | `seed=3407` | `datasets.load_dataset` |
| `LoRA(r=16, alpha=16, dropout=0.0, target_modules="all-linear", **kwargs)` | `use_gradient_checkpointing="unsloth"` | `FastLanguageModel.get_peft_model` |
| `Train(**kwargs)` | none | the TRL config for the stage |
| `Load(**kwargs)` | `max_seq_length=2048`, `load_in_4bit=True` | `FastLanguageModel.from_pretrained` |

All helpers accept a plain dict: `dq.Train(**my_dict)`. Unknown keywords pass
through unchanged. `Data.columns` maps the **expected** name to the **source**
name: `columns={"prompt": "question"}` renames `question` to `prompt`.
`Data.eval` is a dict or `Data` with its own `name`, `split`, and `limit`.

### Alias table

| Short name | Real name |
|---|---|
| `epochs` | `num_train_epochs` |
| `lr` | `learning_rate` |
| `batch` | `per_device_train_batch_size` |
| `grad_accum` | `gradient_accumulation_steps` |
| `generations` | `num_generations` |
| `alpha` | `lora_alpha` |
| `four_bit` | `load_in_4bit` |

Aliasing happens once, in `__init__`. Passing both names raises `ValueError`.

## Resume behavior

- The repo layout is `<run>/<stage>/ckpt-000200/`, `<run>/<stage>/final/`,
  `<run>/logs/<stage>.jsonl`, and `<run>/dqntrain.json` (the manifest).
- A checkpoint exists only after all files and the manifest entry are
  committed. Files are committed first, the manifest second, so a crash in
  between leaves an orphan folder that the next run ignores.
- Full trainer state (`optimizer.pt`, `scheduler.pt`, `trainer_state.json`,
  RNG state) is saved by default. Pass `full_state=False` for adapter-only
  checkpoints: faster, but resume starts with a fresh optimizer.
- Every stage has a SHA-256 config hash chained to the previous stage. Changing
  any option on a done stage raises `ConfigChangedError` with the changed keys;
  call `pipe.rerun("stage")` to accept and retrain it and everything after it.
  Nothing is ever retrained silently.
- A lock with a heartbeat prevents two notebooks from running the same repo.
  A live lock raises `LockHeldError`; a stale heartbeat (over 15 minutes) is
  taken over and logged to `logs/_pipeline.jsonl`.
- `pipe.status()` prints stage status, latest checkpoint, and hash.
  `pipe.reset(confirm=True)` deletes the run in the store.

## Control methods

```python
pipe.summary()               # stages, dataset rows, columns, what run() will do
pipe.plan()                  # alias of summary()
pipe.status()                # stage, status, latest checkpoint, config hash
pipe.run()                   # resume or start; safe to call again
pipe.rerun("dpo-prefs")      # retrain this stage and all later stages
pipe.reset(confirm=True)     # delete this run's folder in the store
dq.env_report()              # python, torch, CUDA/ROCm, unsloth, bnb, trl, ...
```

## Data pipeline

`load_stage_data` runs a fixed order: load, shuffle (`seed`), `limit`,
`filter`, rename `columns`, `map`, required-column check, eval split.
Required columns: SFT needs `messages` or `text`; DPO needs `prompt`,
`chosen`, `rejected`; GRPO needs `prompt`. Missing columns raise
`DataColumnError` listing the required, missing, and found columns, and this
happens before any model loads.

## Rewards

`dqntrain.rewards` provides `format_reward(pattern)`, `exact_match_reward(column)`,
`contains_reward(text)`, and `length_penalty(target)`. Each returns a function
accepting `completions, **dataset_columns` and returning one float per
completion. Chat-format completions are handled through `_text`.

Reward source is part of the GRPO config hash. Functions without source (for
example some notebook lambdas) fall back to bytecode hashing, and
`summary()`/`plan()` print a warning naming the function.

> **Warning:** reward functions run inside the training process with full
> access to it. A reward that executes model-generated code (for example,
> running generated Python to check it) must be executed in a subprocess with
> a timeout; do not `exec` model output in the reward function itself.

## Stores and tests

`hf` repos are the default (`repo="you/checkpoints"`; new repos are created
private). `LocalStore(root_dir)` writes the same layout locally and never
touches the network; pass `store=LocalStore(...)` to `Sequential`. The test
suite uses `LocalStore` plus mock trainers; `DQNTRAIN_TEST_NETWORK=1 pytest`
additionally runs a real-TRL tiny-model integration test.

## Tested versions

The pipeline and resume logic are tested against:

- Python 3.10, torch 2.11
- transformers 5.0.0 (the TRL trainer API is used via `processing_class`, with
  a `tokenizer=` fallback for older TRL)
- trl 1.2.0, datasets 4.8.4, huggingface_hub 1.2.3

Unsloth and ROCm support depend on exact environment pins; install the Unsloth
version recommended for your GPU and driver, then match `torch`, `trl`, and
`transformers` to it.

## Out of scope for v1

Branching pipelines, automatic checkpoint pruning, automatic eval splits,
reward sandboxing, and multi-GPU orchestration beyond Unsloth and TRL.
