# Configuration helpers

Each helper stores keyword arguments and applies the alias table below.
Unknown keys pass through untouched, and any helper accepts a plain dict:
`dq.Train(**my_dict)` works.

## `Data`

```python
dq.Data(
    name,
    split="train",
    subset=None,
    revision=None,
    seed=3407,
    limit=None,
    columns=None,
    filter=None,
    map=None,
    eval=None,
)
```

| Argument | Meaning |
|---|---|
| `name` | Dataset id passed to `datasets.load_dataset` |
| `split`, `subset`, `revision` | Forwarded to `load_dataset` |
| `seed` | Shuffle seed; default `3407` |
| `limit` | Keep only the first N rows after shuffling |
| `columns` | Rename map: expected name to source name |
| `filter` | Function taking one example, returning `bool` |
| `map` | Function taking one example, returning a dict merged into it |
| `eval` | A dict or `Data` with its own `name`, `split`, and `limit` |

Two important details:

- `columns` maps the **expected** column name to the **source** name:
  `columns={"prompt": "question", "answer": "solution"}` renames `question` to
  `prompt` and `solution` to `answer`. The required-column check runs after
  renaming.
- `eval` builds an explicit eval split. There is no automatic eval split
  (`test_size`) in v1; the default is no held-out split.

A string is shorthand for `Data(name)`:

```python
dpo = dq.DPO("dpo-prefs", data="org/prefs")
```

## `LoRA`

```python
dq.LoRA(r=16, alpha=16, dropout=0.0, target_modules="all-linear", **kwargs)
```

Forwarded to `FastLanguageModel.get_peft_model`, with
`use_gradient_checkpointing="unsloth"` by default. `alpha` maps to `lora_alpha`
and `dropout` to `lora_dropout`. Passing both `alpha` and `lora_alpha` raises
`ValueError`.

## `Train`

```python
dq.Train(**kwargs)
```

Forwarded to the TRL config for the stage (`SFTConfig`, `DPOConfig`, or
`GRPOConfig`) after aliasing. There are no defaults; anything TRL accepts
works, including `optim`, `max_steps`, `beta`, `generations`,
`max_completion_length`, `warmup_ratio`, and so on. `save_every` on the stage
controls `save_strategy`/`save_steps`.

## `Load`

```python
dq.Load(**kwargs)
```

Forwarded to `FastLanguageModel.from_pretrained`. Defaults are
`max_seq_length=2048` and `load_in_4bit=True`; overrides are allowed:

```python
dq.Load(four_bit=False, max_seq_length=8192)
```

## Alias table

| Short name | Real name |
|---|---|
| `epochs` | `num_train_epochs` |
| `lr` | `learning_rate` |
| `batch` | `per_device_train_batch_size` |
| `grad_accum` | `gradient_accumulation_steps` |
| `generations` | `num_generations` |
| `alpha` | `lora_alpha` |
| `four_bit` | `load_in_4bit` |

Aliasing happens once, in `__init__`. If both the short and long name are
given, `ValueError` is raised; aliasing never silently overrides.

`dq.Train(lr=2e-4)` and `dq.Train(learning_rate=2e-4)` are equivalent and
produce the same config hash.

## Sequential

```python
dq.Sequential(
    name,
    repo=None,          # HF repo id; creates it private if missing
    base=None,          # base model name
    load=None,          # Load helper
    stages=None,        # list of stage objects, appended in order
    store=None,         # a Store instance instead of repo (e.g. LocalStore)
    full_state=True,    # save optimizer/scheduler/RNG state per checkpoint
    token=None,         # HF token; defaults to HF_TOKEN
    cache_dir=None,     # local cache root; defaults to /tmp/dqntrain
)
```
