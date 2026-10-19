# API reference

## Top-level exports

`import dqntrain as dq` exports exactly:

| Name | Kind | Purpose |
|---|---|---|
| `Sequential` | class | Pipeline: `add`, `run`, `summary`, `plan`, `status`, `rerun`, `reset` |
| `SFT`, `DPO`, `GRPO`, `Merge`, `Eval`, `Custom` | classes | Stages |
| `Data`, `LoRA`, `Train`, `Load` | classes | Config helpers |
| `env_report()` | function | Print detected environment versions |
| `__version__` | string | `"1.0.0"` |

## Stage constructors

```python
dq.SFT(name, data, lora=None, train=None, save_every=None, format="chatml")
dq.DPO(name, data, lora=None, train=None, save_every=None)
dq.GRPO(name, data, reward_funcs, reward_weights=None, lora=None, train=None, save_every=None)
dq.Merge(name, push_to=None, dtype="bf16")
dq.Eval(name, tasks, limit=None)
dq.Custom(name, fn, data=None, save_every=None)
```

## `Sequential`

```python
dq.Sequential(
    name, repo=None, base=None, load=None, stages=None, store=None,
    full_state=True, token=None, cache_dir=None,
)

pipe.add(*stages)             # append stages in order; returns self
pipe.run()                    # resume or start; safe to call again
pipe.summary()                # dataset rows, columns, planned actions; no model load
pipe.plan()                   # alias of summary()
pipe.status()                 # stage, status, latest checkpoint, config hash
pipe.rerun("stage-name")      # mark this stage and all later stages as pending
pipe.reset(confirm=True)      # delete the run folder in the store
```

## Config helpers

```python
dq.Data(name, split="train", subset=None, revision=None, seed=3407,
        limit=None, columns=None, filter=None, map=None, eval=None)

dq.LoRA(r=16, alpha=16, dropout=0.0, target_modules="all-linear", **kwargs)

dq.Train(**kwargs)

dq.Load(**kwargs)             # defaults: max_seq_length=2048, load_in_4bit=True
```

Alias table: `epochs`→`num_train_epochs`, `lr`→`learning_rate`,
`batch`→`per_device_train_batch_size`, `grad_accum`→`gradient_accumulation_steps`,
`generations`→`num_generations`, `alpha`→`lora_alpha`, `four_bit`→`load_in_4bit`.
See [Configuration](config.md).

## Rewards

```python
from dqntrain.rewards import (
    format_reward, exact_match_reward, contains_reward, length_penalty, _text,
)
```

See [Rewards](rewards.md).

## Stores

```python
from dqntrain.hub import Store, HFStore, LocalStore
```

See [Stores and testing](stores.md).

## Errors

```python
from dqntrain.errors import (
    LockHeldError,         # another live notebook holds the lock
    ConfigChangedError,    # a done/running stage hash changed; lists changed keys
    DataColumnError,       # required columns missing; lists required/missing/found
    DuplicateStageError,   # stage name repeated
)
```

## Custom stage context

```python
ctx.resume_dir                     # checkpoint folder to resume from, or None
ctx.run_dir                        # local working directory for the stage
ctx.save(step, checkpoint_dir=None)  # upload a checkpoint
ctx.log(dict)                      # append a line to logs/<stage>.jsonl
```

See [Custom stages](custom-stages.md).
