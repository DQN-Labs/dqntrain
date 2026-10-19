# Custom stages

A `Custom` stage runs your own training function. The signature is:

```python
fn(model, tokenizer, data, ctx) -> model
```

- `model`: the current model, already loaded and carrying the previous
  stage's adapter when one exists.
- `tokenizer`: the matching tokenizer.
- `data`: `None` when the stage was created without `data`; otherwise a dict
  `{"train": dataset, "eval": dataset or None}` built by the fixed
  [data pipeline](data.md).
- `ctx`: the context object below.
- The returned model (or `None` to keep the current one) is passed to the next
  stage.

## Example

```python
def my_stage(model, tokenizer, data, ctx):
    trainer = MyTrainer(model=model, train_dataset=data["train"])
    trainer.train(resume_from_checkpoint=ctx.resume_dir)   # None on a fresh start
    ctx.log({"trained_steps": trainer.state.global_step})
    ctx.save(trainer.state.global_step)                    # uploads a checkpoint
    return model

custom = dq.Custom("my-stage", fn=my_stage, data="org/data")
pipe.add(custom)
```

## The context object

| Attribute / method | Meaning |
|---|---|
| `ctx.resume_dir` | Local folder of the checkpoint to resume from, or `None` |
| `ctx.run_dir` | Local working directory for this stage; safe to write into |
| `ctx.save(step, checkpoint_dir=None)` | Upload a checkpoint using the same files-first, manifest-second protocol as built-in stages |
| `ctx.log(dict)` | Append one JSON line to `logs/<stage>.jsonl` in the store |

`ctx.save(step)` uploads `checkpoint_dir` when given, otherwise
`<run_dir>/checkpoint-<step>`, so save your trainer state into that folder
before calling it (or pass your own path).

## What the pipeline does around your function

- The stage is marked `running` and the lock heartbeat is refreshed before
  `fn` is called.
- `ctx.save` commits files first and the manifest entry second, exactly like
  built-in checkpointing. A crash in between is safe.
- After `fn` returns, the model is saved into `final/` and uploaded, unless it
  cannot be saved, in which case the stage is still marked `done`.
- The returned model stays in memory for the next stage. When a run restarts
  in a new process, the next stage loads `final/` if it exists.
