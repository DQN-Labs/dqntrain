# Stages

Every stage takes `name` as its first positional argument. Names must be
unique within a pipeline.

| Class | Arguments | Notes |
|---|---|---|
| `SFT` | `data`, `lora=None`, `train=None`, `save_every=None`, `format="chatml"`, `callbacks=None` | Uses `SFTTrainer`. `format` is `"chatml"`, `"text"`, or `"template"` |
| `DPO` | `data`, `lora=None`, `train=None`, `save_every=None`, `callbacks=None` | Uses `DPOTrainer`. `beta` lives in `train` |
| `GRPO` | `data`, `reward_funcs`, `reward_weights=None`, `lora=None`, `train=None`, `save_every=None`, `callbacks=None` | Uses `GRPOTrainer`. `reward_funcs` is a list of callables |
| `Merge` | `push_to=None`, `dtype="bf16"` | Merges the LoRA into the base and dequantizes. Optional upload to a separate repo |
| `Eval` | `tasks`, `limit=None` | Runs lm-eval tasks on the current model. Saves results as JSON in the repo |
| `Custom` | `fn`, `data=None`, `save_every=None` | `fn(model, tokenizer, data, ctx)` runs user training. Returns the model for the next stage |

## LoRA defaults

`lora=None` on `SFT`, `DPO`, or `GRPO` does **not** mean full fine-tuning: it
means LoRA with the `LoRA()` defaults (`r=16`, `lora_alpha=16`,
`lora_dropout=0.0`, `target_modules="all-linear"`,
`use_gradient_checkpointing="unsloth"`).

## Callbacks

`SFT`, `DPO`, and `GRPO` accept `callbacks=[...]`, a list of transformers
`TrainerCallback` objects. They run at every save, before dqntrain uploads the
checkpoint, so files a callback writes into the checkpoint folder (for example
an `eval.json`) are uploaded with it. The callback class source and instance
attributes are part of the stage hash.

## How stages chain

Each stage passes the previous stage's output to the next one: the in-memory
model when the pipeline is still running, or the previous stage's `final/`
artifact when a run restarts in a new process.

- After SFT/DPO/GRPO, `final/` is the LoRA adapter. A following training stage
  keeps training that adapter.
- After `Merge`, `final/` is a full merged model.
- A resumed stage first restores its own last checkpoint (adapter plus
  optimizer state); adapter weights from a checkpoint already include all
  earlier stages.

## SFT

```python
sft = dq.SFT("sft-code",
    data=dq.Data("org/code-instruct", columns={"messages": "conversation"}),
    lora=dq.LoRA(r=32, alpha=64),
    train=dq.Train(epochs=2, lr=2e-4, batch=4),
    save_every=200,
)
```

- `format="chatml"`: the dataset has a conversational `messages` column.
- `format="text"`: the dataset has a `text` column.
- `format="template"`: `messages` are rendered through the tokenizer's chat
  template into a `text` column before training.

## DPO

```python
dpo = dq.DPO("dpo-prefs",
    data="org/prefs",
    train=dq.Train(epochs=1, lr=5e-6, beta=0.1),
    save_every=100,
)
```

The data needs `prompt`, `chosen`, and `rejected` columns. With PEFT the
reference model is the base with the adapter disabled.

## GRPO

```python
grpo = dq.GRPO("grpo-code",
    data=dq.Data("org/code-problems", columns={"prompt": "question"}, limit=2000),
    reward_funcs=[format_reward(r"^[\s\S]*$"), contains_reward("def ")],
    reward_weights=[0.2, 1.0],
    train=dq.Train(generations=8, max_completion_length=512, lr=5e-6),
    save_every=50,
)
```

- `reward_funcs` is a list of callables following the reward
  [contract](rewards.md). Custom functions work the same as built-ins.
- `reward_weights` is forwarded to the GRPO config and is part of the config
  hash.
- Data needs a `prompt` column. All other columns are passed through to the
  reward functions as keyword arguments.

## Merge

```python
merge = dq.Merge("merged", push_to="DQN-Labs/nexus-27b", dtype="bf16")
```

Calls Unsloth's `save_pretrained_merged` on the current model, producing a
dequantized merged model in `final/`. `dtype` is `"bf16"`, `"fp16"`, or
`"fp32"` (merged 16-bit), or `"4bit"` (merged 4-bit). When `push_to` is set,
the merged model is uploaded to that repo; repos created this way are private.

When a run restarts mid-pipeline, adapters are loaded back with Unsloth's own
adapter loading (`FastLanguageModel.from_pretrained(adapter_dir)`), which keeps
the merged-save fast path working. If a model was attached through plain
`peft` instead, Unsloth refuses the merged save; the stage then falls back to
`merge_and_unload()` plus a standard `save_pretrained`.

## Eval

```python
eval_stage = dq.Eval("eval-arc", tasks=["arc_easy", "arc_challenge"], limit=200)
pipe.add(eval_stage)
```

Runs [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness), which is
an optional dependency:

```bash
pip install "dqntrain[eval]"
```

The model is evaluated through lm-eval's `hf` backend using the base model plus
the previous stage's adapter when one exists. Results are written to
`final/results.json` in the run folder.

## Custom

```python
def my_stage(model, tokenizer, data, ctx):
    trainer = MyTrainer(model=model, train_dataset=data["train"])
    trainer.train(resume_from_checkpoint=ctx.resume_dir)   # None on a fresh start
    ctx.save(trainer.state.global_step)                    # uploads a checkpoint
    return model                                           # passed to the next stage

custom = dq.Custom("my-stage", fn=my_stage, data="org/data")
```

See [Custom stages](custom-stages.md) for the `ctx` object.
