# Quickstart

## Before you start

- A CUDA or ROCm machine with PyTorch and a Hugging Face-supported GPU
  (`transformers`, `peft`, and `trl` are installed with dqntrain).
- A Hugging Face token with write access in `HF_TOKEN`. The library reads it
  from the environment and never stores it in code, the manifest, or logs.

```bash
export HF_TOKEN=hf_...
```

## The notebook

```python
import dqntrain as dq
from dqntrain.rewards import format_reward, contains_reward

sft = dq.SFT("sft-code",
    data=dq.Data("org/code-instruct", columns={"messages": "conversation"}),
    lora=dq.LoRA(r=32, alpha=64),
    train=dq.Train(epochs=2, lr=2e-4, batch=4),
    save_every=200,
)

dpo = dq.DPO("dpo-prefs",
    data="org/prefs",                        # string shorthand for Data(name)
    train=dq.Train(epochs=1, lr=5e-6, beta=0.1),
    save_every=100,
)

grpo = dq.GRPO("grpo-code",
    data=dq.Data("org/code-problems",
                 columns={"prompt": "question", "answer": "solution"},
                 limit=2000),
    reward_funcs=[format_reward(r"^[\s\S]*$"), contains_reward("def ")],
    reward_weights=[0.2, 1.0],
    train=dq.Train(generations=8, max_completion_length=512, lr=5e-6),
    save_every=50,
)

merge = dq.Merge("merged", push_to="DQN-Labs/nexus-27b", dtype="bf16")

pipe = dq.Sequential("nexus-27b-v1",
    repo="DQN-Labs/nexus-ckpt",
    base="Qwen/Qwen3-4B",
    load=dq.Load(four_bit=True, max_seq_length=4096),
    stages=[sft, dpo, grpo, merge],
)
pipe.run()
```

Stages are defined separately and added in one line. `pipe.add(*stages)` works
too; both forms append in order. Stage names must be unique — a duplicate name
raises `DuplicateStageError` at construction or `add` time.

## Before running

```python
dq.env_report()      # python, torch, CUDA/ROCm, peft, trl, versions
pipe.summary()       # stage list, dataset row counts, columns, planned actions
pipe.plan()          # alias of summary()
```

`summary()` loads datasets to count rows and check columns, but never loads a
model and never trains. Missing required columns fail here, in seconds, before
a GPU is touched.

## After running (or after a crash)

```python
pipe.status()        # stage, status, latest checkpoint, config hash
pipe.run()           # safe to call again; resumes or starts
```

Every `run()` reconciles the pipeline config with the manifest:

| Situation | What `run()` does |
|---|---|
| Stage not in the manifest | adds it as `pending`, runs from scratch |
| Hash matches and status is `done` | skips |
| Hash differs on a `done` or `running` stage | raises `ConfigChangedError` listing the changed keys |
| Stage is `running` with checkpoints | downloads the last checkpoint and resumes |

Nothing is ever retrained silently. To accept a config change, call
`pipe.rerun("stage-name")` and run again; that stage and everything after it
will retrain.

## Example script

`examples/nexus_pipeline.py` in the repository is the same pipeline with a tiny
model (`Qwen/Qwen2.5-0.5B-Instruct`) and small public datasets, and runs end to
end on a single GPU.

## Test notebook

`examples/quick_test.ipynb` is a self-contained run on Qwen3-0.6B 4-bit that
exercises every stage type — SFT, DPO, GRPO, Custom (a hand-written trainer
step), Merge, and optionally Eval — with tiny datasets, in about 10 minutes.
Use it to smoke-test an environment or a change before a real run.

For a 2x T4 (Turing, fp16-only) rig, `examples/qwen3_0_6b_t4_smoke.ipynb`
is a shorter smoke test that also exercises crash/resume, config hashing and
rerun, lock takeover, multi-adapter switching, and an fp16 merge.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DQN-Labs/dqntrain/blob/main/examples/quick_test.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/kernels/welcome?src=https%3A%2F%2Fraw.githubusercontent.com%2FDQN-Labs%2Fdqntrain%2Fmain%2Fexamples%2Fquick_test.ipynb)

The setup cell installs dqntrain and its transformers/PEFT/TRL stack, reading
`HF_TOKEN` from the environment, Colab Secrets, or Kaggle secrets. Unsloth is
not required.
