# dqntrain

Simple, resumable training pipelines for transformers + PEFT + TRL. A pipeline
is a short list of stage objects, and the Hugging Face repo is the only memory:
a wiped notebook resumes exactly where it stopped.

```python
import dqntrain as dq

sft = dq.SFT("sft-code",
    data="org/code-instruct",
    train=dq.Train(epochs=2, lr=2e-4, batch=4),
    save_every=200,
)
pipe = dq.Sequential("nexus-v1", repo="you/nexus-ckpt",
    base="Qwen/Qwen3-4B", stages=[sft])
pipe.run()
```

## Design rules

- **The repo is the only memory.** Local disk is a cache. A fresh machine with
  an empty cache rebuilds the same state by calling `run()`.
- **Linear pipelines only.** Stages run in order. No branching or DAGs in v1.
- **transformers, PEFT, and TRL passthrough.** Options reach the model loader,
  `peft.get_peft_model`, and the TRL trainer config unchanged. Unknown keyword
  arguments are forwarded, never rejected.
- **Nothing private is hardcoded.** Repo names, tokens, and paths come from
  arguments or environment variables (`HF_TOKEN`).
- **Resume is automatic.** You never pass step numbers or checkpoint paths.
- **Plain functions over class hierarchies.** No plugin systems or registries.

## Install

=== "pip"

    ```bash
    pip install dqntrain
    ```

=== "from source"

    ```bash
    git clone https://github.com/DQN-Labs/dqntrain
    cd dqntrain
    pip install -e ".[test]"
    ```

`dqntrain` targets CUDA/ROCm machines. The training backend is plain
transformers + PEFT + TRL (no Unsloth); the core pipeline logic is
dependency-light and the test suite runs anywhere.

## Next steps

- [Quickstart](quickstart.md) — a full notebook pipeline end to end
- [Stages](stages.md) — SFT, DPO, GRPO, Merge, Eval, Custom
- [Resume and state](resume.md) — how the manifest, locks, and hashes work
- [API reference](api.md) — every public name at a glance
