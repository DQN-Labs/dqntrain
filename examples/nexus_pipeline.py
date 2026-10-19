"""End-to-end dqntrain example on a tiny model and small public datasets."""
import os

import dqntrain as dq
from dqntrain.rewards import contains_reward, format_reward

REPO = os.environ.get("DQNTRAIN_EXAMPLE_REPO", "your-name/dqntrain-nexus-demo")
PUSH_TO = os.environ.get("DQNTRAIN_EXAMPLE_PUSH_TO") or None
BASE = os.environ.get("DQNTRAIN_EXAMPLE_BASE", "unsloth/Qwen2.5-0.5B-Instruct")

sft = dq.SFT(
    "sft-code",
    data=dq.Data("trl-lib/Capybara", limit=64),
    lora=dq.LoRA(r=16, alpha=32),
    train=dq.Train(epochs=1, lr=2e-4, batch=1, grad_accum=4, logging_steps=1),
    save_every=10,
)

dpo = dq.DPO(
    "dpo-prefs",
    data=dq.Data("trl-lib/ultrafeedback_binarized", limit=64, map=lambda row: {"prompt": ""}),
    train=dq.Train(epochs=1, lr=5e-6, batch=1, grad_accum=4, beta=0.1, logging_steps=1),
    save_every=10,
)

grpo = dq.GRPO(
    "grpo-code",
    data=dq.Data("trl-lib/DeepMath-103K", columns={"answer": "solution"}, limit=64),
    reward_funcs=[format_reward(r"\\boxed\{.*\}"), contains_reward("\\boxed")],
    reward_weights=[0.2, 1.0],
    train=dq.Train(generations=4, max_completion_length=256, lr=5e-6, batch=1, grad_accum=4, logging_steps=1),
    save_every=10,
)

merge = dq.Merge("merged", push_to=PUSH_TO, dtype="bf16")

pipe = dq.Sequential(
    "nexus-27b-v1",
    repo=REPO,
    base=BASE,
    load=dq.Load(four_bit=True, max_seq_length=2048),
    stages=[sft, dpo, grpo, merge],
)

if __name__ == "__main__":
    dq.env_report()
    pipe.summary()
    pipe.run()
