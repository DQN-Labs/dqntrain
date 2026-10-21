import pytest

import dqntrain as dq
from dqntrain.hub import LocalStore


def test_alias_values_and_conflict():
    assert dq.Train(lr=2e-4).values == {"learning_rate": 2e-4}
    assert dq.Train(learning_rate=2e-4).values == {"learning_rate": 2e-4}
    assert dq.Train(epochs=2, batch=4, grad_accum=2).values == {
        "num_train_epochs": 2,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
    }
    with pytest.raises(ValueError):
        dq.Train(lr=1e-4, learning_rate=2e-4)


def test_train_accepts_plain_dict():
    values = {"epochs": 3, "lr": 1e-5, "optim": "adamw_8bit"}
    assert dq.Train(**values).values == {
        "num_train_epochs": 3,
        "learning_rate": 1e-5,
        "optim": "adamw_8bit",
    }


def test_alias_hash_equivalence(tmp_path):
    store = LocalStore(tmp_path / "store")
    short = dq.Sequential(
        "run", base="base", store=store,
        stages=[dq.SFT("sft", data=dq.Data("x"), train=dq.Train(lr=2e-4))],
    )
    long = dq.Sequential(
        "run", base="base", store=store,
        stages=[dq.SFT("sft", data=dq.Data("x"), train=dq.Train(learning_rate=2e-4))],
    )
    assert short._hashes()[0]["hash"] == long._hashes()[0]["hash"]


def test_callback_hash_changes_stage_hash(tmp_path):
    class RecorderA:
        def __init__(self, tag="a"):
            self.tag = tag

        def on_save(self, *args, **kwargs):
            pass

    class RecorderB:
        def on_save(self, *args, **kwargs):
            pass

    store = LocalStore(tmp_path / "store")

    def pipe(callbacks):
        return dq.Sequential(
            "run", base="base", store=store,
            stages=[dq.SFT("sft", data=dq.Data("x"), train=dq.Train(lr=2e-4), callbacks=callbacks)],
        )

    plain = pipe(None)._hashes()[0]["hash"]
    first = pipe([RecorderA()])._hashes()[0]["hash"]
    second = pipe([RecorderA()])._hashes()[0]["hash"]
    other = pipe([RecorderB()])._hashes()[0]["hash"]
    changed_attr = pipe([RecorderA("b")])._hashes()[0]["hash"]
    assert first == second
    assert first != plain
    assert other != first
    assert changed_attr != first


def test_lora_defaults_and_aliases():
    assert dq.LoRA().values == {
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": "all-linear",
        "use_gradient_checkpointing": True,
    }
    assert dq.LoRA(r=32, alpha=64).values["lora_alpha"] == 64
    assert dq.LoRA(dropout=0.1, target_modules=["q_proj"]).values["lora_dropout"] == 0.1
    with pytest.raises(ValueError):
        dq.LoRA(alpha=32, lora_alpha=64)


def test_load_defaults_and_aliases():
    assert dq.Load().values == {"max_seq_length": 2048, "load_in_4bit": False}
    assert dq.Load(four_bit=False).values["load_in_4bit"] is False
    assert dq.Load(load_in_4bit=True).values["load_in_4bit"] is True
    assert dq.Load(max_seq_length=4096).values["max_seq_length"] == 4096


def test_data_shorthand_and_eval():
    data = dq.Data("org/set", eval={"name": "org/set-eval", "split": "test", "limit": 10})
    assert data.eval.name == "org/set-eval"
    assert data.eval.split == "test"
    assert data.eval.limit == 10
