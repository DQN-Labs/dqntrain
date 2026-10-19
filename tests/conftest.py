import json
import os
import random
from types import SimpleNamespace

import pytest

import dqntrain.stages as stages
from dqntrain.hub import LocalStore


class FakeDataset:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]

    @property
    def column_names(self):
        return list(self.rows[0].keys()) if self.rows else []

    def __len__(self):
        return len(self.rows)

    def shuffle(self, seed=0):
        index = list(range(len(self.rows)))
        random.Random(seed).shuffle(index)
        return FakeDataset([self.rows[i] for i in index])

    def select(self, indices):
        return FakeDataset([self.rows[i] for i in indices])

    def filter(self, fn):
        return FakeDataset([row for row in self.rows if fn(row)])

    def map(self, fn):
        out = []
        for row in self.rows:
            merged = dict(row)
            merged.update(fn(dict(row)))
            out.append(merged)
        return FakeDataset(out)

    def rename_columns(self, mapping):
        return FakeDataset([{mapping.get(key, key): value for key, value in row.items()} for row in self.rows])


class MockModel:
    def __init__(self, name="model"):
        self.name = name
        self.adapter = False
        self.merged_with = None

    def save_pretrained(self, folder):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "adapter_config.json"), "w") as handle:
            json.dump({"r": 16}, handle)

    def save_pretrained_merged(self, folder, tokenizer, save_method="merged_16bit"):
        os.makedirs(folder, exist_ok=True)
        self.merged_with = save_method
        with open(os.path.join(folder, "merged.json"), "w") as handle:
            json.dump({"method": save_method}, handle)

    def merge_and_unload(self):
        return self


class MockTokenizer:
    def save_pretrained(self, folder):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "tokenizer.json"), "w") as handle:
            handle.write("{}")

    def apply_chat_template(self, messages, tokenize=False):
        return "\n".join(message["content"] for message in messages)


class MockFastLanguageModel:
    calls = []

    @classmethod
    def from_pretrained(cls, model_name=None, **kwargs):
        cls.calls.append({"model_name": model_name, **kwargs})
        return MockModel(model_name), MockTokenizer()

    @classmethod
    def get_peft_model(cls, model, **kwargs):
        cls.calls.append({"peft": kwargs})
        model.adapter = True
        return model


class MockConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.kwargs = kwargs


class MockTrainer:
    all = []
    kill_step = None
    kill_kind = None
    kind = "?"

    def __init__(self, model=None, args=None, train_dataset=None, eval_dataset=None,
                 callbacks=None, **kwargs):
        self.kind = type(self).kind
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.callbacks = callbacks or []
        self.extra = kwargs
        self.state = SimpleNamespace(global_step=0, log_history=[])
        self.resume_from = None
        MockTrainer.all.append(self)

    def train(self, resume_from_checkpoint=None):
        self.resume_from = resume_from_checkpoint
        start = 0
        if resume_from_checkpoint:
            with open(os.path.join(resume_from_checkpoint, "trainer_state.json")) as handle:
                start = json.load(handle)["global_step"]
        total = getattr(self.args, "max_steps", None) or 6
        for step in range(start + 1, total + 1):
            self.state.global_step = step
            self.state.log_history.append({"loss": 1.0 / step, "step": step})
            save_steps = getattr(self.args, "save_steps", None)
            if save_steps and step % save_steps == 0:
                self._save(step)
        return SimpleNamespace(global_step=self.state.global_step)

    def _save(self, step):
        folder = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "trainer_state.json"), "w") as handle:
            json.dump({"global_step": step}, handle)
        with open(os.path.join(folder, "adapter_model.safetensors"), "w") as handle:
            handle.write("weights")
        if not getattr(self.args, "save_only_model", False):
            with open(os.path.join(folder, "optimizer.pt"), "w") as handle:
                handle.write("optimizer")
        for callback in self.callbacks:
            callback.on_save(self.args, self.state, None)
        if MockTrainer.kill_step == step and MockTrainer.kill_kind in (None, self.kind):
            MockTrainer.kill_step = None
            raise RuntimeError("simulated crash")

    def save_model(self, folder):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "adapter_model.safetensors"), "w") as handle:
            handle.write("weights")
        with open(os.path.join(folder, "adapter_config.json"), "w") as handle:
            json.dump({"r": 16}, handle)


class FlakyStore(LocalStore):
    fail_manifest_at_checkpoints = None

    def write_manifest(self, run, manifest):
        total = sum(len(record.get("checkpoints") or []) for record in manifest.get("stages", []))
        if self.fail_manifest_at_checkpoints is not None and total >= self.fail_manifest_at_checkpoints:
            self.fail_manifest_at_checkpoints = None
            raise IOError("simulated upload failure")
        super().write_manifest(run, manifest)


@pytest.fixture
def flaky_store():
    return FlakyStore


@pytest.fixture
def datasets(monkeypatch):
    registry = {}

    def register(name, rows):
        registry[name] = FakeDataset(rows)
        return registry[name]

    monkeypatch.setattr("dqntrain.data._load_dataset", lambda config: registry[config.name])
    return register


@pytest.fixture
def hfakes(monkeypatch):
    MockFastLanguageModel.calls = []
    MockTrainer.all = []
    MockTrainer.kill_step = None
    MockTrainer.kill_kind = None
    adapter_calls = []

    def fake_load_adapter(model, adapter_dir):
        adapter_calls.append(adapter_dir)
        model.adapter = True
        return model

    monkeypatch.setattr(stages, "_fast_language_model", lambda: MockFastLanguageModel)
    monkeypatch.setattr(stages, "_load_adapter", fake_load_adapter)
    monkeypatch.setattr(
        stages, "_trainer_cls",
        lambda kind: type(f"Mock{kind}Trainer", (MockTrainer,), {"kind": kind}),
    )
    monkeypatch.setattr(stages, "_config_cls", lambda kind: MockConfig)
    return SimpleNamespace(
        trainer=MockTrainer,
        flm=MockFastLanguageModel,
        adapter_calls=adapter_calls,
    )
