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

    def merge_and_unload(self):
        self.merged_with = True
        return self


class MockTokenizer:
    def save_pretrained(self, folder):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "tokenizer.json"), "w") as handle:
            handle.write("{}")

    def apply_chat_template(self, messages, tokenize=False):
        return "\n".join(message["content"] for message in messages)


class MockBackend:
    calls = []

    @classmethod
    def reset(cls):
        cls.calls = []

    @classmethod
    def load_model(cls, name=None, dtype="bf16", max_seq_length=2048, **kwargs):
        cls.calls.append({"model_name": name, "dtype": dtype, "max_seq_length": max_seq_length, **kwargs})
        model = MockModel(name)
        if name and os.path.isfile(os.path.join(str(name), "adapter_config.json")):
            model.adapter = True
        return model, MockTokenizer()

    @classmethod
    def apply_lora(cls, model, cfg):
        cls.calls.append({"lora": cfg})
        model.adapter = True
        return model

    @classmethod
    def save_adapter(cls, model, path):
        cls.calls.append({"save_adapter": path})
        model.save_pretrained(path)

    @classmethod
    def merge_and_save(cls, model, path, dtype="bf16"):
        cls.calls.append({"merge_dtype": dtype})
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "merged.json"), "w") as handle:
            json.dump({"dtype": dtype}, handle)

    @classmethod
    def _load_adapter(cls, model, adapter_dir):
        cls.calls.append({"adapter": adapter_dir})
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
        with open(os.path.join(folder, "adapter_config.json"), "w") as handle:
            json.dump({"r": 16}, handle)
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
    MockBackend.reset()
    MockTrainer.all = []
    MockTrainer.kill_step = None
    MockTrainer.kill_kind = None

    monkeypatch.setattr(stages.backend, "load_model", MockBackend.load_model)
    monkeypatch.setattr(stages.backend, "apply_lora", MockBackend.apply_lora)
    monkeypatch.setattr(stages.backend, "save_adapter", MockBackend.save_adapter)
    monkeypatch.setattr(stages.backend, "merge_and_save", MockBackend.merge_and_save)
    monkeypatch.setattr(stages.backend, "_load_adapter", MockBackend._load_adapter)
    monkeypatch.setattr(
        stages, "_trainer_cls",
        lambda kind: type(f"Mock{kind}Trainer", (MockTrainer,), {"kind": kind}),
    )
    monkeypatch.setattr(stages, "_config_cls", lambda kind: MockConfig)
    return SimpleNamespace(
        trainer=MockTrainer,
        backend=MockBackend,
        model=MockModel,
        calls=MockBackend.calls,
    )
