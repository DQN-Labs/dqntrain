import pytest

import dqntrain as dq
import dqntrain.stages as stages
from dqntrain.data import load_stage_data
from dqntrain.errors import DataColumnError
from dqntrain.hub import LocalStore


def grpo_rows():
    return [{"question": f"q{i}", "solution": f"s{i}"} for i in range(10)]


def test_rename_maps_expected_to_source(datasets):
    datasets("code", grpo_rows())
    data = dq.Data("code", columns={"prompt": "question", "answer": "solution"})
    loaded = load_stage_data(data, "GRPO")
    assert loaded["train"].column_names == ["prompt", "answer"]


def test_step_order_shuffle_limit_filter_map(datasets):
    datasets("rows", [{"messages": [{"role": "user", "content": str(i)}], "keep": i % 2 == 0} for i in range(10)])
    data = dq.Data(
        "rows",
        limit=6,
        seed=7,
        filter=lambda row: row["keep"],
        map=lambda row: {"seen": True},
    )
    loaded = load_stage_data(data, "SFT")
    assert len(loaded["train"]) == 3
    assert "seen" in loaded["train"].column_names


def test_missing_columns_raise_with_details(datasets):
    datasets("bad", [{"question": "q", "solution": "s"}])
    with pytest.raises(DataColumnError) as exc:
        load_stage_data(dq.Data("bad"), "DPO")
    message = str(exc.value)
    assert "prompt" in message and "chosen" in message and "rejected" in message
    assert "question" in message and "solution" in message
    assert exc.value.required == [["prompt", "chosen", "rejected"]]


def test_sft_requires_messages_or_text(datasets):
    datasets("textonly", [{"text": "hello"}])
    loaded = load_stage_data(dq.Data("textonly"), "SFT")
    assert loaded["train"].column_names == ["text"]
    datasets("neither", [{"foo": 1}])
    with pytest.raises(DataColumnError):
        load_stage_data(dq.Data("neither"), "SFT")


def test_eval_split_is_built(datasets):
    datasets("train-set", [{"messages": [{"role": "user", "content": "a"}]}])
    datasets("eval-set", [{"messages": [{"role": "user", "content": "b"}]}])
    data = dq.Data("train-set", eval={"name": "eval-set", "split": "test"})
    loaded = load_stage_data(data, "SFT")
    assert len(loaded["train"]) == 1
    assert len(loaded["eval"]) == 1


def test_summary_fails_before_model_load(tmp_path, datasets, hfakes, monkeypatch):
    datasets("bad", [{"foo": 1}])

    def explode():
        raise AssertionError("model must not be loaded in summary()")

    monkeypatch.setattr(stages.backend, "load_model", explode)
    pipe = dq.Sequential(
        "run", base="base", store=LocalStore(tmp_path / "store"),
        stages=[dq.SFT("sft", data=dq.Data("bad"))],
    )
    with pytest.raises(DataColumnError):
        pipe.summary()
    assert hfakes.calls == []
