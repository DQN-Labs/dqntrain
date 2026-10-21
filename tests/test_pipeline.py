import json
import os
import shutil
import sys
import types
from datetime import datetime, timedelta, timezone

import pytest

import dqntrain as dq
import dqntrain.stages as stages
from dqntrain.errors import ConfigChangedError, DuplicateStageError, LockHeldError
from dqntrain.hub import HFStore, LocalStore
from dqntrain.state import utcnow

SFT_ROWS = [
    {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}
    for i in range(8)
]
DPO_ROWS = [{"prompt": f"p{i}", "chosen": "good", "rejected": "bad"} for i in range(8)]


def sft_stage(name="sft", save_every=2, **train):
    train.setdefault("max_steps", 6)
    return dq.SFT(name, data=dq.Data("sft-data"), train=dq.Train(**train), save_every=save_every)


def dpo_stage(name="dpo", save_every=2, **train):
    train.setdefault("max_steps", 4)
    return dq.DPO(name, data=dq.Data("dpo-data"), train=dq.Train(**train), save_every=save_every)


def make_pipe(name, store, stages, cache, **kwargs):
    return dq.Sequential(name, base="base", store=store, stages=stages, cache_dir=str(cache), **kwargs)


def test_kill_and_resume_losses_match(tmp_path, datasets, hfakes):
    datasets("sft-data", SFT_ROWS)
    store = LocalStore(tmp_path / "store")
    cache = tmp_path / "cache"
    hfakes.trainer.kill_step = 2
    pipe = make_pipe("run", store, [sft_stage()], cache)
    with pytest.raises(RuntimeError):
        pipe.run()
    assert hfakes.trainer.all[0].state.global_step == 2
    hfakes.trainer.kill_step = None

    resumed_pipe = make_pipe("run", store, [sft_stage()], cache)
    resumed_pipe.run()
    resumed = hfakes.trainer.all[1]
    assert resumed.resume_from.endswith("ckpt-000002")
    losses_after = [record["loss"] for record in resumed.state.log_history]

    full_pipe = make_pipe("uninterrupted", store, [sft_stage()], cache)
    full_pipe.run()
    full = hfakes.trainer.all[2]
    expected = [record["loss"] for record in full.state.log_history if record["step"] > 2]
    assert losses_after == pytest.approx(expected)

    manifest = store.read_manifest("run")
    assert manifest["stages"][0]["status"] == "done"
    assert [c["path"] for c in manifest["stages"][0]["checkpoints"]] == [
        "sft/ckpt-000002", "sft/ckpt-000004", "sft/ckpt-000006",
    ]


def test_half_finished_upload_is_ignored(tmp_path, datasets, hfakes, flaky_store):
    datasets("sft-data", SFT_ROWS)
    store = flaky_store(tmp_path / "store")
    store.fail_manifest_at_checkpoints = 2
    cache = tmp_path / "cache"
    pipe = make_pipe("run", store, [sft_stage()], cache)
    with pytest.raises(IOError):
        pipe.run()
    assert (tmp_path / "store" / "run" / "sft" / "ckpt-000004").is_dir()
    record = store.read_manifest("run")["stages"][0]
    assert [c["path"] for c in record["checkpoints"]] == ["sft/ckpt-000002"]
    assert record["status"] == "running"

    restarted = make_pipe("run", store, [sft_stage()], cache)
    restarted.run()
    resumed = hfakes.trainer.all[-1]
    assert resumed.resume_from.endswith("ckpt-000002")
    assert store.read_manifest("run")["stages"][0]["status"] == "done"


def test_two_stages_resume_without_retraining_first(tmp_path, datasets, hfakes):
    datasets("sft-data", SFT_ROWS)
    datasets("dpo-data", DPO_ROWS)
    store = LocalStore(tmp_path / "store")
    cache = tmp_path / "cache"
    hfakes.trainer.kill_step = 2
    hfakes.trainer.kill_kind = "DPO"
    pipe = make_pipe("run", store, [sft_stage(), dpo_stage()], cache)
    with pytest.raises(RuntimeError):
        pipe.run()
    assert [t.kind for t in hfakes.trainer.all] == ["SFT", "DPO"]
    hfakes.trainer.kill_step = None
    hfakes.trainer.kill_kind = None

    restarted = make_pipe("run", store, [sft_stage(), dpo_stage()], cache)
    restarted.run()
    assert [t.kind for t in hfakes.trainer.all] == ["SFT", "DPO", "DPO"]
    resumed = hfakes.trainer.all[-1]
    assert resumed.resume_from.endswith("dpo/ckpt-000002")
    assert hfakes.calls[-1]["adapter"].endswith("dpo/ckpt-000002")
    assert resumed.model.adapter is True
    assert (cache / "run" / "sft" / "final" / "adapter_config.json").exists()
    assert sum(1 for call in hfakes.calls if "adapter" in call) == 1
    assert store.has_dir("run", "sft/final")


def test_config_change_raises_then_rerun(tmp_path, datasets, hfakes):
    datasets("sft-data", SFT_ROWS)
    store = LocalStore(tmp_path / "store")
    cache = tmp_path / "cache"
    make_pipe("run", store, [sft_stage(learning_rate=2e-4)], cache).run()

    changed = make_pipe("run", store, [sft_stage(learning_rate=1e-5)], cache)
    with pytest.raises(ConfigChangedError) as exc:
        changed.run()
    message = str(exc.value)
    assert "learning_rate" in message
    assert "0.0002" in message and "1e-05" in message

    changed.rerun("sft")
    changed.run()
    assert len(hfakes.trainer.all) == 2
    assert hfakes.trainer.all[-1].resume_from is None


def test_lock_held_and_stale_takeover(tmp_path, datasets, hfakes):
    datasets("sft-data", SFT_ROWS)
    store = LocalStore(tmp_path / "store")
    cache = tmp_path / "cache"
    make_pipe("run", store, [sft_stage()], cache).run()

    manifest = store.read_manifest("run")
    manifest["lock"] = {"holder": "notebook-other", "heartbeat": utcnow()}
    store.write_manifest("run", manifest)
    blocked = make_pipe("run", store, [sft_stage()], cache)
    with pytest.raises(LockHeldError) as exc:
        blocked.run()
    assert "notebook-other" in str(exc.value)

    stale = (datetime.now(timezone.utc) - timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["lock"] = {"holder": "notebook-ghost", "heartbeat": stale}
    store.write_manifest("run", manifest)
    taker = make_pipe("run", store, [sft_stage()], cache)
    taker.run()
    log = (tmp_path / "store" / "run" / "logs" / "_pipeline.jsonl").read_text()
    assert "lock_takeover" in log and "notebook-ghost" in log


def test_passthrough_reaches_trainer_config_and_changes_hash(tmp_path, datasets, hfakes):
    datasets("sft-data", SFT_ROWS)
    store = LocalStore(tmp_path / "store")
    cache = tmp_path / "cache"
    stage = sft_stage(optim="adamw_8bit")
    pipe = make_pipe("run", store, [stage], cache)
    baseline = make_pipe("other", store, [sft_stage()], cache)
    assert pipe._hashes()[0]["hash"] != baseline._hashes()[0]["hash"]

    pipe.run()
    trainer = hfakes.trainer.all[-1]
    assert trainer.args.optim == "adamw_8bit"
    assert trainer.args.max_steps == 6


def test_empty_machine_rebuilds_from_store(tmp_path, datasets, hfakes):
    datasets("sft-data", SFT_ROWS)
    store = LocalStore(tmp_path / "store")
    cache = tmp_path / "cache"
    hfakes.trainer.kill_step = 2
    with pytest.raises(RuntimeError):
        make_pipe("run", store, [sft_stage()], cache).run()
    hfakes.trainer.kill_step = None

    shutil.rmtree(cache)
    assert not cache.exists()

    make_pipe("run", store, [sft_stage()], cache).run()
    assert hfakes.trainer.all[-1].resume_from.endswith("ckpt-000002")
    manifest = store.read_manifest("run")
    assert manifest["stages"][0]["status"] == "done"
    assert store.has_dir("run", "sft/final")


def test_duplicate_stage_names(tmp_path):
    store = LocalStore(tmp_path / "store")
    with pytest.raises(DuplicateStageError):
        dq.Sequential("run", base="base", store=store, stages=[sft_stage(), sft_stage()])
    pipe = dq.Sequential("run", base="base", store=store, stages=[sft_stage()])
    with pytest.raises(DuplicateStageError):
        pipe.add(sft_stage())


def test_adapter_only_checkpoints(tmp_path, datasets, hfakes):
    datasets("sft-data", SFT_ROWS)
    store = LocalStore(tmp_path / "store")
    pipe = make_pipe("run", store, [sft_stage()], tmp_path / "cache", full_state=False)
    pipe.run()
    checkpoint = tmp_path / "store" / "run" / "sft" / "ckpt-000002"
    assert (checkpoint / "adapter_model.safetensors").exists()
    assert not (checkpoint / "optimizer.pt").exists()
    assert hfakes.trainer.all[-1].args.save_only_model is True


def test_grpo_rewards_reach_trainer(tmp_path, datasets, hfakes):
    datasets("grpo-data", [{"prompt": "q"}])
    store = LocalStore(tmp_path / "store")
    reward = lambda completions, **kwargs: [1.0 for _ in completions]
    stage = dq.GRPO(
        "grpo", data=dq.Data("grpo-data"), reward_funcs=[reward], reward_weights=[1.0],
        train=dq.Train(max_steps=2, generations=4), save_every=2,
    )
    make_pipe("run", store, [stage], tmp_path / "cache").run()
    trainer = hfakes.trainer.all[-1]
    assert trainer.extra["reward_funcs"] == [reward]
    assert trainer.args.reward_weights == [1.0]
    assert trainer.args.num_generations == 4


def test_summary_actions_and_hash_warning(tmp_path, datasets, hfakes, capsys):
    datasets("sft-data", SFT_ROWS)
    datasets("grpo-data", [{"prompt": "q1"}, {"prompt": "q2"}])
    store = LocalStore(tmp_path / "store")
    fn = eval("lambda completions, **kwargs: [1.0 for _ in completions]")
    grpo = dq.GRPO("grpo", data=dq.Data("grpo-data"), reward_funcs=[fn], train=dq.Train(max_steps=2))
    pipe = make_pipe("run", store, [sft_stage(), grpo], tmp_path / "cache")
    pipe.summary()
    out = capsys.readouterr().out
    assert "run from scratch" in out
    assert "WARNING" in out and "lambda" in out


def test_custom_stage_saves_and_logs(tmp_path, datasets, hfakes):
    datasets("sft-data", SFT_ROWS)
    store = LocalStore(tmp_path / "store")
    calls = []

    def custom_fn(model, tokenizer, data, ctx):
        calls.append((model, data, ctx.run_dir, ctx.resume_dir))
        ctx.log({"event": "custom"})
        ctx.save(3)
        return model

    pipe = make_pipe("run", store, [sft_stage(save_every=None), dq.Custom("custom", fn=custom_fn)], tmp_path / "cache")
    pipe.run()
    assert calls and calls[0][1] is None
    record = store.read_manifest("run")["stages"][1]
    assert [c["step"] for c in record["checkpoints"]] == [3]
    assert record["status"] == "done"
    log = (tmp_path / "store" / "run" / "logs" / "custom.jsonl").read_text()
    assert "custom" in log


def test_merge_stage_uploads_merged_model(tmp_path, datasets, hfakes):
    datasets("sft-data", SFT_ROWS)
    store = LocalStore(tmp_path / "store")
    pipe = make_pipe("run", store, [sft_stage(save_every=None), dq.Merge("merge", dtype="bf16")], tmp_path / "cache")
    pipe.run()
    merged = tmp_path / "store" / "run" / "merge" / "final" / "merged.json"
    assert merged.exists()
    assert json.loads(merged.read_text())["dtype"] == "bf16"
    assert (tmp_path / "store" / "run" / "merge" / "final" / "tokenizer.json").exists()
    assert hfakes.calls[-1]["merge_dtype"] == "bf16"
    assert store.read_manifest("run")["stages"][1]["status"] == "done"


def test_checkpoint_meta_records_backend_versions(tmp_path, datasets, hfakes):
    datasets("sft-data", SFT_ROWS)
    store = LocalStore(tmp_path / "store")
    make_pipe("run", store, [sft_stage()], tmp_path / "cache").run()
    meta = json.loads((tmp_path / "store" / "run" / "sft" / "ckpt-000002" / "meta.json").read_text())
    for key in ("torch", "transformers", "trl", "peft", "dqntrain"):
        assert key in meta


def test_adapter_resume_uses_peft_loading(tmp_path, datasets, hfakes):
    datasets("sft-data", SFT_ROWS)
    datasets("dpo-data", DPO_ROWS)
    store = LocalStore(tmp_path / "store")
    cache = tmp_path / "cache"
    make_pipe("run", store, [sft_stage(save_every=None)], cache).run()

    adapter_dir = cache / "run" / "sft" / "final"
    assert (adapter_dir / "adapter_config.json").exists()

    pipe = make_pipe("run", store, [sft_stage(save_every=None), dpo_stage()], cache)
    pipe.run()
    assert hfakes.calls[-1]["adapter"] == str(adapter_dir)
    assert hfakes.trainer.all[-1].model.adapter is True


def test_local_upload_skips_hidden_files(tmp_path):
    store = LocalStore(tmp_path / "store")
    source = tmp_path / "source"
    (source / ".cache" / "huggingface").mkdir(parents=True)
    (source / ".cache" / "huggingface" / ".gitignore").write_text("junk")
    (source / ".hidden").write_text("junk")
    (source / "model.safetensors").write_text("weights")
    store.upload_dir("run", "stage/final", str(source), "message")
    dest = tmp_path / "store" / "run" / "stage" / "final"
    assert (dest / "model.safetensors").exists()
    assert not (dest / ".cache").exists()
    assert not (dest / ".hidden").exists()


def test_hf_upload_skips_hidden_files(tmp_path):
    added = []

    class FakeApi:
        def create_commit(self, repo, ops, commit_message):
            added.extend(op.path_in_repo for op in ops)

    store = HFStore.__new__(HFStore)
    store.repo = "org/repo"
    store.token = None
    store.api = FakeApi()
    source = tmp_path / "source"
    (source / ".cache" / "huggingface").mkdir(parents=True)
    (source / ".cache" / "huggingface" / ".gitignore").write_text("junk")
    (source / "model.safetensors").write_text("weights")
    store.upload_dir("run", "stage/final", str(source), "message")
    assert added == ["run/stage/final/model.safetensors"]


def test_stage_callbacks_run_before_upload(tmp_path, datasets, hfakes):
    datasets("sft-data", SFT_ROWS)
    store = LocalStore(tmp_path / "store")
    seen = []

    class Recorder:
        def __init__(self, tag):
            self.tag = tag

        def on_save(self, args, state, control, **kwargs):
            seen.append(state.global_step)
            folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            with open(os.path.join(folder, "eval.json"), "w") as handle:
                json.dump({"tag": self.tag}, handle)

    stage = dq.SFT(
        "sft", data=dq.Data("sft-data"), train=dq.Train(max_steps=6), save_every=2,
        callbacks=[Recorder("sft")],
    )
    make_pipe("run", store, [stage], tmp_path / "cache").run()

    assert seen == [2, 4, 6]
    for step in (2, 4, 6):
        uploaded = tmp_path / "store" / "run" / "sft" / f"ckpt-00000{step}" / "eval.json"
        assert json.loads(uploaded.read_text()) == {"tag": "sft"}


def test_eval_stage_writes_results(tmp_path, hfakes, monkeypatch):
    lm_eval = types.ModuleType("lm_eval")
    lm_eval.simple_evaluate = lambda **kwargs: {"results": {"arc": 0.5}, "call_args": kwargs["tasks"]}
    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval)
    store = LocalStore(tmp_path / "store")
    pipe = make_pipe("run", store, [dq.Eval("eval", tasks=["arc"], limit=4)], tmp_path / "cache")
    pipe.run()
    results = json.loads((tmp_path / "store" / "run" / "eval" / "final" / "results.json").read_text())
    assert results["call_args"] == ["arc"]
    assert store.read_manifest("run")["stages"][0]["status"] == "done"


def test_status_and_reset(tmp_path, datasets, hfakes, capsys):
    datasets("sft-data", SFT_ROWS)
    store = LocalStore(tmp_path / "store")
    cache = tmp_path / "cache"
    pipe = make_pipe("run", store, [sft_stage()], cache)
    pipe.run()
    pipe.status()
    out = capsys.readouterr().out
    assert "sft" in out and "done" in out and "ckpt-000006" in out

    with pytest.raises(ValueError):
        pipe.reset()
    pipe.reset(confirm=True)
    assert not (tmp_path / "store" / "run").exists()
    assert not (cache / "run").exists()
