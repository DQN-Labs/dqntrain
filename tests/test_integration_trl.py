import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DQNTRAIN_TEST_NETWORK") != "1",
    reason="set DQNTRAIN_TEST_NETWORK=1 to run the real-TRL integration test",
)


def test_real_trl_sft_checkpoint_and_resume(tmp_path, monkeypatch):
    datasets = pytest.importorskip("datasets")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("trl")

    import dqntrain as dq
    import dqntrain.data as data_mod
    import dqntrain.stages as stages
    from dqntrain.hub import LocalStore

    model_name = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

    class FakeFLM:
        @staticmethod
        def from_pretrained(model_name=None, **kwargs):
            return model, tokenizer

        @staticmethod
        def get_peft_model(inner, **kwargs):
            return inner

    monkeypatch.setattr(stages, "_fast_language_model", lambda: FakeFLM)
    monkeypatch.setattr(stages, "_load_adapter", lambda inner, folder: inner)
    rows = [[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]] * 8
    monkeypatch.setattr(
        data_mod, "_load_dataset",
        lambda config: datasets.Dataset.from_dict({"messages": rows}),
    )

    store = LocalStore(tmp_path / "store")
    cache = tmp_path / "cache"
    stage = dq.SFT(
        "sft",
        data=dq.Data("tiny"),
        train=dq.Train(max_steps=4, batch=1, lr=2e-5, bf16=False, use_cpu=True,
                       report_to=[], logging_steps=1),
        save_every=2,
    )
    dq.Sequential("run", base=model_name, store=store, stages=[stage], cache_dir=str(cache)).run()

    manifest = store.read_manifest("run")
    assert manifest["stages"][0]["status"] == "done"
    checkpoint = tmp_path / "store" / "run" / "sft" / "ckpt-000002"
    assert (checkpoint / "optimizer.pt").exists()
    assert (checkpoint / "scheduler.pt").exists()
    assert (checkpoint / "trainer_state.json").exists()
    assert (checkpoint / "meta.json").exists()
    assert json.loads((checkpoint / "meta.json").read_text())["torch"]

    manifest["stages"][0]["checkpoints"] = manifest["stages"][0]["checkpoints"][:1]
    manifest["stages"][0]["status"] = "running"
    store.write_manifest("run", manifest)

    dq.Sequential("run", base=model_name, store=store, stages=[stage], cache_dir=str(cache)).run()
    manifest = store.read_manifest("run")
    assert manifest["stages"][0]["status"] == "done"
    assert [c["path"] for c in manifest["stages"][0]["checkpoints"]] == [
        "sft/ckpt-000002", "sft/ckpt-000004",
    ]
    assert (tmp_path / "store" / "run" / "sft" / "final").is_dir()
