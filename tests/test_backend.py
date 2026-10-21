import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("tokenizers")

import dqntrain.backend as backend


def make_tiny_model(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "hello": 3, "world": 4, "hi": 5}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    config = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config)
    folder = tmp_path / "tiny"
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return str(folder)


def lora_config():
    return {
        "r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
        "use_gradient_checkpointing": False,
    }


def test_load_model_apply_lora_generate(tmp_path, capsys):
    folder = make_tiny_model(tmp_path)
    model, tokenizer = backend.load_model(folder, dtype="fp32", max_seq_length=64)
    assert type(model).__name__ == "LlamaForCausalLM"
    assert "LlamaForCausalLM" in capsys.readouterr().out
    assert tokenizer.model_max_length == 64

    model = backend.apply_lora(model, lora_config())
    assert hasattr(model, "peft_config")
    texts = backend.generate(model, tokenizer, ["hello world"], max_new_tokens=2)
    assert isinstance(texts, list) and len(texts) == 1 and isinstance(texts[0], str)


def test_save_adapter_and_merge_and_save(tmp_path):
    folder = make_tiny_model(tmp_path)
    model, tokenizer = backend.load_model(folder, dtype="fp32", max_seq_length=64)
    model = backend.apply_lora(model, lora_config())

    adapter_dir = tmp_path / "adapter"
    backend.save_adapter(model, str(adapter_dir))
    assert (adapter_dir / "adapter_config.json").exists()
    assert (adapter_dir / "adapter_model.safetensors").exists()

    merged_dir = tmp_path / "merged"
    backend.merge_and_save(model, str(merged_dir), "fp32")
    assert (merged_dir / "config.json").exists()
    assert (merged_dir / "model.safetensors").exists()
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(merged_dir)
    assert type(reloaded).__name__ == "LlamaForCausalLM"

    with pytest.raises(ValueError):
        backend.merge_and_save(model, str(tmp_path / "four"), "4bit")


def test_adapter_switching(tmp_path):
    from peft import LoraConfig

    folder = make_tiny_model(tmp_path)
    model, tokenizer = backend.load_model(folder, dtype="fp32", max_seq_length=64)
    model = backend.apply_lora(model, lora_config())
    model.add_adapter(
        "teacher",
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], task_type="CAUSAL_LM"),
    )
    assert set(model.peft_config) == {"default", "teacher"}

    model.set_adapter("teacher")
    with model.disable_adapter():
        with torch.no_grad():
            model(**tokenizer("hi", return_tensors="pt"))
    model.set_adapter("default")
    text = backend.generate(model, tokenizer, ["hi"], max_new_tokens=1)
    assert isinstance(text[0], str)
    model.delete_adapter("teacher")
    assert set(model.peft_config) == {"default"}


def test_load_model_falls_back_to_image_text(tmp_path, monkeypatch, capsys):
    folder = make_tiny_model(tmp_path)
    model, _ = backend.load_model(folder, dtype="fp32", max_seq_length=64)
    capsys.readouterr()
    calls = []

    class FailingCausal:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append("causal")
            raise ValueError("this checkpoint needs a vision class")

    class FakeVision:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append("vision")
            return model

    monkeypatch.setattr(backend, "_model_classes", lambda: (FailingCausal, FakeVision))
    loaded, _ = backend.load_model(folder, dtype="fp32", max_seq_length=64)
    assert calls == ["causal", "vision"]
    assert loaded is model
    assert "FakeVision" in capsys.readouterr().out


def test_4bit_without_bitsandbytes_raises(tmp_path, monkeypatch):
    folder = make_tiny_model(tmp_path)
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    with pytest.raises(RuntimeError, match="bitsandbytes"):
        backend.load_model(folder, dtype="fp32", load_in_4bit=True)


def test_4bit_compute_dtype_follows_load_dtype(monkeypatch):
    import types

    monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("transformers.BitsAndBytesConfig", FakeConfig)
    assert backend._quantization_config(torch.float16).kwargs["bnb_4bit_compute_dtype"] is torch.float16
    assert backend._quantization_config().kwargs["bnb_4bit_compute_dtype"] is torch.bfloat16
