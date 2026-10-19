# Environment

## `dq.env_report()`

Prints detected versions of Python, torch, CUDA or ROCm (with the HIP version
when present), MPS when available, unsloth, bitsandbytes, trl, transformers,
datasets, huggingface_hub, peft, and dqntrain. Missing packages print a
warning; the call never fails.

```python
import dqntrain as dq

dq.env_report()
```

```
python: 3.10.12
torch: 2.11.0
cuda: None
rocm: None
unsloth: 2025.1.3
bitsandbytes: 0.45.0
trl: 1.2.0
transformers: 4.49.0
datasets: 4.8.4
huggingface_hub: 1.2.3
peft: 0.14.0
dqntrain: 1.0.0
```

## `meta.json` in every checkpoint

The same environment information is written into each checkpoint folder,
together with the stage name, step, config hash, and base model name:

```json
{
  "stage": "sft-code",
  "step": 400,
  "config_hash": "a91c3f0d2b1e",
  "base": "unsloth/Qwen3-4B",
  "python": "3.10.12",
  "torch": "2.11.0",
  "unsloth": "2025.1.3",
  "trl": "1.2.0",
  "transformers": "4.49.0",
  "dqntrain": "1.0.0"
}
```

This makes every checkpoint self-describing: whoever picks up the run later
can see which environment produced it.

## Reproducibility

The config hash deliberately excludes environment versions: the same pipeline
on a machine with a different torch still hashes identically. `meta.json` is
the place to look when training results differ between machines.
