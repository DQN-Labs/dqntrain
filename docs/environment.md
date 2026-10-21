# Environment

## `dq.env_report()`

Prints detected versions of Python, torch, CUDA or ROCm (with the HIP version
when present), MPS when available, unsloth (optional; `None` when absent),
bitsandbytes (optional; `None` when unavailable), trl, transformers, datasets,
huggingface_hub, peft, and dqntrain. Missing required packages print a warning;
unsloth and bitsandbytes print `None`. The call never fails.

```python
import dqntrain as dq

dq.env_report()
```

```
python: 3.10.12
torch: 2.11.0
cuda: None
rocm: 7.0.51831
unsloth: None
bitsandbytes: None
trl: 1.13.0
transformers: 5.17.0
datasets: 4.8.5
huggingface_hub: 1.33.0
peft: 0.21.2
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
  "base": "Qwen/Qwen3-4B",
  "python": "3.10.12",
  "torch": "2.11.0",
  "rocm": "7.0.51831",
  "unsloth": null,
  "bitsandbytes": null,
  "trl": "1.13.0",
  "transformers": "5.17.0",
  "datasets": "4.8.5",
  "huggingface_hub": "1.33.0",
  "peft": "0.21.2",
  "dqntrain": "1.0.0"
}
```

This makes every checkpoint self-describing: whoever picks up the run later
can see which environment produced it.

## Reproducibility

The config hash deliberately excludes environment versions: the same pipeline
on a machine with a different torch still hashes identically. `meta.json` is
the place to look when training results differ between machines.
