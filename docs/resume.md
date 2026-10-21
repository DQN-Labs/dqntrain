# Resume and state

The Hugging Face repo is the only memory. Local disk is a cache: everything
under `cache_dir` (default `/tmp/dqntrain-cache`) can be deleted at any time and
`run()` rebuilds it from the store.

## Repo layout

```
<repo>/
└── <run-name>/
    ├── dqntrain.json              # manifest: the only file that declares validity
    ├── <stage-name>/
    │   ├── ckpt-000200/           # full trainer state
    │   ├── ckpt-000400/
    │   └── final/                 # adapter or merged output, present only when done
    └── logs/
        └── <stage-name>.jsonl     # one line per checkpoint with loss and metrics
```

## Manifest

```json
{
  "version": 1,
  "stages": [
    {
      "name": "sft-code",
      "config_hash": "a91c3f0d2b1e",
      "status": "running",
      "checkpoints": [
        {"step": 200, "path": "sft-code/ckpt-000200"},
        {"step": 400, "path": "sft-code/ckpt-000400"}
      ]
    },
    {"name": "dpo-prefs", "config_hash": "7be20d1f9a33", "status": "pending", "checkpoints": []}
  ],
  "lock": {"holder": "notebook-7f3a", "heartbeat": "2026-10-19T10:42:00Z"}
}
```

Stage statuses are `pending`, `running`, and `done`. A checkpoint appears in
`checkpoints` only after all of its files have uploaded. Folders that exist in
the repo but are not listed in the manifest are ignored.

## Checkpoint contents

Exact resume needs optimizer and scheduler state, not only adapter weights.
Each `ckpt-*` folder contains what `trainer.save_state()` and
`trainer.save_model()` write: adapter weights, `optimizer.pt`, `scheduler.pt`,
`trainer_state.json`, and RNG state files. A `meta.json` is added with the
stage name, config hash, base model name, and environment versions.

`full_state=False` on `Sequential` uploads adapter weights only
(`save_only_model`). It is faster but resumes with a fresh optimizer.

## Commit protocol

At each save step the runner:

1. saves the checkpoint locally,
2. uploads the whole folder **plus** the new log line in one commit,
3. commits the manifest entry that makes the checkpoint valid.

Files first, manifest second. A crash in between leaves an orphan folder that
the next run ignores, so a checkpoint is never half-valid.

## Config hashing

Each stage has a SHA-256 hash (first 12 hex characters) over a canonical JSON
object:

- stage class name and stage name;
- the data config with defaults filled in, including `revision`, `seed`,
  `limit`, `columns`, and the source hashes of `filter` and `map`;
- resolved `lora`, `train`, and `load` values;
- `save_every`, `format`, and for `Merge` the `push_to` and `dtype` values;
- for GRPO, the reward source hashes and `reward_weights`;
- the config hash of the previous stage.

The chain means a change upstream changes everything downstream. On `run()`, a
hash mismatch on a `done` or `running` stage raises `ConfigChangedError` with
the changed keys, old and new values:

```
config for stage 'sft-code' changed; call rerun('sft-code') to accept it.
changed keys: train.learning_rate: 0.0002 -> 1e-05
```

`pipe.rerun("sft-code")` marks that stage and all later ones as `pending` and
clears their checkpoints, so the next `run()` retrains them from scratch.

Reward functions are hashed from source (`inspect.getsource`). Functions
without source, such as some notebook lambdas, fall back to bytecode hashing,
and `summary()`/`plan()` print a warning naming the function.

## Locking

Two notebooks cannot drive the same run at once. `run()` writes a lock with a
heartbeat and refreshes it at each stage start and each checkpoint.

- A live lock (heartbeat younger than 15 minutes) raises `LockHeldError`
  naming the holder.
- A stale lock is taken over; the takeover is appended to
  `logs/_pipeline.jsonl`.
- The lock is cleared in a `finally` block on success, failure, or crash.

## Control methods

```python
pipe.run()                   # resume or start; safe to call again
pipe.summary()               # stages, dataset rows, columns, planned actions
pipe.plan()                  # alias of summary()
pipe.status()                # stage, status, latest checkpoint, config hash
pipe.rerun("dpo-prefs")      # retrain this stage and all later stages
pipe.reset(confirm=True)     # delete this run's folder in the store
```

`reset()` requires `confirm=True` since it deletes the run in the store.
Checkpoints are kept indefinitely by default; pruning is out of scope for v1.
