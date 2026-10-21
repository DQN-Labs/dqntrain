# Stores and testing

`hub.py` defines a small store interface with two implementations.

## `HFStore(repo, token=None, private=True)`

Used automatically when `Sequential` is given a string repo:

```python
pipe = dq.Sequential("run", repo="you/dqntrain-ckpts", base="Qwen/Qwen3-4B", stages=[sft])
```

- The repo is created private if it does not exist.
- The token comes from the `token` argument or the `HF_TOKEN` environment
  variable. It is never stored in code, the manifest, or logs.
- Uploads use `HfApi.create_commit` with `CommitOperationAdd`, so each
  checkpoint is a single commit.
- Downloads use `huggingface_hub` snapshot download into the local cache.

## `LocalStore(root_dir)`

Writes the same layout to a local folder:

```python
from dqntrain.hub import LocalStore

store = LocalStore("/tmp/dqntrain-store")
pipe = dq.Sequential("run", store=store, base="Qwen/Qwen3-4B", stages=[sft])
```

Every path, commit, and manifest operation has a local equivalent. No network
access is required.

## Passing a store

`Sequential` resolves its store like this:

1. `store=` if given;
2. otherwise `HFStore(repo, token)` from the string `repo=`;
3. otherwise a `ValueError` explaining that one of the two is required.

## Store interface

Both implementations expose:

| Method | Purpose |
|---|---|
| `read_manifest(run)` / `write_manifest(run, manifest)` | Manifest load and save |
| `upload_dir(run, rel, local_dir, message, extra_files=None)` | One atomic commit of a folder plus extra files |
| `download_dir(run, rel, local_dir)` | Fetch a folder into the cache |
| `read_bytes(run, rel)` / `write_file(run, rel, data, message)` | Raw file access, used for log appends |
| `has_dir(run, rel)` | Existence check for `final/` artifacts |
| `delete_run(run)` | Delete the run folder, used by `reset(confirm=True)` |
| `describe()` | Human-readable store description for logs and `summary()` |

The interface is deliberately tiny; a third backend only needs these methods.

## Testing

`tests/` uses `LocalStore` and mock trainers, and never touches the network.
Covered behaviors include kill-and-resume loss continuity, half-finished
uploads, two-stage restarts, config changes, lock takeover, alias hashing,
and a fully cleared local cache.

A real-TRL integration test runs a tiny model on CPU and is opt-in:

```bash
DQNTRAIN_TEST_NETWORK=1 pytest tests/test_integration_trl.py
```
