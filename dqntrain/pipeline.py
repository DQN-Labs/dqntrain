"""Sequential pipeline: add, run, status, summary, plan, rerun, reset."""
from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .config import Load
from .data import load_stage_data
from .env import env_info
from .errors import ConfigChangedError, DuplicateStageError, LockHeldError
from .hub import HFStore
from .stages import run_stage
from .state import (
    config_diff,
    empty_manifest,
    find_stage,
    lock_is_fresh,
    sha12,
    stage_config,
    utcnow,
)


class _StageContext:
    def __init__(self, pipe, stage, record, run_dir, resume_dir, prev_final_dir,
                 model, tokenizer, adapter_loaded):
        self.pipe = pipe
        self.stage = stage
        self.record = record
        self.run_dir = run_dir
        self.resume_dir = resume_dir
        self.prev_final_dir = prev_final_dir
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_loaded = adapter_loaded
        self.base = pipe.base
        self.load_values = pipe.load.values
        self.config_hash = record["config_hash"]

    def on_checkpoint(self, step, checkpoint_dir, metrics):
        return self.pipe._checkpoint(self.stage, self.record, step, checkpoint_dir, metrics)

    def finish(self, final_dir):
        return self.pipe._finish_stage(self.stage, self.record, final_dir)

    def log(self, record):
        return self.pipe._append_log(self.stage.name, record)


class Sequential:
    """A linear list of stages that resumes from the store automatically."""

    def __init__(self, name, repo=None, base=None, load=None, stages=None, store=None,
                 full_state=True, token=None, cache_dir=None):
        if not name:
            raise ValueError("run name is required")
        if store is None:
            if not repo:
                raise ValueError("pass repo=<hf repo id> or store=<Store>")
            store = HFStore(repo, token=token)
        self.name = name
        self.base = base
        self.load = load if load is not None else Load()
        self.store = store
        self.full_state = full_state
        self.holder = f"notebook-{uuid4().hex[:4]}"
        self.cache_dir = Path(cache_dir or os.environ.get("DQNTRAIN_CACHE", "/tmp/dqntrain-cache")) / name
        self._manifest = None
        self.stages = []
        if stages:
            self.add(*stages)

    def add(self, *stages):
        names = {stage.name for stage in self.stages}
        for stage in stages:
            if stage.name in names:
                raise DuplicateStageError(stage.name)
            names.add(stage.name)
            self.stages.append(stage)
        return self

    def _read_manifest(self):
        return self.store.read_manifest(self.name)

    def _write_manifest(self):
        self.store.write_manifest(self.name, self._manifest)

    def _hashes(self):
        previous = sha12({"base": self.base})
        entries = []
        for stage in self.stages:
            config, warnings = stage_config(stage, self.base, self.load.values, previous)
            digest = sha12(config)
            previous = digest
            entries.append({"name": stage.name, "hash": digest, "config": config, "warnings": warnings})
        return entries

    @contextmanager
    def _lock(self):
        manifest = self._read_manifest() or empty_manifest()
        self._manifest = manifest
        lock = manifest.get("lock") or {}
        holder = lock.get("holder")
        if holder and holder != self.holder:
            if lock_is_fresh(lock):
                raise LockHeldError(holder, lock.get("heartbeat"))
            self._append_log("_pipeline", {
                "event": "lock_takeover",
                "previous_holder": holder,
                "heartbeat": lock.get("heartbeat"),
                "by": self.holder,
            })
        manifest["lock"] = {"holder": self.holder, "heartbeat": utcnow()}
        self._write_manifest()
        try:
            yield manifest
        finally:
            manifest["lock"] = {}
            try:
                self._write_manifest()
            except Exception:
                pass

    def _touch_lock(self):
        self._manifest.setdefault("lock", {})["holder"] = self.holder
        self._manifest["lock"]["heartbeat"] = utcnow()
        self._write_manifest()

    def _reconcile(self, manifest, hashes):
        for item in hashes:
            for warning in item["warnings"]:
                print(f"WARNING: {item['name']}: {warning}")
            record = find_stage(manifest, item["name"])
            if record is None:
                manifest["stages"].append({
                    "name": item["name"],
                    "config_hash": item["hash"],
                    "status": "pending",
                    "checkpoints": [],
                    "config": item["config"],
                })
            elif record.get("config_hash") != item["hash"]:
                if record.get("status") in ("done", "running"):
                    raise ConfigChangedError(item["name"], config_diff(record.get("config"), item["config"]))
                record["config_hash"] = item["hash"]
                record["config"] = item["config"]

    def _fetch_resume(self, stage, record):
        checkpoints = record.get("checkpoints") or []
        if not checkpoints:
            return None
        rel = checkpoints[-1]["path"]
        dest = self.cache_dir / stage.name / Path(rel).name
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.store.download_dir(self.name, rel, dest)
        print(f"[{self.name}] resuming {stage.name!r} from {rel}")
        return str(dest)

    def _fetch_prev_final(self, start):
        if start == 0:
            return None
        previous = self.stages[start - 1]
        record = find_stage(self._manifest, previous.name)
        rel = f"{previous.name}/final"
        if record is None or record.get("status") != "done" or not self.store.has_dir(self.name, rel):
            return None
        dest = self.cache_dir / previous.name / "final"
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.store.download_dir(self.name, rel, dest)
        return str(dest)

    def _checkpoint(self, stage, record, step, checkpoint_dir, metrics):
        step = int(step)
        rel = f"{stage.name}/ckpt-{step:06d}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        meta = {
            "stage": stage.name,
            "step": step,
            "config_hash": record["config_hash"],
            "base": self.base,
            **env_info(),
        }
        with open(os.path.join(checkpoint_dir, "meta.json"), "w") as handle:
            json.dump(meta, handle, indent=2, default=str)
        line = {"step": step, "time": utcnow(), "metrics": metrics}
        if "loss" in metrics:
            line["loss"] = metrics["loss"]
        log_name = f"logs/{stage.name}.jsonl"
        existing = self.store.read_bytes(self.name, log_name) or b""
        message = f"{self.name}: {stage.name} checkpoint {step}"
        self.store.upload_dir(
            self.name, rel, checkpoint_dir, message,
            extra_files=[(log_name, existing + (json.dumps(line, default=str) + "\n").encode("utf-8"))],
        )
        record.setdefault("checkpoints", []).append({"step": step, "path": rel})
        try:
            self._touch_lock()
        except Exception:
            record["checkpoints"].pop()
            raise

    def _finish_stage(self, stage, record, final_dir):
        if final_dir and os.path.isdir(final_dir):
            self.store.upload_dir(self.name, f"{stage.name}/final", final_dir, f"{self.name}: {stage.name} final")
        record["status"] = "done"
        self._touch_lock()
        print(f"[{self.name}] {stage.name!r} done")

    def _append_log(self, log_name, record):
        path = f"logs/{log_name}.jsonl"
        existing = self.store.read_bytes(self.name, path) or b""
        line = json.dumps({"time": utcnow(), **record}, default=str) + "\n"
        self.store.write_file(self.name, path, existing + line.encode("utf-8"), f"{self.name}: log {log_name}")
        return path

    def run(self):
        """Resume or start the pipeline. Safe to call again."""
        with self._lock():
            manifest = self._manifest
            hashes = self._hashes()
            self._reconcile(manifest, hashes)
            self._write_manifest()
            records = [find_stage(manifest, stage.name) for stage in self.stages]
            start = next((i for i, record in enumerate(records) if record["status"] != "done"), len(records))
            if start == len(records):
                print(f"{self.name}: all stages done")
                return
            prev_final = self._fetch_prev_final(start)
            model = tokenizer = None
            adapter_loaded = False
            for stage, record in zip(self.stages[start:], records[start:]):
                record["status"] = "running"
                self._touch_lock()
                print(f"[{self.name}] running {stage.name!r} ({stage.kind})")
                resume_dir = self._fetch_resume(stage, record)
                run_dir = self.cache_dir / stage.name
                run_dir.mkdir(parents=True, exist_ok=True)
                ctx = _StageContext(self, stage, record, str(run_dir), resume_dir, prev_final,
                                    model, tokenizer, adapter_loaded)
                run_stage(stage, ctx)
                model, tokenizer, adapter_loaded = ctx.model, ctx.tokenizer, ctx.adapter_loaded
                prev_final = str(run_dir / "final") if (run_dir / "final").is_dir() else None

    def _action(self, item, record):
        if record is None:
            return "run from scratch"
        if record.get("config_hash") != item["hash"]:
            return "blocked: config changed, call rerun()"
        if record.get("status") == "done":
            return "skip (done)"
        checkpoints = record.get("checkpoints") or []
        if checkpoints:
            return f"resume from {checkpoints[-1]['path']}"
        return "run from scratch"

    def summary(self):
        """Keras-style overview. Loads datasets, never a model."""
        manifest = self._read_manifest()
        hashes = self._hashes()
        print(f"run: {self.name}")
        print(f"store: {self.store.describe()}")
        print(f"base: {self.base}")
        for stage, item in zip(self.stages, hashes):
            record = find_stage(manifest, stage.name) if manifest else None
            status = record.get("status") if record else "new"
            print(f"\n- {stage.name} ({stage.kind}) status={status} action={self._action(item, record)}")
            for warning in item["warnings"]:
                print(f"  WARNING: {warning}")
            data = getattr(stage, "data", None)
            if data is not None:
                loaded = load_stage_data(data, stage.kind)
                line = f"  rows: {len(loaded['train'])}"
                if loaded["eval"] is not None:
                    line += f" eval rows: {len(loaded['eval'])}"
                print(line)
                print(f"  columns: {list(loaded['train'].column_names)}")

    def plan(self):
        """Alias of summary()."""
        self.summary()

    def status(self):
        """Print stage, status, latest checkpoint, and config hash."""
        manifest = self._read_manifest()
        if manifest is None:
            print(f"{self.name}: no manifest in {self.store.describe()}")
            return
        print(f"{'stage':<24}{'status':<10}{'latest checkpoint':<38}config hash")
        for record in manifest.get("stages", []):
            checkpoints = record.get("checkpoints") or []
            latest = checkpoints[-1]["path"] if checkpoints else "-"
            print(f"{record.get('name', '?'):<24}{record.get('status', '?'):<10}{latest:<38}{record.get('config_hash', '-')}")
        lock = manifest.get("lock") or {}
        if lock.get("holder"):
            print(f"lock: {lock['holder']} heartbeat {lock.get('heartbeat')}")

    def rerun(self, name):
        """Mark a stage and all later stages as needing to run again."""
        names = [stage.name for stage in self.stages]
        if name not in names:
            raise ValueError(f"unknown stage {name!r}")
        manifest = self._read_manifest()
        if manifest is None:
            print(f"{self.name}: no manifest; nothing to rerun")
            return
        for stage_name in names[names.index(name):]:
            record = find_stage(manifest, stage_name)
            if record is not None:
                record["status"] = "pending"
                record["checkpoints"] = []
        self.store.write_manifest(self.name, manifest)
        print(f"{name!r} and all later stages marked for rerun")

    def reset(self, confirm=False):
        """Delete this run's folder in the store. Requires confirm=True."""
        if not confirm:
            raise ValueError("reset deletes the run in the store; call reset(confirm=True)")
        self.store.delete_run(self.name)
        shutil.rmtree(self.cache_dir, ignore_errors=True)
