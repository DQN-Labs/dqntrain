"""Checkpoint storage: Hugging Face repos or local folders."""
from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
from pathlib import Path

from .state import MANIFEST_NAME


class Store:
    """Interface implemented by HFStore and LocalStore. Paths are relative to the run."""

    def describe(self):
        raise NotImplementedError

    def read_manifest(self, run):
        raise NotImplementedError

    def write_manifest(self, run, manifest):
        raise NotImplementedError

    def upload_dir(self, run, rel, local_dir, message, extra_files=None):
        raise NotImplementedError

    def download_dir(self, run, rel, local_dir):
        raise NotImplementedError

    def read_bytes(self, run, rel):
        raise NotImplementedError

    def write_file(self, run, rel, data, message):
        raise NotImplementedError

    def has_dir(self, run, rel):
        raise NotImplementedError

    def delete_run(self, run):
        raise NotImplementedError


class LocalStore(Store):
    """Writes the repository layout into a local folder. Used by tests."""

    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def describe(self):
        return f"local:{self.root}"

    def _run_root(self, run):
        return self.root / run

    def read_manifest(self, run):
        path = self._run_root(run) / MANIFEST_NAME
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def write_manifest(self, run, manifest):
        root = self._run_root(run)
        root.mkdir(parents=True, exist_ok=True)
        path = root / MANIFEST_NAME
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n")
        tmp.replace(path)

    def upload_dir(self, run, rel, local_dir, message, extra_files=None):
        dest = self._run_root(run) / rel
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(local_dir, dest, ignore=shutil.ignore_patterns(".*"))
        for name, data in extra_files or []:
            target = self._run_root(run) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    def download_dir(self, run, rel, local_dir):
        src = self._run_root(run) / rel
        if not src.exists():
            raise FileNotFoundError(f"{rel} not found in {self.describe()}")
        shutil.copytree(src, local_dir, dirs_exist_ok=True)
        return local_dir

    def read_bytes(self, run, rel):
        path = self._run_root(run) / rel
        return path.read_bytes() if path.exists() else None

    def write_file(self, run, rel, data, message):
        target = self._run_root(run) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def has_dir(self, run, rel):
        return (self._run_root(run) / rel).is_dir()

    def delete_run(self, run):
        shutil.rmtree(self._run_root(run), ignore_errors=True)


class HFStore(Store):
    """Hugging Face Hub storage. The repo is created private if missing."""

    def __init__(self, repo, token=None, private=True):
        from huggingface_hub import HfApi

        self.repo = repo
        self.token = token or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.token)
        self.api.create_repo(repo, private=private, exist_ok=True)

    def describe(self):
        return f"hf:{self.repo}"

    def _manifest_path(self, run):
        return f"{run}/{MANIFEST_NAME}"

    def read_manifest(self, run):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

        try:
            path = hf_hub_download(self.repo, self._manifest_path(run), token=self.token)
        except (EntryNotFoundError, RepositoryNotFoundError):
            return None
        return json.loads(Path(path).read_text())

    def write_manifest(self, run, manifest):
        from huggingface_hub import CommitOperationAdd

        data = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        self.api.create_commit(
            self.repo,
            [CommitOperationAdd(path_in_repo=self._manifest_path(run), path_or_fileobj=io.BytesIO(data))],
            commit_message=f"manifest: {run}",
        )

    def upload_dir(self, run, rel, local_dir, message, extra_files=None):
        from huggingface_hub import CommitOperationAdd

        prefix = f"{run}/{rel}/"
        ops = []
        for root, dirs, files in os.walk(local_dir):
            dirs[:] = sorted(name for name in dirs if not name.startswith("."))
            for name in sorted(files):
                if name.startswith("."):
                    continue
                full = os.path.join(root, name)
                inner = os.path.relpath(full, local_dir).replace(os.sep, "/")
                ops.append(CommitOperationAdd(path_in_repo=prefix + inner, path_or_fileobj=full))
        for name, data in extra_files or []:
            ops.append(
                CommitOperationAdd(path_in_repo=f"{run}/{name}", path_or_fileobj=io.BytesIO(data))
            )
        if not ops:
            return
        self.api.create_commit(self.repo, ops, commit_message=message)

    def download_dir(self, run, rel, local_dir):
        from huggingface_hub import snapshot_download

        stage_root = tempfile.mkdtemp(prefix="dqntrain-dl-")
        try:
            snapshot_download(
                self.repo,
                allow_patterns=[f"{run}/{rel}/**"],
                local_dir=stage_root,
                token=self.token,
            )
            source = Path(stage_root) / run / rel
            if not source.exists():
                raise FileNotFoundError(f"{rel} not found in {self.describe()}")
            shutil.copytree(source, local_dir, dirs_exist_ok=True)
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)
        return local_dir

    def read_bytes(self, run, rel):
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            path = hf_hub_download(self.repo, f"{run}/{rel}", token=self.token)
        except EntryNotFoundError:
            return None
        return Path(path).read_bytes()

    def write_file(self, run, rel, data, message):
        from huggingface_hub import CommitOperationAdd

        self.api.create_commit(
            self.repo,
            [CommitOperationAdd(path_in_repo=f"{run}/{rel}", path_or_fileobj=io.BytesIO(data))],
            commit_message=message,
        )

    def has_dir(self, run, rel):
        prefix = f"{run}/{rel}/"
        return any(path.startswith(prefix) for path in self.api.list_repo_files(self.repo))

    def delete_run(self, run):
        from huggingface_hub import CommitOperationDelete

        prefix = f"{run}/"
        files = [path for path in self.api.list_repo_files(self.repo) if path.startswith(prefix)]
        if not files:
            return
        ops = [CommitOperationDelete(path_in_repo=path) for path in files]
        self.api.create_commit(self.repo, ops, commit_message=f"reset: delete {run}")
