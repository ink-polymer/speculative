"""Use the pinned authors' loader and Dataset.shuffle(seed=0) without reformatting."""
from __future__ import annotations

import contextlib
import copy
import re
from pathlib import Path

from .common import atomic_json, digest, file_hash, load_json, run_lock
from .official_spec import COMMIT, LIMITS, MODELS, PINNED_MODEL_REVISIONS, SOURCES, data_utils, verify_sources


def validate_lock(lock):
    expected = {v[0] for v in SOURCES.values()}
    models = {name for pair in MODELS for name in pair}
    if (lock.get("official_commit") != COMMIT or set(lock["datasets"]) != expected
            or set(lock["models"]) != models):
        raise ValueError("Official source lock identity mismatch")
    if any(not isinstance(s, str) or not re.fullmatch(r"[0-9a-f]{40}", s)
           for s in [*lock["datasets"].values(), *lock["models"].values()]):
        raise ValueError("Revisions must be immutable commit SHAs")
    if any(lock["models"][name] != sha for name, sha in PINNED_MODEL_REVISIONS.items()):
        raise ValueError("Pinned 4B/8B model revisions changed; use a new data/run directory")


@contextlib.contextmanager
def frozen_loader(utils, lock):
    validate_lock(lock)
    original = utils.load_dataset
    def load(path, *args, **kwargs):
        if path == "json":
            kwargs = copy.deepcopy(kwargs)
            files = kwargs["data_files"]["test"]
            prefix = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
            if (len(files) != 6 or any(not url.startswith(prefix) for url in files)):
                raise ValueError("Unexpected upstream raw JSON file layout")
            sha = lock["datasets"]["livecodebench/code_generation_lite"]
            kwargs["data_files"]["test"] = [url.replace("/resolve/main/", f"/resolve/{sha}/") for url in files]
        else:
            if path not in lock["datasets"]:
                raise ValueError(f"Unrecorded dataset source: {path}")
            kwargs["revision"] = lock["datasets"][path]
        return original(path, *args, **kwargs)
    utils.load_dataset = load
    try:
        yield
    finally:
        utils.load_dataset = original


def select_official(dataset, limit):
    # Exactly benchmark.py: don't shuffle datasets whose full size is <= limit.
    return dataset.shuffle(seed=0).select(range(limit)) if len(dataset) > limit else dataset


def prepare(directory: Path):
    from huggingface_hub import HfApi
    with run_lock(directory):
        if (directory / "manifest.json").exists():
            return check_manifest(directory)
        lock_path = directory / "source_revisions.json"
        if lock_path.exists():
            lock = load_json(lock_path)
        else:
            api = HfApi()
            lock = {"official_commit": COMMIT,
                    "datasets": {repo: api.dataset_info(repo).sha for repo in sorted({s[0] for s in SOURCES.values()})},
                    "models": {name: (PINNED_MODEL_REVISIONS[name] if name in PINNED_MODEL_REVISIONS
                                      else api.model_info(name).sha)
                               for name in sorted({v for pair in MODELS for v in pair})}}
            validate_lock(lock)
            atomic_json(lock_path, lock)
        validate_lock(lock)
        files = {}
        utils = data_utils()
        with frozen_loader(utils, lock):
            for name, limit in LIMITS.items():
                full = utils.load_and_process_dataset(name)
                selected = select_official(full, limit)
                if len(selected) != limit:
                    raise ValueError(f"Official sample count changed: {name} expected {limit}, got {len(selected)}")
                rows = []
                for i, row in enumerate(selected):
                    turns = row["turns"]
                    expected = 2 if name == "mt-bench" else 1
                    if len(turns) != expected or any(not isinstance(t, str) or not t.strip() for t in turns):
                        raise ValueError(f"Invalid official turns: {name}/{i}")
                    source_index = int(selected._indices.column(0)[i].as_py()) if selected._indices is not None else i
                    rows.append({"index": i, "source_index": source_index, "turns": list(turns),
                                 "turns_sha256": digest(list(turns))})
                path = directory / f"{name}.json"
                atomic_json(path, rows)
                files[path.name] = {"rows": len(rows), "full_source_rows": len(full), "sha256": file_hash(path)}
        manifest = {"version": 4, "official_commit": COMMIT, "sampling_seed": 0,
                    "sample_limits": LIMITS, "training": False, "full_split": False,
                    "source_lock_sha256": file_hash(lock_path), "files": files,
                    "official_source_manifest": verify_sources()}
        atomic_json(directory / "manifest.json", manifest)
        return check_manifest(directory)


def check_manifest(directory: Path):
    manifest = load_json(directory / "manifest.json")
    lock = load_json(directory / "source_revisions.json")
    validate_lock(lock)
    if (manifest.get("version") != 4 or manifest.get("official_commit") != COMMIT
            or manifest["sample_limits"] != LIMITS or manifest["sampling_seed"] != 0
            or manifest["training"] is not False or manifest["full_split"] is not False
            or manifest["source_lock_sha256"] != file_hash(directory / "source_revisions.json")
            or manifest["official_source_manifest"] != verify_sources()
            or set(manifest["files"]) != {f"{name}.json" for name in LIMITS}):
        raise ValueError("Not the frozen official sampled-data manifest")
    for name, limit in LIMITS.items():
        path = directory / f"{name}.json"
        rows = load_json(path)
        info = manifest["files"][path.name]
        if file_hash(path) != info["sha256"] or len(rows) != limit or info["rows"] != limit:
            raise ValueError(f"Dataset file changed: {name}")
        if len({r["source_index"] for r in rows}) != len(rows):
            raise ValueError("Duplicate official sample index")
        for i, row in enumerate(rows):
            if (row["index"] != i or row["turns_sha256"] != digest(row["turns"])
                    or not 0 <= row["source_index"] < info["full_source_rows"]
                    or len(row["turns"]) != (2 if name == "mt-bench" else 1)):
                raise ValueError("Official sample identity/turns mismatch")
    return manifest
