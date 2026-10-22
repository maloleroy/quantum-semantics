"""Unique run directories and replace-on-success artifact writes."""

import json
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4

import numpy as np


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def save_npz(path, **payload):
    with atomic_path(path) as temporary, temporary.open("wb") as destination:
        np.savez_compressed(destination, **payload)


@contextmanager
def new_run_directory(root, label, runtime):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    output = Path(mkdtemp(prefix=f"{label}-{started:%Y%m%dT%H%M%SZ}-", dir=root)).resolve()
    with run_status(output, runtime):
        yield output


@contextmanager
def run_status(output, runtime):
    output = Path(output).resolve()
    path = output / "run_info.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    now = datetime.now(UTC).isoformat()
    info = {
        **previous,
        "started_at": previous.get("started_at", now),
        "last_started_at": now,
        "status": "running",
        **runtime,
    }
    info.pop("error", None)
    info.pop("finished_at", None)
    write_json(output / "run_info.json", info)
    print(f"Output directory: {output}", flush=True)
    try:
        yield output
    except BaseException as error:
        info.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
        )
        raise
    else:
        info["status"] = "complete"
    finally:
        info["finished_at"] = datetime.now(UTC).isoformat()
        write_json(output / "run_info.json", info)
