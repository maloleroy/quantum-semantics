"""Small metadata, CSV histories and portable NPZ weights."""

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

Array = npt.NDArray[np.generic]


def fingerprint(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def new_directory(root: Path, label: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{label}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    path.mkdir()
    return path.resolve()


def write_json(path: Path, metadata: Mapping[str, object]) -> None:
    text = json.dumps(dict(metadata), indent=2, ensure_ascii=False) + "\n"
    if len(text.encode()) > 65_536:
        raise ValueError("Metadata exceeds 64 KiB; put arrays in NPZ and histories in CSV")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a metadata object")
    return cast(dict[str, object], value)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_arrays(path: Path, arrays: Mapping[str, Array]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as output:
        for name, array in arrays.items():
            with output.open(name + ".npy", "w", force_zip64=True) as member:
                np.save(member, array, allow_pickle=False)
    temporary.replace(path)


def save_weights(path: Path, model: nn.Module) -> None:
    arrays = {
        name: cast(Array, value.detach().cpu().numpy())
        for name, value in model.state_dict().items()
    }
    save_arrays(path, arrays)


def load_weights(path: Path, model: nn.Module) -> None:
    with np.load(path, allow_pickle=False) as arrays:
        state = {name: torch.tensor(cast(Array, arrays[name])) for name in arrays.files}
    model.load_state_dict(state, strict=True)
