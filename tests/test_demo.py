import json
import threading
from http.server import HTTPServer
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import pytest
import torch

from src.demo import CachedBases, complete, handler, load_entry
from src.generation import generate
from src.models import ModelConfig
from src.storage import save_arrays
from src.subwords import prepare_subwords
from src.training import TrainConfig, train


def adaptive_run(tmp_path: Path) -> Path:
    source = tmp_path / "corpus"
    source.mkdir()
    text = "A cat sees the river. The bird sees a cat near the house.\n"
    (source / "wiki.train.raw").write_text(f"\n = Train = \n{text * 4}", encoding="utf-8")
    (source / "wiki.valid.raw").write_text(" = Valid = \nThe cat sees a bird.\n", encoding="utf-8")
    (source / "wiki.test.raw").write_text(" = Test = \nA bird near home.\n", encoding="utf-8")
    prepare_subwords(source, tmp_path, window=8, validation_limit=16)
    config = ModelConfig("adaptive", 512, 8, ancilla_size=2, rank=4, recent_window=4)
    run = train(tmp_path / "dataset.npz", [config], TrainConfig(epochs=1), [11], tmp_path)
    return run


def test_cached_bases_and_completion_match_the_locked_model(tmp_path: Path) -> None:
    run = adaptive_run(tmp_path)
    entry = load_entry("Hamiltonian", run, "cpu")
    assert isinstance(entry.model, CachedBases)
    contexts = torch.tensor(entry.dataset.validation.contexts)
    with torch.no_grad():
        torch.testing.assert_close(entry.model(contexts), entry.model.inner(contexts))
    assert entry.info["name"] == "Hamiltonian"
    assert entry.info["summary"] == "10 qubits · 512 tokens · 53,276 parameters"
    for argmax in (True, False):
        result = complete(entry, "The cat ", 4, argmax)
        expected = generate(entry.dataset, entry.model.inner, "The cat ", 4, argmax=argmax)
        assert "The cat " + cast(str, result["text"]) == expected.text
        top = cast(list[tuple[str, float]], result["next"])
        assert len(top) == 5 and sorted(top, key=lambda pair: -pair[1]) == top
    assert complete(entry, "", 4, True) == {"text": "", "next": [], "ms": 0}


def test_changed_validation_scores_refuse_to_load(tmp_path: Path) -> None:
    run = adaptive_run(tmp_path)
    saved = next(run.glob("*/validation.npz"))
    with np.load(saved) as arrays:
        changed = {name: arrays[name] for name in arrays.files}
    changed["success"] = changed["success"] * 1.01
    save_arrays(saved, changed)
    with pytest.raises(ValueError, match="reproduce"):
        load_entry("", run, "cpu")


def test_server_routes(tmp_path: Path) -> None:
    entry = load_entry("", adaptive_run(tmp_path), "cpu")
    server = HTTPServer(("127.0.0.1", 0), handler([entry], 3))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    root = f"http://127.0.0.1:{server.server_port}"
    with urlopen(root + "/") as page:
        assert b"<textarea" in page.read()
    with urlopen(root + "/models") as models:
        assert json.loads(models.read())[0]["name"] == "adaptive-s11"
    with urlopen(f"{root}/complete?model=0&mode=sample&text={quote('A cat ')}") as reply:
        assert set(json.loads(reply.read())) == {"text", "next", "ms"}
    with pytest.raises(HTTPError, match="404"):
        urlopen(root + "/complete?model=1&text=A")
    server.shutdown()
    server.server_close()
    thread.join()
