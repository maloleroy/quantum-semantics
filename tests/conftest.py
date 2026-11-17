"""Small generated corpora keep physics tests independent of research datasets."""

from pathlib import Path

import pytest


@pytest.fixture
def sentence_source(tmp_path: Path) -> Path:
    source = tmp_path / "sentences.txt"
    sentences = (
        f"the {color} {animal} sees the {place} near home"
        for color in ("red", "blue", "green", "white")
        for animal in ("bird", "cat", "fish", "dog")
        for place in ("tree", "river", "house", "garden")
    )
    source.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    return source
