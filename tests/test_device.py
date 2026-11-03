"""CPU defaults and explicit CUDA failure, independent of laptop hardware."""

import sys

import pytest
import torch

from src.cli import parse_arguments
from src.device import execution_device


def test_device_defaults_and_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert execution_device() == torch.device("cpu")
    with pytest.raises(ValueError, match="CUDA requested but unavailable"):
        execution_device("cuda")
    with pytest.raises(ValueError, match="Use cpu or cuda"):
        execution_device("meta")


def test_cli_device_environment_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    arguments = ["qcse", "train", "--dataset", "dataset.npz"]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.delenv("QCSE_DEVICE", raising=False)
    assert parse_arguments().device == "cpu"
    monkeypatch.setenv("QCSE_DEVICE", "cuda")
    assert parse_arguments().device == "cuda"
    monkeypatch.setattr(sys, "argv", [*arguments, "--device", "cpu"])
    assert parse_arguments().device == "cpu"
