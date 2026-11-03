"""Explicit execution devices; CPU remains the laptop default."""

import torch


def execution_device(name: str | torch.device = "cpu") -> torch.device:
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use cpu or cuda for model execution")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; install the cuda extra on the DGX")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def device_metadata(device: torch.device) -> dict[str, object]:
    return {
        "device": str(device),
        "accelerator": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "cuda_runtime": torch.version.cuda,
        "threads": torch.get_num_threads(),
    }
