"""Embed isolated vocabulary words locally, following Qwen's last-token pooling."""

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import model_info
from transformers import AutoModel, AutoTokenizer

from qcse.data import load_phrases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/huggingface"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if (args.output / "vocabulary.npz").exists():
        raise FileExistsError("Embedding artifact already exists; choose another output")
    args.output.mkdir(parents=True, exist_ok=True)
    vocabulary = sorted({w for sentence in load_phrases() for w in sentence})
    model_id = "Qwen/Qwen3-Embedding-0.6B"
    revision = model_info(model_id, revision=args.revision).sha
    torch.set_num_threads(4)
    torch.manual_seed(0)
    start = time.perf_counter()
    common = {"revision": revision, "cache_dir": args.cache, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", **common)
    # CPU float32 is comfortably below 16GB for short words, avoids MPS availability assumptions.
    model = (
        AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float32, attn_implementation="sdpa", **common
        )
        .to(args.device)
        .eval()
    )
    loaded = time.perf_counter()
    vectors = []
    max_tokens = 0
    with torch.inference_mode():
        for start_id in range(0, len(vocabulary), args.batch_size):
            words = vocabulary[start_id : start_id + args.batch_size]
            tokens = tokenizer(words, padding=True, truncation=False, return_tensors="pt")
            max_tokens = max(max_tokens, tokens["input_ids"].shape[1])
            tokens = tokens.to(args.device)
            hidden = model(**tokens).last_hidden_state[:, -1]
            vectors.append(torch.nn.functional.normalize(hidden, dim=1).cpu().numpy())
            if start_id % (args.batch_size * 8) == 0:
                print(f"Embedded {start_id + len(words)}/{len(vocabulary)} words", flush=True)
    result = np.concatenate(vectors).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite embeddings")
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-5)
    np.savez_compressed(args.output / "vocabulary.npz", vocabulary=vocabulary, vectors=result)
    metadata = {
        "model": model_id,
        "revision": revision,
        "device": args.device,
        "dtype": "float32",
        "pooling": "last nonpadding token (left padding), L2 normalized",
        "prompt": None,
        "text_unit": "isolated word, not sentence-contextual embedding",
        "batch_size": args.batch_size,
        "shape": list(result.shape),
        "max_tokens": max_tokens,
        "download_and_load_seconds": loaded - start,
        "inference_seconds": time.perf_counter() - loaded,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": 4,
        "vocabulary_sha256": hashlib.sha256(json.dumps(vocabulary).encode()).hexdigest(),
        "vectors_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        "source": "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
