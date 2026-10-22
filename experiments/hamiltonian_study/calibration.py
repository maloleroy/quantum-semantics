"""Low-parameter classical controls for apparent quantum residual gains."""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import transition_probabilities, write_json
from .run import metrics

ROOT = Path(__file__).parent / "results"


class CalibratedBigram(nn.Module):
    def __init__(self, temperature=False):
        super().__init__()
        self.mixture = nn.Parameter(torch.tensor(-6.0))
        self.log_temperature = nn.Parameter(torch.tensor(0.0), requires_grad=temperature)

    def forward(self, base, unigram):
        p = (base.log() / self.log_temperature.exp()).softmax(1)
        eta = self.mixture.sigmoid()
        return (1 - eta) * p + eta * unigram


def fit(method, seed, evaluate=False):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    data = dict(np.load(ROOT / "tatoeba_data" / "dataset.npz"))
    transition = torch.tensor(
        transition_probabilities(data["transitions"], data["unigram"]), dtype=torch.float32
    )
    unigram = torch.tensor(data["unigram"], dtype=torch.float32)
    model = CalibratedBigram(method == "calibrated_bigram")
    output = ROOT / "calibration"
    output.mkdir(exist_ok=True)
    key = f"{method}_count_L0_s{seed}"
    if evaluate:
        import json

        lock = json.loads((output / "selection_lock.json").read_text())
        if key not in lock["jobs"]:
            raise ValueError("Unselected test evaluation")
        model.load_state_dict(torch.load(output / f"{key}.pt", weights_only=True))
        with torch.no_grad():
            p = model(transition[data["test_contexts"][:, -1]], unigram).numpy()
        score, arrays = metrics(p, data["test_targets"])
        arrays["groups"] = data["test_groups"]
        np.savez_compressed(output / f"{key}_test.npz", **arrays)
        return key, score
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    history = []
    best_loss = float("inf")
    best = None
    best_epoch = 0
    for epoch in range(31):
        if epoch:
            order = rng.permutation(len(data["train_targets"]))
            for offset in range(0, len(order), 128):
                ids = order[offset : offset + 128]
                optimizer.zero_grad()
                p = model(transition[data["train_contexts"][ids, -1]], unigram)
                loss = -p[np.arange(len(ids)), data["train_targets"][ids]].log().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        if epoch % 5 == 0:
            with torch.no_grad():
                p = model(transition[data["validation_contexts"][:, -1]], unigram).numpy()
            score, _ = metrics(p, data["validation_targets"])
            history.append({"epoch": epoch, "validation_nll": score["nll"]})
            if score["nll"] < best_loss:
                best_loss = score["nll"]
                best = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
    assert best is not None
    model.load_state_dict(best)
    torch.save(best, output / f"{key}.pt")
    with torch.no_grad():
        p = model(transition[data["validation_contexts"][:, -1]], unigram).numpy()
    score, arrays = metrics(p, data["validation_targets"])
    arrays["groups"] = data["validation_groups"]
    np.savez_compressed(output / f"{key}_validation.npz", **arrays)
    record = {
        "config": {
            "method": method,
            "seed": seed,
            "layers": 0,
            "epochs": 30,
            "batch_size": 128,
            "dataset": "tatoeba_data",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "selected_epoch": best_epoch,
        "validation": score,
        "history": history,
        "mixture": float(model.mixture.detach().sigmoid()),
        "temperature": float(model.log_temperature.detach().exp()),
        "seconds": time.perf_counter() - start,
    }
    write_json(output / f"{key}.json", record)
    return key, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    result = {}
    for method in ["mixed_bigram", "calibrated_bigram"]:
        for seed in [11, 22, 33]:
            key, score = fit(method, seed, args.evaluate)
            result[key] = score
    if args.evaluate:
        write_json(ROOT / "calibration" / "test_results.json", {"metrics": result})
    print({k: v["nll"] for k, v in result.items()})


if __name__ == "__main__":
    main()
