"""Mechanistic checks and audit of the existing saved run; no fitting on test data."""

import argparse
import json
from pathlib import Path

import numpy as np

from qcse.data import DEFAULT_DATA, Example
from qcse.training import load_run

from .encoders import METHODS, Features
from .run import prepare, vocabulary_metrics, write_json
from .simulator import Simulator


def main():
    args = argparse.Namespace(data=DEFAULT_DATA, train_limit=0, val_limit=0)
    vocabulary, examples, train, val, test = prepare(args)
    synthetic = [Example(-1, 1, 0, (i,)) for i in range(64)]
    synthetic += [Example(-1, 4, 0, (1, 4, 8, i)) for i in range(64)]
    synthetic += [Example(-1, 4, 0, (8, 4, 1, i)) for i in range(64)]
    features = Features(examples + synthetic, train, len(vocabulary), 10)
    sim = Simulator(10)
    result = {}
    for method in METHODS:
        spec = features.build(method)
        state = spec["state"]
        if state is None:
            state = sim.encode(spec["angles"][-192:], spec["kind"], spec["entangle"])
        else:
            state = state[-192:]
        single, prefix, reverse = state[:64], state[64:128], state[128:]
        fid = np.abs(single @ single.conj().T) ** 2
        unique = []
        for i in range(64):
            if not any(fid[i, j] > 1 - 1e-12 for j in unique):
                unique.append(i)
        row = {
            "distinct_states_first64_singletons": len(unique),
            "singleton_adjacent_mean_fidelity": float(np.diag(fid, 1).mean()),
            "last_token_adjacent_mean_fidelity": float(
                np.mean(np.abs(np.sum(prefix[:-1].conj() * prefix[1:], axis=1)) ** 2)
            ),
            "history_reversal_mean_fidelity": float(
                np.mean(np.abs(np.sum(prefix.conj() * reverse, axis=1)) ** 2)
            ),
            "state_kind": spec["kind"],
            "entangle": spec["entangle"],
        }
        if spec["angles"] is not None:
            layers = spec["angles"].shape[1]
            row["encoding_rotation_gates"] = int(20 * layers * (2 if spec["reupload"] else 1))
            row["encoding_cnot_gates"] = int(
                9 * layers * spec["entangle"] * (2 if spec["reupload"] else 1)
            )
        result[method] = row
    # A shared downstream unitary preserves all pairwise fidelities exactly.
    spec = features.build("exponential_ry")
    state = sim.encode(spec["angles"][-192:-128], "ry")
    after = sim.ansatz(state, np.random.default_rng(88).normal(size=58))
    fidelity_error = float(
        np.max(np.abs(np.abs(state @ state.conj().T) ** 2 - np.abs(after @ after.conj().T) ** 2))
    )
    old_path = DEFAULT_DATA.parent / "outputs/train/run.npz"
    old = load_run(old_path)
    old_examples = old["examples"]
    old_features = Features(old_examples, old["train_ids"], len(vocabulary), 10)
    ids = old["test_ids"]
    targets = np.array([old_examples[i].target for i in ids])
    audit = {
        "path": str(old_path),
        "model": {k: v for k, v in old["metadata"]["model"].items() if k != "vocabulary"},
        "last_history": old["state"]["history"][-1],
        "saved_model_metrics": vocabulary_metrics(
            old["state"]["embeddings"][ids], targets, old_features.target_bits
        ),
        "same_split_bit_prior": vocabulary_metrics(
            np.broadcast_to(old_features.bit_prior, (len(ids), 10)),
            targets,
            old_features.target_bits,
        ),
        "same_split_bigram_bits": vocabulary_metrics(
            old_features.bigram_bits[[old_examples[i].context[-1] for i in ids]],
            targets,
            old_features.target_bits,
        ),
    }
    out = Path("experiments/encoding_study/results")
    write_json(
        out / "diagnostics.json",
        {
            "encoders": result,
            "shared_unitary_fidelity_max_error": fidelity_error,
            "full_split_counts": {"train": len(train), "validation": len(val), "test": len(test)},
            "saved_run_audit": audit,
        },
    )
    print(
        json.dumps(
            {"shared_unitary_fidelity_max_error": fidelity_error, "saved_run_audit": audit},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
