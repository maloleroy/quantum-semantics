"""Exploratory SimLex-999 check; never used for training or model selection."""

import csv
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from scipy.stats import spearmanr

from .data import transition_probabilities, write_json
from .models import ClassicalEmbedding, QCSEModel
from .run import make_model, prediction


def run():
    torch.set_num_threads(1)
    root = Path(__file__).parent / "results"
    data = dict(np.load(root / "tatoeba_data" / "dataset.npz"))
    vocab = {word: i for i, word in enumerate(data["vocabulary"])}
    with (root / "external" / "SimLex-999.txt").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    pairs = [r for r in rows if r["word1"] in vocab and r["word2"] in vocab]
    left = np.array([vocab[r["word1"]] for r in pairs])
    right = np.array([vocab[r["word2"]] for r in pairs])
    gold = np.array([float(r["SimLex999"]) for r in pairs])
    contexts = torch.full((len(vocab), 8), -1, dtype=torch.long)
    contexts[:, -1] = torch.arange(len(vocab))
    result = {}

    def add(name, scores):
        constant = np.ptp(scores) < 1e-6
        result[name] = {
            "spearman": None if constant else float(cast(float, spearmanr(scores, gold)[0])),
            "score_range": float(np.ptp(scores)),
            "constant_to_tolerance_1e6": bool(constant),
        }

    def probability_similarity(p):
        return (np.sqrt(p[left] * p[right]).sum(1)) ** 2

    def cosine(x):
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-15)
        return (x[left] * x[right]).sum(1)

    # Static basis-state fidelity is identically zero for distinct word pairs.
    add("static_walk_state_fidelity", (left == right).astype(float))
    transition = transition_probabilities(data["transitions"], data["unigram"])
    add("bigram_probability_overlap", probability_similarity(transition))
    w = data["cooc"] + data["cooc"].T
    denominator = w.sum(1)[:, None] * w.sum(0)[None, :]
    ppmi = np.where(
        w > 0,
        np.maximum(np.log(np.maximum(w * w.sum(), 1e-30) / np.maximum(denominator, 1e-30)), 0),
        0,
    )
    add("ppmi_cosine", cosine(ppmi))
    for stage, methods in [
        ("main", ["walk", "predictive_residual", "qcse_nll"]),
        ("full_classical", ["embedding"]),
    ]:
        for method in methods:
            for seed in [11, 22, 33]:
                key = f"{method}_count_L16_s{seed}"
                config = json.loads((root / stage / f"{key}.json").read_text())["config"]
                model = make_model(data, config)
                model.load_state_dict(
                    torch.load(root / stage / f"{key}.pt", weights_only=True), strict=False
                )
                inputs = model.encode(contexts) if isinstance(model, QCSEModel) else contexts
                p, _ = prediction(model, inputs)
                add(f"{method}_probability_overlap_s{seed}", probability_similarity(p))
                if method in ("walk", "predictive_residual"):
                    with torch.no_grad():
                        _, aux = model(inputs, return_aux=True)
                        states = aux["state"].numpy()
                    add(
                        f"{method}_state_fidelity_s{seed}",
                        abs((states[left].conj() * states[right]).sum(1)) ** 2,
                    )
                if method == "embedding":
                    assert isinstance(model, ClassicalEmbedding)
                    add(
                        f"embedding_table_cosine_s{seed}",
                        cosine(model.embedding.weight.detach().numpy()[: len(vocab)]),
                    )
    output = {
        "source": json.loads((root / "simlex_source.json").read_text()),
        "pairs_used": len(pairs),
        "gold_score_range": [float(gold.min()), float(gold.max())],
        "results": result,
        "limitations": (
            "Small common-word intersection; singleton word contexts, not contextual sense "
            "disambiguation. No hyperparameters selected on SimLex. Constant representations "
            "have undefined correlation, not zero correlation."
        ),
    }
    write_json(root / "semantic_probe.json", output)
    return output


if __name__ == "__main__":
    print(run())
