"""Check Hamiltonian evolution, paired controls and a learnable ordered task."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.linalg import expm
from torch.nn import functional as F

from experiments.hamiltonian_study.contextual import (
    ContextResidual,
    ResidualConfig,
    directional_generators,
    hamiltonian_features,
    load_data,
    score,
    toy_data,
)
from experiments.hamiltonian_study.tune_contextual import select_recipes


def test_full_born_features_against_independent_exponential_and_derivative():
    generators = directional_generators(8).to(torch.complex128)
    torch.testing.assert_close(generators, generators.mH, atol=0, rtol=0)
    coefficients = torch.linspace(-0.4, 0.7, 16, dtype=torch.float64).reshape(2, 8)
    coefficients.requires_grad_()
    actual = hamiltonian_features(coefficients, generators)
    initial = np.ones(8) / np.sqrt(8)
    expected = []
    for row in coefficients.detach().numpy():
        h = np.einsum("k,kij->ij", row, generators.numpy())
        expected.append(abs(expm(-1j * h) @ initial) ** 2)
    np.testing.assert_allclose(actual.detach(), expected, atol=1e-12)
    torch.testing.assert_close(actual.sum(-1), torch.ones(2, dtype=torch.float64))
    loss = -actual[:, 3].log().mean()
    derivative = torch.autograd.grad(loss, coefficients)[0]
    shift = torch.zeros_like(coefficients)
    shift[0, 2] = 1e-6
    plus = -hamiltonian_features(coefficients.detach() + shift, generators)[:, 3].log().mean()
    minus = -hamiltonian_features(coefficients.detach() - shift, generators)[:, 3].log().mean()
    torch.testing.assert_close(derivative[0, 2], (plus - minus) / 2e-6, atol=1e-8, rtol=1e-6)


def test_controls_start_at_exact_bigram_with_paired_parameters():
    data = toy_data()
    config = ResidualConfig(25, 4, unknown_id=None)
    parameter_sets = []
    for kind in ("hamiltonian", "classical", "last_word"):
        torch.manual_seed(11)
        model = ContextResidual(config, data.prior, kind)
        parameter_sets.append(dict(model.named_parameters()))
        context = data.validation_contexts[:8]
        torch.testing.assert_close(model(context).softmax(-1), data.prior[context[:, -1]])
        assert torch.count_nonzero(model.lookup.weight[25]) == 0
    for name in parameter_sets[0]:
        for other in parameter_sets[1:]:
            torch.testing.assert_close(parameter_sets[0][name], other[name], atol=0, rtol=0)


def test_gradients_reach_context_and_hamiltonian_coefficients():
    data = toy_data()
    torch.manual_seed(11)
    model = ContextResidual(ResidualConfig(25, 4), data.prior, "hamiltonian")
    with torch.no_grad():
        model.head.weight.normal_(0, 0.2)
    loss = F.cross_entropy(model(data.train_contexts[:16]), data.train_targets[:16])
    loss.backward()
    for name, parameter in model.named_parameters():
        if name not in {"lookup.weight", "encoder.0.weight", "encoder.2.weight"}:
            continue
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 1e-6


def test_training_learns_earlier_context_on_unseen_combinations():
    torch.set_num_threads(1)
    data = toy_data()
    torch.manual_seed(11)
    config = ResidualConfig(25, 4, unknown_id=None)
    model = ContextResidual(config, data.prior, "hamiltonian")
    initial = score(model, data.validation_contexts, data.validation_targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(80):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(data.train_contexts), data.train_targets)
        loss.backward()
        optimizer.step()
    final = score(model, data.validation_contexts, data.validation_targets)
    assert final["nll"] < initial["nll"] - 0.2
    assert final["top1"] > 0.7
    swapped = data.validation_contexts.clone()
    swapped[:, :2] = swapped[:, [1, 0]]
    assert not torch.allclose(model(swapped), model(data.validation_contexts))
    last_only = ContextResidual(config, data.prior, "last_word")
    last_only.load_state_dict(model.state_dict())
    torch.testing.assert_close(last_only(swapped), last_only(data.validation_contexts))


def test_vocabulary_relabeling_preserves_named_predictions():
    data = toy_data()
    config = ResidualConfig(25, 4, unknown_id=None)
    torch.manual_seed(19)
    model = ContextResidual(config, data.prior, "hamiltonian")
    changed = ContextResidual(config, data.prior, "hamiltonian")
    with torch.no_grad():
        model.head.weight.normal_(0, 0.3)
        model.head.bias.normal_(0, 0.3)
        changed.load_state_dict(model.state_dict())
        permutation = torch.randperm(25)
        inverse = permutation.argsort()
        changed.lookup.weight[:25].copy_(model.lookup.weight[permutation])
        changed.head.weight.copy_(model.head.weight[permutation])
        changed.head.bias.copy_(model.head.bias[permutation])
        changed.prior.copy_(model.prior[permutation][:, permutation])
    contexts = data.validation_contexts[:12]
    torch.testing.assert_close(changed(inverse[contexts]), model(contexts)[:, permutation])


def test_loader_preserves_groups_and_does_not_require_test_arrays(tmp_path):
    np.savez(
        tmp_path / "dataset.npz",
        train_contexts=np.array([[-1, 1], [1, 2], [2, 3]]),
        train_targets=np.array([2, 3, 1]),
        train_groups=np.array([10, 10, 11]),
        validation_contexts=np.array([[-1, 2], [3, 2]]),
        validation_targets=np.array([1, 3]),
        validation_groups=np.array([12, 12]),
        unigram=np.full(4, 0.25),
        transitions=np.ones((4, 4)),
        vocabulary=np.array(["<unk>", "one", "two", "three"]),
    )
    (tmp_path / "dataset.json").write_text(json.dumps({"partitions": "fixture"}))
    data = load_data(tmp_path, 2, 0)
    assert data.train_contexts.shape == (2, 2)
    assert data.validation_contexts.shape == (2, 2)
    assert set(data.train_groups) <= {10, 11}
    assert set(data.validation_groups) == {12}


def test_toy_sentences_are_disjoint_and_invalid_latent_dimension_is_rejected():
    data = toy_data()
    train = {tuple(row) for row in data.train_contexts.tolist()}
    validation = {tuple(row) for row in data.validation_contexts.tolist()}
    assert train.isdisjoint(validation)
    assert torch.equal(data.train_targets, data.train_contexts[:, 0] + 4)
    with pytest.raises(ValueError, match="even"):
        directional_generators(7)


def test_recipe_selection_uses_only_nll_with_deterministic_ties():
    records = [
        {
            "job": {"kind": kind, "lr": lr, "weight_decay": decay},
            "directory": f"{kind}-{lr}-{decay}",
            "result": {"selected_validation": {"nll": nll, "top1": accuracy}},
        }
        for kind in ("hamiltonian", "classical")
        for lr, decay, nll, accuracy in [
            (0.003, 0.0, 1.0, 0.99),
            (0.001, 0.0, 1.0, 0.01),
            (0.01, 0.01, 0.9, 0.05),
        ]
    ]
    selected = select_recipes(records, 2)
    for rows in selected.values():
        assert [row["lr"] for row in rows] == [0.01, 0.001]
    with pytest.raises(ValueError, match="Not enough"):
        select_recipes(records, 4)


def test_cpu_search_locks_recipes_and_pairs_fresh_seed_fits(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.hamiltonian_study.tune_contextual",
            "--toy",
            "--output",
            str(tmp_path),
            "--epochs",
            "2",
            "--batch-size",
            "64",
            "--lrs",
            "0.003",
            "--decays",
            "0.02",
            "--keep",
            "1",
            "--confirmation-seeds",
            "17",
            "--workers",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    directories = list(tmp_path.glob("paired-search-*"))
    assert len(directories) == 1
    output = directories[0]
    result = json.loads((output / "summary.json").read_text())
    assert len(result["screen"]) == len(result["confirmation"]) == 2
    lock = hashlib.sha256((output / "selection_lock.json").read_bytes()).hexdigest()
    for row in result["confirmation"]:
        assert row["job"]["seed"] == 17
        assert row["selection_lock_sha256"] == lock
        assert row["result"]["optimizer"] == {"lr": 0.003, "weight_decay": 0.02}
        assert len(row["result"]["history"]) == 3
    for stage in ("screen", "confirmation"):
        left, right = result[stage]
        assert (
            left["result"]["initial_parameter_hashes"]
            == right["result"]["initial_parameter_hashes"]
        )
    assert json.loads((output / "run_info.json").read_text())["status"] == "complete"
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["test_scored"] is False
    for name, checksum in protocol["source_sha256"].items():
        assert hashlib.sha256((output / "source" / name).read_bytes()).hexdigest() == checksum
