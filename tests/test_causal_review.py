"""Verify the audit's statistics and the gate-family interpretation."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def module(name):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        assert spec and spec.loader
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("qubits,layers", [(2, 1), (3, 2), (8, 4)])
def test_mapped_crz_zz_states_and_gradient_chain_rule(qubits, layers):
    record = module("probe_gate_coordinates").probe(qubits, layers)
    assert (
        max(
            record[k]
            for k in ("state_max_abs", "coordinate_gradient_max_abs", "input_gradient_max_abs")
        )
        < 1e-12
    )


def test_coordinate_map_is_invertible_but_not_an_identity_initialization():
    torch.manual_seed(47)
    weights = torch.randn(2, 8, dtype=torch.float64)  # q3: three RX, three RZ, two edges.
    converted = module("probe_gate_coordinates").crz_to_zz(weights, 3)
    recovered_edges = -2 * converted[:, 6:]
    incoming = torch.cat((torch.zeros_like(weights[:, :1]), recovered_edges / 2), dim=-1)
    recovered = torch.cat((converted[:, :3], converted[:, 3:6] - incoming, recovered_edges), dim=-1)
    torch.testing.assert_close(recovered, weights, atol=1e-14, rtol=0)
    assert not torch.equal(converted, weights)


@pytest.mark.parametrize("kind", ["crz-chain", "zz-chain", "xx-chain"])
def test_two_layer_product_input_chain_has_bond_at_most_four(kind):
    record = module("probe_gate_coordinates").schmidt_probe(kind, 2)
    assert max(record["numerical_ranks"]) <= 4
    assert record["norm_max_abs"] < 1e-12


def test_cluster_values_cannot_claim_more_hits_than_tokens(tmp_path):
    audit = module("audit_causal_controls")
    path = tmp_path / "clusters.npz"
    np.savez(
        path,
        groups=np.array([1]),
        counts=np.array([2]),
        labels=np.array([["arm", "47"]]),
        sums=np.array([[[1.0, 1.0, 3.0]]]),
    )
    with pytest.raises(AssertionError):
        audit.load_clusters(path, np.array([1]), np.array([2]))


def test_rank_counts_must_reproduce_full_word_metrics():
    audit = module("audit_causal_controls")
    ledger = dict(
        validation=True,
        selected_epoch=100,
        stopped_epoch=100,
        training_draws=2_000_000,
        optimizer_updates=31_300,
        train_examples=1_014_085,
        metrics=dict(nll=2.0, top1=0.5, top5=0.75, examples=252120),
    )
    record = dict(
        ledger=ledger,
        metrics=ledger["metrics"],
        rank_counts_1_to_5_then_miss=[126060, 63030, 0, 0, 0, 63030],
    )
    sums = np.array([[504240.0, 126060.0, 189090.0]])
    audit.check_record(record, ledger, sums, np.array([252120]))
    record["rank_counts_1_to_5_then_miss"] = [126059, 63031, 0, 0, 0, 63030]
    with pytest.raises(AssertionError):
        audit.check_record(record, ledger, sums, np.array([252120]))
