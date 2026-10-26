"""Retain the circuit mechanism checks independently of retired cohort audits."""

import pytest
import torch

from scripts import probe_gate_coordinates as probe


@pytest.mark.parametrize("qubits,layers", [(2, 1), (3, 2), (8, 4)])
def test_mapped_crz_zz_states_and_gradient_chain_rule(qubits, layers):
    record = probe.probe(qubits, layers)
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
    converted = probe.crz_to_zz(weights, 3)
    recovered_edges = -2 * converted[:, 6:]
    incoming = torch.cat((torch.zeros_like(weights[:, :1]), recovered_edges / 2), dim=-1)
    recovered = torch.cat((converted[:, :3], converted[:, 3:6] - incoming, recovered_edges), dim=-1)
    torch.testing.assert_close(recovered, weights, atol=1e-14, rtol=0)
    assert not torch.equal(converted, weights)


@pytest.mark.parametrize("kind", ["crz-chain", "zz-chain", "xx-chain"])
def test_two_layer_product_input_chain_has_bond_at_most_four(kind):
    record = probe.schmidt_probe(kind, 2)
    ranks = record["numerical_ranks"]
    norm_error = record["norm_max_abs"]
    assert isinstance(ranks, list)
    assert isinstance(norm_error, float)
    assert max(ranks) <= 4
    assert norm_error < 1e-12
