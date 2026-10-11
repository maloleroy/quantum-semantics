import numpy as np
import pytest
import torch

from qcse.benchmark import configurations
from qcse.benchmark_models import HybridCircuitModel, PaperQCSE, VariationalCircuit, training_loss
from qcse.trainability import circuit_gradient_summary, probe_gradients


@pytest.mark.parametrize("kind", ["crz-chain", "zz-chain", "ry-cx", "local", "xx-chain"])
def test_readout_mask_excludes_only_guaranteed_final_diagonals(kind):
    torch.manual_seed(61)
    circuit = VariationalCircuit(3, 2, kind, scale=0.7).double()
    state = torch.randn(2, 8, 2, dtype=torch.double)
    state /= state.square().sum((1, 2), keepdim=True).sqrt()
    result = circuit(state)
    (result.square().sum(-1) @ circuit.register.z).sum().backward()
    mask = circuit.z_readout_mask()
    assert circuit.weights.grad is not None
    torch.testing.assert_close(
        circuit.weights.grad[~mask],
        torch.zeros_like(circuit.weights.grad[~mask]),
        atol=1e-12,
        rtol=0,
    )
    if kind == "xx-chain":
        assert bool(mask.all())
        assert circuit.weights.grad[-1, 3:6].abs().max() > 1e-4
    else:
        assert bool(mask[-1, :3].all()) and not bool(mask[-1, 3:].any())


@pytest.mark.parametrize("qubits", [2, 4, 8])
def test_hybrid_qubit_control_keeps_word_task_and_all_xyz_parameters(qubits):
    torch.manual_seed(42)
    model = HybridCircuitModel(16, 4, dict(qubits=qubits, layers=2))
    contexts = torch.tensor([[16, 16, 1, 2], [3, 4, 5, 6]])
    scores = model(contexts)
    assert scores.shape == (2, 16)
    training_loss(model, contexts, torch.tensor([3, 7])).backward()
    assert model.circuit.weights.grad is not None
    before = model.circuit.weights.grad.clone()
    diagnostic = circuit_gradient_summary(model)
    assert diagnostic is not None
    assert diagnostic["structurally_dead_parameters"] == 0
    assert diagnostic["active_parameters"] == model.circuit.weights.numel()
    assert len(diagnostic["layer_rms"]) == 2
    torch.testing.assert_close(model.circuit.weights.grad, before, rtol=0, atol=0)


def test_paper_diagnostic_does_not_count_dead_diagonal_gradients():
    model = PaperQCSE(16, dict(layers=2))
    training_loss(model, torch.tensor([[1, 2, 3, 4]]), torch.tensor([2])).backward()
    diagnostic = circuit_gradient_summary(model)
    assert diagnostic is not None
    assert diagnostic["structurally_dead_parameters"] == 7
    assert diagnostic["active_parameters"] == 15


def test_probe_paired_costs_reproducible_variance_and_saved_dead_coordinates():
    rows, arrays = probe_gradients(3, 2, 1.0, [42, 43, 44])
    again, repeated = probe_gradients(3, 2, 1.0, [42, 43, 44])
    assert rows == again
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], repeated[key])
    assert [row["cost"] for row in rows] == ["local", "global"]
    for row in rows:
        gradients = arrays[f"{row['cost']}_gradients"]
        assert gradients.shape == (3, 2, 8)
        np.testing.assert_allclose(gradients[:, ~arrays["eligible_mask"]], 0, atol=1e-12)
        assert row["first_rx_gradient_variance"] == pytest.approx(gradients[:, 0, 0].var(ddof=1))
        assert row["first_rx_gradient_variance"] > 0
    _, bigger = probe_gradients(4, 2, 1.0, [42, 43, 44])
    np.testing.assert_array_equal(arrays["input_angles"], bigger["input_angles"][:, :3])


def test_opt_in_matrix_adds_unique_controls_without_changing_standard():
    standard, extended = configurations(), configurations(True)
    assert len(standard) == 43 and len(extended) == 57
    assert len({arm["name"] for arm in extended}) == 57
    assert extended[:43] == standard
    controls = [a for a in extended if a not in standard]
    assert len(controls) == 14
    assert all(a["family"] == "hybrid" for a in controls)
    assert {a.get("qubits", 4) for a in controls} == {2, 4, 8}
