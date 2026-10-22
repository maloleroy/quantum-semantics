"""Independent numerical and structural tests for the research implementation."""

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from experiments.hamiltonian_study.data import graph_matrix, prepare
from experiments.hamiltonian_study.models import (
    HamiltonianModel,
    HamiltonianQCSEModel,
    QCSEModel,
    context_distribution,
)
from experiments.hamiltonian_study.run import metrics


def toy_data(size=7):
    rng = np.random.default_rng(4)
    cooc = rng.integers(0, 7, (size, size)).astype(float)
    return {"cooc": cooc, "transitions": cooc, "unigram": np.ones(size) / size}


@pytest.mark.parametrize("kind", ["count", "ppmi", "npmi"])
def test_graph_relabeling_and_ties(kind):
    data = toy_data()
    permutation = np.array([4, 1, 3, 0, 2, 6, 5])
    h, _ = graph_matrix(data["cooc"], kind, neighbors=2)
    hp, _ = graph_matrix(data["cooc"][np.ix_(permutation, permutation)], kind, neighbors=2)
    np.testing.assert_allclose(hp, h[np.ix_(permutation, permutation)], atol=1e-14)
    assert np.max(np.abs(np.linalg.eigvalsh(h))) <= 1 + 1e-12


@pytest.mark.parametrize("method", ["bag", "walk", "predictive", "directed", "diffusion"])
def test_model_relabeling(method):
    data = toy_data()
    p = np.array([4, 1, 3, 0, 2, 6, 5])
    inverse = np.argsort(p)
    altered = {k: v[p] if v.ndim == 1 else v[np.ix_(p, p)] for k, v in data.items()}
    torch.manual_seed(4)
    model = HamiltonianModel(data, method, 4)
    torch.manual_seed(4)
    changed = HamiltonianModel(altered, method, 4)
    context = torch.tensor([[1, 2, 4], [6, 5, 2]])
    old = model(context).detach().numpy()
    new = changed(torch.tensor(inverse[context])).detach().numpy()
    np.testing.assert_allclose(new, old[:, p], atol=2e-6)
    np.testing.assert_allclose(old.sum(1), 1, atol=1e-6)


def test_evolution_expm_and_gradient():
    torch.manual_seed(4)
    model = HamiltonianModel(toy_data(), layers=4)
    state = torch.randn(2, 7, dtype=torch.complex64)
    beta = torch.tensor(0.73, requires_grad=True)
    result = model.evolve(state, beta)
    reference = state.numpy() @ expm(-0.73j * model.hamiltonian.numpy()).T
    np.testing.assert_allclose(result.detach(), reference, atol=1e-6)
    loss = result[:, 2].abs().square().sum()
    loss.backward()
    step = 0.002

    def value(b):
        return float(model.evolve(state, torch.tensor(b))[:, 2].abs().square().sum())

    finite = (value(0.73 + step) - value(0.73 - step)) / (2 * step)
    assert beta.grad is not None
    assert abs(float(beta.grad) - finite) < 2e-4


def test_parameter_gradients():
    torch.manual_seed(8)
    model = HamiltonianModel(toy_data(), layers=3)
    contexts = torch.tensor([[1, 2, 4], [6, 5, 2]])
    loss = -model(contexts)[:, 3].log().mean()
    loss.backward()
    for parameter in [model.beta, model.gamma, model.mixture]:
        assert parameter.grad is not None
        analytic = parameter.grad.flatten()[0].item()
        step = 0.002
        with torch.no_grad():
            parameter.view(-1)[0] += step
            plus = -model(contexts)[:, 3].log().mean().item()
            parameter.view(-1)[0] -= 2 * step
            minus = -model(contexts)[:, 3].log().mean().item()
            parameter.view(-1)[0] += step
        assert abs(analytic - (plus - minus) / (2 * step)) < 3e-4


def test_repetitions_and_order():
    repeated = torch.tensor([[1, 1, 2]])
    coherent = context_distribution(repeated, 3, 1.0, True)
    empirical = context_distribution(repeated, 3, 1.0, False)
    np.testing.assert_allclose(coherent, [[0, 0.8, 0.2]], atol=1e-7)
    np.testing.assert_allclose(empirical, [[0, 2 / 3, 1 / 3]], atol=1e-7)
    c = torch.tensor([[1, 2, 3], [3, 2, 1]])
    bag = context_distribution(c, 4, 1.0)
    assert torch.equal(bag[0], bag[1])
    recency = context_distribution(c, 4, 0.7)
    assert not torch.allclose(recency[0], recency[1])


def test_qcse_against_existing_gate_simulator():
    torch.manual_seed(7)
    model = QCSEModel(7, 4)
    contexts = np.array([[-1, -1, 1], [1, 2, 3]])
    initial = model.encode(contexts)
    actual = model.states(initial).detach().numpy()
    expected = model.sim.ansatz(initial.numpy(), model.weights.detach().numpy().ravel())
    np.testing.assert_allclose(actual, expected, atol=4e-7)
    assert np.allclose(model(initial).detach().numpy().sum(1), 1)


def test_train_only_vocabulary_and_grouped_split(tmp_path):
    corpus = tmp_path / "sentences.txt"
    text = "\n".join(f"common unique{chr(97 + i)} rare{chr(97 + i)}" for i in range(20))
    corpus.write_text(text + "\n" + text)
    report = prepare(
        corpus, tmp_path / "prepared", vocabulary_size=0, train_limit=0, val_limit=0, test_limit=0
    )
    data = np.load(tmp_path / "prepared" / "dataset.npz")
    train = set(data["train_groups"])
    val = set(data["validation_groups"])
    test = set(data["test_groups"])
    assert not (train & val or train & test or val & test)
    assert len(data["vocabulary"]) == 1 + 1 + 2 * len(train)
    assert report["partitions"]["test"]["sample_unk_target_rate"] == 1.0
    assert data["transitions"].sum() == len(data["train_targets"])


def test_metrics_tie_equivariance():
    result, _ = metrics(np.full((2, 7), 1 / 7), np.array([1, 2]))
    assert result["top1"] == pytest.approx(1 / 7)
    assert result["top5"] == pytest.approx(5 / 7)


def test_binary_padding_and_context_phases():
    from qiskit.quantum_info import Statevector

    from experiments.hamiltonian_study.circuits import binary_circuit

    h, _ = graph_matrix(toy_data(5)["cooc"])
    c = np.array([0.1, 0.3, 0.2, 0.1, 0.3])
    state = np.sqrt(c).astype(complex)
    betas, gammas = [0.4, -0.7], [0.3, 1.1]
    circuit = binary_circuit(h, c, state, betas, gammas)
    actual = Statevector.from_instruction(circuit).data
    for b, g in zip(betas, gammas, strict=True):
        state = expm(-1j * b * h) @ (np.exp(-1j * g * c) * state)
    np.testing.assert_allclose(actual[:5], state, atol=1e-12)
    np.testing.assert_allclose(actual[5:], 0, atol=1e-12)


def test_ordered_position_potentials_and_future_exclusion():
    torch.manual_seed(5)
    model = HamiltonianModel(toy_data(), "ordered", layers=16)
    context = torch.tensor([[1, 2, 3], [3, 2, 1]])
    p = model(context).detach().numpy()
    assert np.max(np.abs(p[0] - p[1])) > 1e-4
    np.testing.assert_allclose(p.sum(1), 1, atol=3e-6)


def test_context_global_phase_and_final_diagonal():
    model = HamiltonianModel(toy_data(), "no_graph", layers=16)
    c = torch.tensor([[1, 2, 3]])
    p = model(c).detach()
    with torch.no_grad():
        model.gamma.add_(20.0)
    torch.testing.assert_close(model(c), p, atol=2e-6, rtol=2e-6)
    state = torch.randn(2, 7, dtype=torch.complex64)
    phase = torch.exp(-1j * torch.randn(2, 7))
    torch.testing.assert_close((state * phase).abs().square(), state.abs().square())


def test_classical_identity_control_matches_quantum_graph_removal():
    from experiments.hamiltonian_study.calibration import CalibratedBigram

    data = toy_data()
    model = HamiltonianModel(data, "predictive_residual", layers=16)
    calibration = CalibratedBigram()
    contexts = torch.tensor([[1, 2, 3], [3, 2, 1]])
    with torch.no_grad():
        model.eigenvalues.zero_()
        actual = model(contexts)
        expected = calibration(model.transition[contexts[:, -1]], model.unigram)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_qcse_joint_readout_retains_correlations():
    torch.manual_seed(14)
    joint = QCSEModel(8, layers=4, born_readout=True)
    marginal = QCSEModel(8, layers=4)
    marginal.load_state_dict(joint.state_dict())
    inputs = joint.encode(np.array([[1, 2, 3], [4, 1, 2]]))
    p = joint(inputs).detach().numpy()
    expected = abs(joint.sim.ansatz(inputs.numpy(), joint.weights.detach().numpy().ravel())) ** 2
    np.testing.assert_allclose(p, expected, atol=5e-7)
    assert np.max(abs(p - marginal(inputs).detach().numpy())) > 0.01
    with pytest.raises(ValueError, match="power-of-two"):
        invalid = QCSEModel(7, born_readout=True)
        invalid(invalid.encode(np.array([[1, 2]])))


@pytest.mark.parametrize("layers", [0, 4])
def test_hamiltonian_encoder_with_existing_causal_circuit(layers):
    from qiskit.quantum_info import Statevector

    from qcse.circuit import ansatz_circuit

    data = toy_data(7)
    torch.manual_seed(17)
    model = HamiltonianQCSEModel(data, layers=layers, evolution_time=2.3)
    contexts = torch.tensor([[-1, 1, 2], [3, 4, 5]])
    h, _ = graph_matrix(data["cooc"])
    initial = np.sqrt(context_distribution(contexts, 7).numpy()) @ expm(-2.3j * h).T
    initial = np.pad(initial, ((0, 0), (0, 1)))
    actual_inputs = model.encode(contexts)
    np.testing.assert_allclose(actual_inputs, initial, atol=3e-7)
    if layers:
        circuit, parameters = ansatz_circuit(3, layers, "forward")
        weights = model.weights.detach().numpy().ravel()
        circuit = circuit.assign_parameters(dict(zip(parameters, weights, strict=True)))
        states = np.array([Statevector(row).evolve(circuit).data for row in initial])
    else:
        states = initial
    np.testing.assert_allclose(model.states(actual_inputs).detach(), states, atol=6e-7)
    # Independent implementation of the unchanged marginal-bit word decoder.
    basis_bits = (np.arange(8)[:, None] >> np.arange(3)) & 1
    bits = np.clip(abs(states) ** 2 @ basis_bits, 1e-6, 1 - 1e-6)
    word_prob = np.prod(
        np.where(basis_bits[None, :7, :], bits[:, None, :], 1 - bits[:, None, :]), axis=2
    )
    word_prob /= word_prob.sum(1, keepdims=True)
    np.testing.assert_allclose(model(actual_inputs).detach(), word_prob, atol=5e-7)
