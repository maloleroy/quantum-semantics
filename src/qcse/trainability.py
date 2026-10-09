"""Circuit-only gradient diagnostics; small gradients alone do not prove a barren plateau."""

import numpy as np
import torch

from .benchmark_models import PaperQCSE, VariationalCircuit


def circuit_gradient_summary(model):
    """Read accumulated gradients before clipping; ignore known dead Z-readout gates."""
    circuit = getattr(model, "circuit", None)
    if not isinstance(circuit, VariationalCircuit) or circuit.weights.grad is None:
        return None
    gradient = circuit.weights.grad.detach()
    mask = (
        circuit.z_readout_mask()
        if isinstance(model, PaperQCSE)
        else torch.ones_like(gradient, dtype=torch.bool)
    )
    return gradient_summary(gradient, mask)


def gradient_summary(gradient, mask) -> dict:
    if not torch.isfinite(gradient).all():
        raise ValueError("Non-finite circuit gradient")
    counts = mask.sum(-1)
    if not bool((counts > 0).all()):
        raise ValueError("Each layer must have at least one eligible parameter")
    absolute = gradient.abs() * mask
    return dict(
        layer_rms=((gradient.square() * mask).sum(-1) / counts).sqrt().cpu().tolist(),
        layer_max_abs=absolute.max(-1).values.cpu().tolist(),
        layer_near_zero_fraction=((gradient.abs() <= 1e-10) & mask)
        .sum(-1)
        .div(counts)
        .cpu()
        .tolist(),
        active_parameters=int(mask.sum()),
        structurally_dead_parameters=int((~mask).sum()),
    )


def probe_gradients(qubits, layers, scale, seeds, batch_size=4, device="cpu"):
    """Paired local/global initial gradients on fixed product inputs, without fitting.

    The local cost is (1-<Z_0>)/2; the global cost is 1-P(all zeros).
    Neither is the NLP loss. Inputs use the same prefix of a fixed 16-qubit
    angle array for all register sizes. Seeds vary circuit initialization only.
    """
    if not 2 <= qubits <= 16 or batch_size < 1 or len(seeds) < 2:
        raise ValueError("Probe needs 2..16 qubits, a positive batch and at least two seeds")
    gradients = {name: [] for name in ("local", "global")}
    losses = {name: [] for name in gradients}
    mask = None
    # Circuit randomness is isolated from the fixed synthetic input distribution.
    data = np.random.default_rng(619)
    angles = np.stack(
        (data.uniform(1, 2, (batch_size, 16)), data.uniform(-np.pi, np.pi, (batch_size, 16))), -1
    )[:, :qubits]
    for seed in seeds:
        torch.manual_seed(seed)
        circuit = VariationalCircuit(qubits, layers, scale=scale).to(device).double()
        register = circuit.register
        state = register.z.new_zeros(batch_size, 2**qubits, 2)
        state[:, 0, 0] = 1
        values = register.z.new_tensor(angles)
        for q in range(qubits):
            state = register.rotate(state, values[:, q, 0], q, "y")
        state = circuit(register.phase(state, values[:, :, 1]))
        probabilities = state.square().sum(-1)
        costs = {
            "local": ((1 - probabilities @ register.z[:, 0]) / 2).mean(),
            "global": (1 - probabilities[:, 0]).mean(),
        }
        for name, loss in costs.items():
            gradient = torch.autograd.grad(loss, circuit.weights, retain_graph=name == "local")[0]
            if not torch.isfinite(gradient).all() or not torch.isfinite(loss):
                raise ValueError("Non-finite probe loss/gradient")
            gradients[name].append(gradient.detach().cpu().numpy())
            losses[name].append(float(loss.detach()))
        mask = circuit.z_readout_mask().cpu().numpy()
    assert mask is not None
    rows, arrays = [], {}
    for name, samples in gradients.items():
        values = np.stack(samples)
        variance = values.var(0, ddof=1)
        rows.append(
            dict(
                qubits=qubits,
                layers=layers,
                init_scale=scale,
                cost=name,
                seeds=len(seeds),
                parameters=int(mask.sum()),
                structurally_dead_parameters=int((~mask).sum()),
                first_rx_gradient_mean=float(values[:, 0, 0].mean()),
                first_rx_gradient_variance=float(variance[0, 0]),
                mean_eligible_gradient_variance=float(variance[mask].mean()),
                rms_gradient=float(np.sqrt(np.mean(values[:, mask] ** 2))),
                sampled_near_zero_fraction=float(np.mean(np.abs(values[:, mask]) <= 1e-10)),
                loss_mean=float(np.mean(losses[name])),
            )
        )
        arrays[f"{name}_gradients"] = values
        arrays[f"{name}_losses"] = np.asarray(losses[name])
    arrays["eligible_mask"] = mask
    arrays["input_angles"] = angles
    arrays["seeds"] = np.asarray(seeds)
    return rows, arrays
