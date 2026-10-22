"""Verify CRZ/ZZ chain coordinate equivalence without corpus training."""

import argparse
import json
from pathlib import Path

import torch
from torch.func import functional_call

from qcse.benchmark_models import HybridCircuitModel, VariationalCircuit


def crz_to_zz(weights, qubits):
    """Same RX angles; shift target RZ fields and halve/negate chain couplings.

    CRZ(c,t,g) = RZ(t,g/2) RZZ(c,t,-g/2). Every diagonal gate in
    a layer commutes, so free local RZ angles absorb the extra target fields.
    """
    if weights.shape[-1] != 3 * qubits - 1:
        raise ValueError("Expected an RX/RZ/CRZ directed-chain parameter layout")
    edges = weights[..., 2 * qubits :]
    incoming = torch.cat((torch.zeros_like(weights[..., :1]), edges / 2), dim=-1)
    return torch.cat(
        (weights[..., :qubits], weights[..., qubits : 2 * qubits] + incoming, -edges / 2), dim=-1
    )


def probe(qubits=8, layers=2, seed=47):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    crz = VariationalCircuit(qubits, layers, "crz-chain", scale=0.3).double()
    zz = VariationalCircuit(qubits, layers, "zz-chain", scale=0.3).double()
    initial = torch.randn(3, 2**qubits, 2, dtype=torch.float64)
    initial = (initial / initial.square().sum((1, 2), keepdim=True).sqrt()).requires_grad_()
    left = crz(initial)
    right = functional_call(zz, {"weights": crz_to_zz(crz.weights, qubits)}, (initial,))
    observable = torch.randn_like(left)
    a = torch.autograd.grad((left * observable).sum(), (crz.weights, initial), retain_graph=True)
    b = torch.autograd.grad((right * observable).sum(), (crz.weights, initial))
    return dict(
        qubits=qubits,
        layers=layers,
        seed=seed,
        state_max_abs=float((left - right).abs().max().detach()),
        coordinate_gradient_max_abs=float((a[0] - b[0]).abs().max()),
        input_gradient_max_abs=float((a[1] - b[1]).abs().max()),
    )


def schmidt_probe(kind, layers, seed=47, qubits=8):
    """Product inputs plus one rank-two gate per cut/layer give bond <= 2**layers.

    This checks the implemented state, not an MPS evaluator or its runtime.
    No bound is asserted for the entangled input of the separate paper adapter.
    """
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = HybridCircuitModel(
        9, 4, dict(qubits=qubits, layers=layers, ansatz=kind, init_scale=0.3)
    ).double()
    angles = torch.randn(3, 2 * qubits, dtype=torch.float64)
    with torch.no_grad():
        state = model.state_from_angles(angles)
        vector = torch.complex(state[..., 0], state[..., 1])
        ranks, bounds = [], []
        for cut in range(1, qubits):
            singular = torch.linalg.svdvals(vector.reshape(3, 2**cut, 2 ** (qubits - cut)))
            rank = int((singular > 1e-10).sum(-1).max())
            bound = 2 ** min(layers, cut, qubits - cut)
            assert rank <= bound
            ranks.append(rank)
            bounds.append(bound)
    return dict(
        kind=kind,
        qubits=qubits,
        layers=layers,
        seed=seed,
        numerical_ranks=ranks,
        rank_tolerance=1e-10,
        upper_bounds=bounds,
        norm_max_abs=float((vector.abs().square().sum(-1) - 1).abs().max()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [probe(q, depth, seed) for q in (2, 3, 8) for depth in (1, 2, 4) for seed in (47, 48)]
    assert (
        max(
            r[k]
            for r in rows
            for k in ("state_max_abs", "coordinate_gradient_max_abs", "input_gradient_max_abs")
        )
        < 1e-12
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                dtype="float64",
                device="cpu",
                conditions=rows,
                product_input_schmidt_checks=[
                    schmidt_probe(kind, depth, seed)
                    for kind in ("crz-chain", "zz-chain", "xx-chain")
                    for depth in (1, 2, 4)
                    for seed in (47, 48)
                ],
                conclusion="Same unconstrained chain function class with free per-layer RZ fields. "
                "Independent Gaussian initialization and coordinate-wise Adam are not invariant "
                "under this non-orthogonal linear reparameterization. No training was performed.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
