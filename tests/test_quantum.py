import math

import pytest
import torch

from src.encoding import amplitudes
from src.quantum import cnot, evolve, hermitian, independent_bits, readout, rotate, seed_random


@pytest.mark.parametrize("scale", [0.0, 0.2])
def test_hermitian_evolution_and_gradient_match_generic_exponential(scale: float) -> None:
    torch.set_num_threads(1)
    seed_random(19)
    coefficients = (torch.randn(16, dtype=torch.float64) * scale).requires_grad_(True)
    matrix = hermitian(coefficients, 4)
    actual, expected = evolve(matrix), torch.matrix_exp(-1j * matrix)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
    cotangent = torch.randn((4, 4), dtype=torch.complex128)
    first = torch.autograd.grad(
        (actual.conj() * cotangent).real.sum(), coefficients, retain_graph=True
    )[0]
    second = torch.autograd.grad((expected.conj() * cotangent).real.sum(), coefficients)[0]
    torch.testing.assert_close(first, second, atol=1e-12, rtol=1e-10)

    def operator(value: torch.Tensor) -> torch.Tensor:
        return evolve(hermitian(value, 4))

    assert torch.autograd.gradcheck(operator, (coefficients,))


def test_gates_readout_and_joint_distribution() -> None:
    zero = torch.tensor([[1, 0]], dtype=torch.complex128)
    one = rotate(zero, 0, torch.tensor(math.pi, dtype=torch.float64), "y")
    torch.testing.assert_close(one.abs().square(), torch.tensor([[0, 1]], dtype=torch.float64))
    pair = torch.tensor([[0, 1, 0, 0]], dtype=torch.complex128)
    torch.testing.assert_close(
        cnot(pair, 0, 1), torch.tensor([[0, 0, 0, 1]], dtype=torch.complex128)
    )
    plus = torch.tensor([[1, 1]], dtype=torch.complex128) / math.sqrt(2)
    changed = rotate(plus, 0, torch.tensor(math.pi, dtype=torch.float64), "z")
    torch.testing.assert_close(plus.abs().square(), changed.abs().square())
    basis = torch.tensor([[0, math.pi / 2]], dtype=torch.float64)
    assert not torch.allclose(
        readout(plus, basis).abs().square(), readout(changed, basis).abs().square()
    )
    bell = torch.tensor([[0.5, 0, 0, 0.5]], dtype=torch.float64)
    torch.testing.assert_close(
        independent_bits(bell), torch.full((1, 4), 0.25, dtype=torch.float64)
    )


def test_order_repeats_padding_and_vocabulary_renaming() -> None:
    contexts = torch.tensor([[-1, -1, 1, 2], [1, 2, 1, 2]])
    encoded = amplitudes(contexts, 8)
    torch.testing.assert_close(encoded.abs().square().sum(1), torch.ones(2, dtype=torch.float64))
    assert not torch.allclose(encoded[0], amplitudes(torch.tensor([[-1, -1, 2, 1]]), 8)[0])
    permutation = torch.tensor([4, 2, 6, 0, 1, 7, 3, 5])
    renamed = torch.where(contexts >= 0, permutation[contexts.clamp_min(0)], -1)
    torch.testing.assert_close(amplitudes(renamed, 8)[:, permutation], encoded)
