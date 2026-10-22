"""Check the causal mask, word head, paired initialization and gradient flow."""

import importlib.util
from pathlib import Path

import torch
from torch.nn import functional as F

from qcse.benchmark_models import HybridCircuitModel


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/research_model_controls.py"
    spec = importlib.util.spec_from_file_location("research_controls", path)
    assert spec and spec.loader
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_decoder_is_causal_and_predicts_only_valid_words_without_label_shift():
    torch.manual_seed(47)
    model = module().CausalWordDecoder(9, width=8, layers=1).eval()
    context = torch.tensor([[1, 2, 3, 4]])
    changed = torch.tensor([[1, 2, 7, 8]])
    a, b = model.sequence_logits(context), model.sequence_logits(changed)
    torch.testing.assert_close(a[:, :2], b[:, :2], atol=1e-7, rtol=0)
    assert not torch.allclose(a[:, -1], b[:, -1])
    assert model(context).shape == (1, 9)
    torch.testing.assert_close(model(context), a[:, -1])
    assert (
        model.gpt.get_input_embeddings().weight.data_ptr()
        == model.gpt.get_output_embeddings().weight.data_ptr()
    )
    F.cross_entropy(model(context), torch.tensor([5])).backward()
    assert model.gpt.transformer.h[0].attn.c_attn.weight.grad.norm() > 0


def test_decoder_padding_mask_and_supplied_context_are_explicit():
    model = module().CausalWordDecoder(9, width=8, layers=1).eval()
    captured = {}

    def before(_, args, kwargs):
        captured.update(kwargs)

    hook = model.gpt.register_forward_pre_hook(before, with_kwargs=True)
    contexts = torch.tensor([[9, 9, 2, 3]])
    model(contexts)
    hook.remove()
    torch.testing.assert_close(captured["input_ids"], contexts)
    torch.testing.assert_close(
        captured["attention_mask"], torch.tensor([[False, False, True, True]])
    )
    assert "labels" not in captured


def test_classical_features_preserve_common_initial_weights_and_backpropagate():
    torch.manual_seed(47)
    quantum = HybridCircuitModel(9, 4, dict(qubits=3, layers=2))
    torch.manual_seed(47)
    classical = module().ClassicalAngleFeatures(9, qubits=3, layers=2)
    for name in ("embedding", "encoder", "decoder"):
        for a, b in zip(
            getattr(quantum, name).parameters(), getattr(classical, name).parameters(), strict=True
        ):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(quantum.output_bias, classical.output_bias, atol=0, rtol=0)
    assert not hasattr(classical, "circuit")
    context = torch.tensor([[9, 1, 2, 3], [2, 3, 4, 5]])
    F.cross_entropy(classical(context), torch.tensor([4, 6])).backward()
    assert classical.feature_map[0].weight.grad.norm() > 0
    assert classical.encoder.weight.grad.norm() > 0
    assert classical.decoder.weight.grad.norm() > 0


def test_classical_feature_parameter_difference_is_disclosed():
    quantum = HybridCircuitModel(9, 4, dict(qubits=8, layers=2))
    classical = module().ClassicalAngleFeatures(9)

    def total(model):
        return sum(p.numel() for p in model.parameters())

    assert total(classical) - total(quantum) == 634
