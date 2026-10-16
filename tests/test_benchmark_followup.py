import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_robust_benchmark import small_study as small_study
from torch.nn import functional as F

from qcse import benchmark
from qcse.benchmark_models import ClassicalContext, HybridCircuitModel, create_benchmark_model
from qcse.benchmark_report import report
from qcse.semantic import SemanticModel


def script(name="prepare_benchmark_followup.py"):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location("followup", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("qubits", [2, 4, 8])
@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-12), (torch.float32, 3e-6)])
def test_analytic_identity_logits_and_all_parameter_gradients(qubits, dtype, tolerance):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    actual = HybridCircuitModel(17, 4, dict(qubits=qubits, ansatz="identity")).to(dtype)
    reference = copy.deepcopy(actual)
    contexts = torch.tensor([[17, 17, 1, 2], [3, 4, 5, 6]])
    targets = torch.tensor([3, 7])
    left = actual(contexts)
    hidden = SemanticModel.representation(reference, contexts)
    right = F.linear(hidden, reference.embedding.weight[:-1], reference.output_bias)
    torch.testing.assert_close(left, right, atol=tolerance, rtol=tolerance)
    F.cross_entropy(left, targets).backward()
    F.cross_entropy(right, targets).backward()
    for (name, a), (other, b) in zip(
        actual.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == other
        if a.grad is not None:
            assert b.grad is not None
            torch.testing.assert_close(a.grad, b.grad, atol=tolerance, rtol=tolerance)


def test_same_size_circuit_controls_share_encoder_and_decoder_initialization():
    models = []
    for kind, frozen in [
        ("identity", False),
        ("crz-chain", False),
        ("crz-chain", True),
        ("local", False),
        ("xx-chain", False),
        ("zz-chain", False),
    ]:
        torch.manual_seed(43)
        models.append(HybridCircuitModel(17, 4, dict(qubits=8, ansatz=kind, frozen=frozen)))
    for other in models[1:]:
        for name, value in models[0].state_dict().items():
            if name.startswith(("embedding.", "encoder.", "decoder.")) or name == "output_bias":
                torch.testing.assert_close(value, other.state_dict()[name], atol=0, rtol=0)


def test_followup_grid_capacity_budget_and_frozen_common_lr_selection(monkeypatch):
    module = script()
    known = {arm["name"]: arm for arm in benchmark.configurations(True)}
    refine = module.settings_for("refine", "cbow")
    assert len(refine["arms"]) == len(set(refine["arms"])) == 18
    assert set(refine["arms"]) <= known.keys()
    assert refine["folds"] == 3 and refine["seeds"] == [43]
    assert refine["epochs"] == 100 and refine["samples_per_epoch"] == 20000
    assert refine["fixed_epochs"] and refine["final_eval_sentences"] == 0
    counts = [
        sum(p.numel() for p in create_benchmark_model(10864, 4, "causal", known[n]).parameters())
        for n in ("mlp-32", "bert-32")
    ]
    assert 0.9 < counts[0] / counts[1] < 1.1  # Near-capacity control, not exact parameter matching.
    candidates: dict[str, dict] = {
        name: dict(objective="causal", arm=known[name], mean=dict(top1=0.1, top5=0.2, nll=5.0))
        for name in refine["arms"]
    }
    for name in module.LR_ARMS[0.003][2:]:
        candidates[name]["mean"]["top1"] = 0.5
    candidates[module.LR_ARMS[0.01][2]]["mean"]["top1"] = 0.8
    candidates[module.LR_ARMS[0.01][0]]["mean"]["top1"] = 0.6
    candidates[module.LR_ARMS[0.0003][1]]["mean"]["top1"] = 0.7
    monkeypatch.setattr(
        module, "load_manifest", lambda _: dict(settings=dict(followup=dict(stage="refine")))
    )
    monkeypatch.setattr(
        module,
        "load_selection",
        lambda _: dict(candidates=list(candidates.values()), fingerprint="parent"),
    )
    confirmation = module.settings_for("confirm", "causal", Path("unused"))
    assert confirmation["arms"][1:3] == module.LR_ARMS[0.003][2:]
    assert confirmation["arms"][3:] == [module.LR_ARMS[0.0003][1], module.LR_ARMS[0.01][0]]
    assert confirmation["refit_controls"] == confirmation["arms"]
    assert confirmation["folds"] == 5 and confirmation["seeds"] == [44, 45, 46]
    assert confirmation["primary_confidence"] == 0.975
    assert confirmation["followup"]["parent_selection"] == "parent"
    for stage in ("window-2", "window-8", "phrases", "cleaned"):
        settings = module.settings_for(stage, "causal", Path("unused"))
        assert settings["arms"] == confirmation["arms"]
        assert settings["folds"] == 3 and settings["seeds"] == [43]


def test_retained_same_family_refits_remain_distinct_and_paired(small_study):
    original, original_manifest = small_study
    root = original / "retained"
    arms = ["hybrid-crz-chain", "hybrid-identity"]
    manifest = benchmark.prepare(
        root,
        dict(
            original_manifest["settings"],
            objectives=["causal"],
            arms=arms,
            refit_controls=arms,
            epochs=1,
            fixed_epochs=True,
            primary_comparisons=[arms],
            primary_confidence=0.975,
        ),
    )
    for i, task in enumerate(manifest["tasks"]):
        benchmark.fit(root, task, root / "cv" / f"task-{i:04d}")
    selected = benchmark.selection(root)
    assert len(selected["refits"]) == 4
    for i, task in enumerate(selected["refits"]):
        benchmark.fit(root, task, root / "refit" / f"task-{i:04d}", refit_epochs=task["epochs"])
    report(root, draws=20)
    result = json.loads((root / "report/results.json").read_text())
    assert {r["arm"] for r in result["summaries"]} == set(arms)
    primary = [r for r in result["paired_intervals"] if r["primary"]]
    assert len(primary) == 1 and [primary[0]["left"], primary[0]["right"]] == arms
    assert primary[0]["confidence"] == 0.975
    by_arm = {name: [r for r in result["refits"] if r["arm"] == name] for name in arms}
    expected = np.mean([r["top1"] for r in by_arm[arms[0]]]) - np.mean(
        [r["top1"] for r in by_arm[arms[1]]]
    )
    assert primary[0]["difference"][1] == pytest.approx(expected)


def test_primary_interval_is_wider_with_identical_bootstrap_draws():
    left = np.random.default_rng(42).normal(size=(3, 10, 3))
    right = np.zeros_like(left)
    ordinary = benchmark.paired_bootstrap(np.arange(10), left, right, draws=200)
    primary = benchmark.paired_bootstrap(np.arange(10), left, right, draws=200, confidence=0.975)
    assert np.all(np.array(primary["low"]) <= ordinary["low"])
    assert np.all(np.array(primary["high"]) >= ordinary["high"])
    for bad in (0, 1, float("nan")):
        with pytest.raises(ValueError):
            benchmark.paired_bootstrap(np.arange(10), left, right, confidence=bad)


def test_actual_loss_probe_fixes_common_weights_across_depth_and_repeats():
    module = script("probe_nlp_trainability.py")
    contexts = torch.tensor([[17, 17, 1, 2], [3, 4, 5, 6]])
    targets = torch.tensor([3, 7])
    args = (17, 4, contexts, targets, 2, 2, 0.05, [42, 43, 44], "cpu")
    a, x = module.probe(*args)
    b, y = module.probe(*args)
    deeper, _ = module.probe(17, 4, contexts, targets, 2, 4, 0.05, [42, 43, 44], "cpu")
    assert a == b and a["common_state_sha256"] == deeper["common_state_sha256"]
    np.testing.assert_array_equal(x["gradients"], y["gradients"])
    assert a["first_rx_gradient_variance"] > 0
    assert np.isfinite(x["gradients"]).all()


def test_mean_initialization_control_changes_only_scale():
    torch.manual_seed(43)
    default = create_benchmark_model(17, 4, "causal", dict(model="mean"))
    torch.manual_seed(43)
    scaled = create_benchmark_model(17, 4, "causal", dict(model="mean", embedding_std=0.1))
    assert isinstance(default, ClassicalContext) and isinstance(scaled, ClassicalContext)
    torch.testing.assert_close(
        scaled.embedding.weight, default.embedding.weight * 0.1, atol=0, rtol=0
    )
    torch.testing.assert_close(scaled.bias, default.bias, atol=0, rtol=0)


def test_followup_pilot_records_actual_training_loss_probe(tmp_path, monkeypatch):
    import sys

    module = script("run_robust_benchmark_job.py")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                tasks=[dict(objective="cbow", arm=dict(qubits=8))],
                settings=dict(followup=dict(stage="refine")),
            )
        )
    )
    commands = []
    monkeypatch.setattr(
        module.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    monkeypatch.setattr(sys, "argv", ["job", "pilot", str(tmp_path), "--task-id", "0"])
    module.main()
    probe = commands[-2]
    assert probe[1:3] == ["scripts/probe_nlp_trainability.py", str(tmp_path)]
    assert probe[-2:] == ["--device", "cuda"]
    ledger = json.loads(next((tmp_path / "job-timings").glob("*.json")).read_text())
    assert ledger["phases"][-2]["name"] == "nlp_trainability_probe"
