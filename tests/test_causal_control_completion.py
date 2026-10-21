"""Eight-fit completion preserves parents; compact exports retain paired anchors."""

import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from test_robust_benchmark import small_study as small_study
from test_target_controls import cohort as cohort

from qcse import benchmark


@pytest.fixture
def completed_parent(cohort):
    control, parent = cohort
    rows = []
    for recipe in ("L4", "L2pad", "L2R2"):
        manifest = benchmark.load_manifest(parent / recipe)
        for index, task in enumerate(manifest["tasks"]):
            ledger = dict(
                manifest=manifest["fingerprint"],
                task=task,
                validation=True,
                selected_epoch=100,
                stopped_epoch=100,
                training_draws=2_000_000,
                optimizer_updates=31_300,
            )
            benchmark.write_json(
                parent / recipe / "cv" / f"task-{index:04d}" / "result.json", ledger
            )
            rows.append(
                dict(recipe=recipe, arm=task["arm"]["name"], seed=task["seed"], ledger=ledger)
            )
        # Sufficient statistics contents are irrelevant to preparation/copy tests.
        benchmark.save_npz(parent / recipe / "validation-clusters.npz", fixture=np.arange(4))
    benchmark.write_json(parent / "development-results.json", dict(rows=rows))
    return control, parent


def test_exactly_eight_missing_tasks_preserve_arrays_seeds_and_full_budget(
    completed_parent, tmp_path
):
    control, parent = completed_parent
    root = control.prepare_causal_completion(parent, tmp_path / "completion")
    control.verify_tools(root)
    before = benchmark.load_manifest(parent / "L4")
    after = benchmark.load_manifest(root / "L4")
    assert len(after["tasks"]) == 8
    assert {(t["arm"]["name"], t["seed"]) for t in after["tasks"]} == {
        (name, seed) for name in control.EXTRA for seed in (47, 48)
    }
    assert after["target_sha256"] == before["target_sha256"]
    assert after["artifact_sha256"] == before["artifact_sha256"]
    assert after["code_sha256"] == before["code_sha256"]
    assert (root / "L4/causal.npz").read_bytes() == (parent / "L4/causal.npz").read_bytes()
    for key in ("epochs", "samples_per_epoch", "microbatch_size", "batch_size", "fixed_epochs"):
        assert after["settings"][key] == before["settings"][key]
    assert before["tasks"] == benchmark.load_manifest(parent / "L4")["tasks"]
    report = json.loads((root / "anchors/development-results.json").read_text())
    assert len(report["rows"]) == 32
    (root / "anchors/L4-validation-clusters.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="parent evidence"):
        control.verify_tools(root)


def test_completion_report_has_eight_new_fits_and_includes_anchors(completed_parent, tmp_path):
    control, parent = completed_parent
    root = control.prepare_causal_completion(parent, tmp_path / "completion")
    study = root / "L4"
    m = benchmark.load_manifest(study)
    # Tiny integration fixture only; production preparation is checked above.
    m["settings"].update(
        epochs=1, samples_per_epoch=4, batch_size=4, microbatch_size=2, eval_examples=4, threads=1
    )
    m.pop("fingerprint")
    m["fingerprint"] = benchmark.fingerprint(m)
    benchmark.write_json(study / "manifest.json", m)
    torch.set_num_threads(1)
    for index, task in enumerate(m["tasks"]):
        benchmark.fit(study, task, study / "cv" / f"task-{index:04d}", "cpu")
    pilot = root / "pilot"
    pilot.mkdir()
    benchmark.save_npz(pilot / "microbatch-inputs-L4.npz", ids=np.array([1]))
    control.report(root)
    report = json.loads((root / "development-results.json").read_text())
    assert len(report["rows"]) == 8 and all(r["recipe"] == "L4" for r in report["rows"])
    with zipfile.ZipFile(root.parent / (root.name + "-diagnostics.zip")) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert sum(n.endswith("validation-clusters.npz") for n in names) == 3
        assert any(n.endswith("microbatch-inputs-L4.npz") for n in names)
        assert not any(n.endswith(("predictions.npz", "checkpoint.pt")) for n in names)


def test_changed_parent_budget_is_rejected(completed_parent, tmp_path):
    control, parent = completed_parent
    records = json.loads((parent / "development-results.json").read_text())
    records["rows"][0]["ledger"]["training_draws"] = 5000
    benchmark.write_json(parent / "development-results.json", records)
    with pytest.raises(ValueError, match="provenance or full budget"):
        control.prepare_causal_completion(parent, tmp_path / "completion")


def test_eight_fit_forecast_and_cuda_pilot_provenance(completed_parent, tmp_path):
    control, parent = completed_parent
    root = control.prepare_causal_completion(parent, tmp_path / "completion")
    manifest = benchmark.load_manifest(root / "L4")
    benchmark.write_json(
        root / "L4/profile-causal.json",
        dict(
            manifest=manifest["fingerprint"],
            device="cuda",
            objective="causal",
            batch=64,
            microbatch=8,
            threads=4,
            effective_source_sha256=manifest["code_sha256"],
            measurements=[
                dict(arm=t["arm"], training_batch_seconds=0.01, evaluation_batch_seconds=0.005)
                for t in manifest["tasks"][::2]
            ],
        ),
    )
    control.estimate(root)
    assert json.loads((root / "compute.json").read_text())["total_tasks"] == 8
    protocol = json.loads((root / "protocol.json").read_text())
    pilot: dict = dict(
        status="complete",
        device="cuda",
        all_parity_passed=True,
        effective_source_sha256=manifest["code_sha256"],
        probe_sha256=protocol["extension_sha256"]["profile_benchmark_microbatches.py"],
        manifests={r: benchmark.load_manifest(parent / r)["fingerprint"] for r in ("L4", "L2R2")},
        runtime=dict(cuda_device=dict(total_memory_bytes=10_200_547_328)),
        rows=[
            dict(recipe=r, arm=a, batch=b, microbatch=m)
            for r in ("L4", "L2R2")
            for a in control.MAIN
            for b in (64, 32)
            for m in (8, 16, 32, 64)
        ],
    )
    path = root / "pilot/microbatch-report.json"
    benchmark.write_json(path, pilot)
    control.verify_pilot(root)
    pilot["rows"].pop()
    benchmark.write_json(path, pilot)
    with pytest.raises(ValueError, match="Incomplete microbatch"):
        control.verify_pilot(root)


def pilot_module():
    path = Path(__file__).resolve().parents[1] / "scripts/profile_benchmark_microbatches.py"
    spec = importlib.util.spec_from_file_location("microbatch_probe", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind", ["mlp", "hybrid", "bert"])
def test_microbatch_parity_covers_partial_batch_and_adam_state(kind):
    from qcse.benchmark_models import create_benchmark_model

    pilot = pilot_module()
    arm = dict(model=kind, width=8, qubits=3, layers=1, lr=0.003)
    inputs = torch.tensor([[9, 1, 2, 3], [1, 2, 3, 4], [9, 9, 2, 3], [5, 4, 3, 2], [1, 1, 2, 2]])
    targets = torch.tensor([4, 5, 4, 1, 3])
    states = []
    for micro in (1, 2, 4, 8):
        torch.manual_seed(42)
        model = create_benchmark_model(9, 4, "causal", arm)
        states.append(pilot.parity_state(model, inputs, targets, micro, 0.003))
    assert all(pilot.compare(states[0], s)["passed"] for s in states)
    assert states[0]["moments"].numel() > 0
    broken = {**states[0], "loss": states[0]["loss"] + 0.01}
    assert not pilot.compare(states[0], broken)["passed"]


def test_actual_cpu_pilot_records_both_tracks_inputs_and_timing(completed_parent, tmp_path):
    _, parent = completed_parent
    output = tmp_path / "pilot"
    result = pilot_module().profile(
        parent,
        output,
        device="cpu",
        repeats=1,
        microbatches=(2, 4),
        batch_sizes=(4, 3),
        names=("mlp-32", "hybrid-q8-depth2-init0.05"),
        threads=1,
    )
    assert result["status"] == "complete" and result["adopted_microbatch"] == 8
    assert len(result["rows"]) == 16
    assert result["all_parity_passed"]
    assert all(r["median_training_seconds"] > 0 for r in result["rows"])
    for recipe in ("L4", "L2R2"):
        with np.load(output / f"microbatch-inputs-{recipe}.npz") as saved:
            assert saved["contexts"].shape == (4, 4)
            assert saved["ids"].shape == saved["targets"].shape == (4,)
