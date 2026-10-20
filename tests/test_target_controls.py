"""Tiny real fits exercise common targets, fixed slots, plans and compact exports."""

import copy
import importlib.util
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from test_robust_benchmark import small_study as small_study

from qcse import benchmark


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/target_control_study.py"
    spec = importlib.util.spec_from_file_location("target_controls", path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def cohort(small_study, tmp_path):
    root, manifest = small_study
    parents = []
    for objective in ("causal", "cbow"):
        parent = tmp_path / objective
        shutil.copytree(root, parent)
        m = copy.deepcopy(manifest)
        m["settings"]["objectives"] = [objective]
        m["tasks"] = [t for t in m["tasks"] if t["objective"] == objective]
        m.pop("fingerprint")
        m["fingerprint"] = benchmark.fingerprint(m)
        benchmark.write_json(parent / "manifest.json", m)
        parents.append(parent)
    control = module()
    return control, control.prepare(*parents, tmp_path / "controls")


def test_target_and_sampler_identity_and_fixed_right_slots(cohort):
    control, root = cohort
    hashes = []
    arrays = {}
    for recipe in control.RECIPES:
        manifest = benchmark.load_manifest(root / recipe)
        settings = manifest["settings"]
        assert settings["epochs"] == 100 and settings["samples_per_epoch"] == 20000
        assert settings["fixed_epochs"] and settings["seeds"] == [47, 48]
        (objective,) = settings["objectives"]
        with np.load(root / recipe / f"{objective}.npz") as a:
            arrays[recipe] = {k: a[k] for k in a.files}
        hashes.append(manifest["target_sha256"])
        assert settings["refit_controls"] == []
        assert all(t["fold"] == 0 and t["arm"]["lr"] == 0.003 for t in manifest["tasks"])
    assert hashes[0] == hashes[1] == hashes[2]
    np.testing.assert_array_equal(
        arrays["L2pad"]["contexts"][:, :2], arrays["L2R2"]["contexts"][:, :2]
    )
    vocabulary = manifest["vocabulary_size"]
    assert (arrays["L2pad"]["contexts"][:, 2:] == vocabulary).all()
    for seed in (47, 48):
        draws = [
            np.random.default_rng(seed).choice(a["train_0"], 20000, replace=True)
            for a in arrays.values()
        ]
        np.testing.assert_array_equal(draws[0], draws[1])
        np.testing.assert_array_equal(draws[0], draws[2])
    assert [len(benchmark.load_manifest(root / r)["tasks"]) for r in control.RECIPES] == [8, 8, 16]
    control.verify_tools(root)


def test_cuda_plan_contains_only_32_development_fits_and_rejects_oversized_fits(cohort):
    control, root = cohort
    for recipe in control.RECIPES:
        m = benchmark.load_manifest(root / recipe)
        (objective,) = m["settings"]["objectives"]
        arms = {t["arm"]["name"]: t["arm"] for t in m["tasks"]}
        benchmark.write_json(
            root / recipe / f"profile-{objective}.json",
            dict(
                manifest=m["fingerprint"],
                device="cuda",
                effective_source_sha256=m["code_sha256"],
                objective=objective,
                batch=64,
                microbatch=8,
                threads=4,
                measurements=[
                    dict(arm=a, training_batch_seconds=0.01, evaluation_batch_seconds=0.005)
                    for a in arms.values()
                ],
            ),
        )
    control.estimate(root)
    plan = json.loads((root / "compute.json").read_text())
    assert plan["total_tasks"] == 32 and plan["refits"] == 0
    p = root / "L4/profile-causal.json"
    profile = json.loads(p.read_text())
    profile["measurements"][0]["training_batch_seconds"] = 100
    benchmark.write_json(p, profile)
    with pytest.raises(ValueError, match="allocation allowance"):
        control.estimate(root)


def test_all_adapters_fit_tiny_controlled_arrays_and_export_only_validation(cohort):
    control, root = cohort
    torch.set_num_threads(1)
    for recipe in control.RECIPES:
        study = root / recipe
        m = benchmark.load_manifest(study)
        # Unit-fixture budget only; production preparation above always uses 100x20000.
        m["settings"].update(
            epochs=1,
            samples_per_epoch=4,
            batch_size=4,
            microbatch_size=2,
            eval_examples=4,
            threads=1,
        )
        m.pop("fingerprint")
        m["fingerprint"] = benchmark.fingerprint(m)
        benchmark.write_json(study / "manifest.json", m)
        assert not control.complete(study)
        for i, task in enumerate(m["tasks"]):
            result = benchmark.fit(study, task, study / "cv" / f"task-{i:04d}", "cpu")
            assert result["validation"] and result["stopped_epoch"] == 1
        assert control.complete(study)
    control.report(root)
    results = json.loads((root / "development-results.json").read_text())
    assert len(results["rows"]) == 32
    for r in results["rows"]:
        assert r["ledger"]["validation"]
        assert sum(r["rank_counts_1_to_5_then_miss"]) == r["ledger"]["metrics"]["examples"]
    archive = root.parent / (root.name + "-diagnostics.zip")
    with zipfile.ZipFile(archive) as a:
        assert a.testzip() is None
        assert sum(n.endswith("validation-clusters.npz") for n in a.namelist()) == 3
        assert not any(
            n.endswith(("checkpoint.pt", "selected.pt", "predictions.npz")) for n in a.namelist()
        )
    # Rerunning report is deterministic and does not duplicate the export manifest.
    control.report(root)
    with zipfile.ZipFile(archive) as a:
        assert len(a.namelist()) == len(set(a.namelist()))
