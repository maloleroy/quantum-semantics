import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_robust_benchmark import small_study as small_study

from qcse import benchmark
from qcse import benchmark_compute_report as compute
from qcse.benchmark_timing import FitTimer
from qcse.outputs import write_json


def test_fit_timer_preserves_failed_phase_and_synchronizes(tmp_path, monkeypatch):
    timer = FitTimer(tmp_path / "timer.json", "cuda", "attempt")
    calls = []
    monkeypatch.setattr(timer, "synchronize", lambda: calls.append("sync"))
    with timer.phase("training", 1):
        pass
    with pytest.raises(KeyboardInterrupt), timer.phase("validation", 1):
        raise KeyboardInterrupt
    data = json.loads(timer.path.read_text())
    assert len(calls) == 3  # No further device work is required after interruption.
    assert data["status"] == "interrupted"
    assert [p["status"] for p in data["phases"]] == ["complete", "interrupted"]
    assert data["seconds"] >= sum(p["seconds"] for p in data["phases"])


@pytest.mark.parametrize("failed_stage", [None, "backend_preflight", "profile"])
@pytest.mark.parametrize("objective", ["causal", "cbow"])
def test_job_runner_records_overhead_and_stops_on_failure(
    tmp_path, monkeypatch, failed_stage, objective
):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_robust_benchmark_job.py"
    spec = importlib.util.spec_from_file_location("benchmark_job", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    write_json(tmp_path / "manifest.json", dict(tasks=[dict(objective=objective, arm={})]))
    monkeypatch.setattr(sys, "argv", ["job", "pilot", str(tmp_path), "--task-id", "0"])
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "9115")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-test")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if failed_stage and (
            (failed_stage == "backend_preflight" and "scripts/check_backend.py" in command)
            or (failed_stage == "profile" and "scripts/profile_robust_benchmark.py" in command)
        ):
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module.subprocess, "run", run)
    if failed_stage:
        with pytest.raises(subprocess.CalledProcessError):
            module.main()
    else:
        module.main()
    files = list((tmp_path / "job-timings").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["job"] == "9115_0" and data["scheduler"]["CUDA_VISIBLE_DEVICES"] == "MIG-test"
    assert data["status"] == ("failed" if failed_stage else "complete")
    assert all(p["seconds"] >= 0 for p in data["phases"])
    if failed_stage:
        assert data["phases"][-1]["name"] == failed_stage
        assert all("scripts/robust_benchmark.py" not in c for c in commands)
    else:
        assert len(commands) == 6
        assert commands[3][commands[3].index("--objective") + 1] == objective
        assert commands[3][-1] == str(tmp_path / f"profile-{objective}.json")
        assert commands[-1][-5:] == ["--task-id", "0", "--device", "cuda", "--resume"]


def test_measured_report_uses_allocation_costs_without_pilot_profile_or_step_double_count(
    tmp_path, monkeypatch
):
    tasks = [
        dict(objective="causal", arm=dict(name=arm), fold=fold, seed=42)
        for arm in ("a", "b")
        for fold in range(2)
    ]
    manifest = dict(fingerprint="study", settings=dict(epochs=4), tasks=tasks)
    monkeypatch.setattr(compute, "load_manifest", lambda _: manifest)
    for i, (task, seconds) in enumerate(zip(tasks, (10, 14, 20, 24), strict=True)):
        output = tmp_path / "cv" / f"task-{i:04d}"
        write_json(
            output / "result.json",
            dict(
                manifest="study",
                task=task,
                seconds=seconds,
                selected_epoch=1,
                stopped_epoch=2,
                trainable_parameters=1,
                metrics=dict(nll=1, top1=0.1),
            ),
        )
        write_json(output / "runtime.json", dict(device="cpu", device_name="cpu"))
        write_json(output / "history.json", [dict(epoch=1, clipped_batch_fraction=0.5)])
        write_json(
            output / "timing-attempts" / "run.json",
            dict(
                phases=[
                    dict(name=name, epoch=epoch, seconds=cost, status="complete")
                    for epoch in (1, 2)
                    for name, cost in (("training", 1), ("validation", 0.5), ("checkpoint", 0.5))
                ]
            ),
        )
        phases: list[dict] = [dict(name="fit", seconds=seconds)]
        if i == 0:
            phases += [
                dict(name="profile", seconds=4),
                dict(name="estimate", seconds=2),
                dict(name="trainability_probe", seconds=3),
                dict(name="nlp_trainability_probe", seconds=5),
            ]
        write_json(
            tmp_path / "job-timings" / f"{i}.json",
            dict(
                stage="pilot" if i == 0 else "fit",
                task_id=i,
                status="complete",
                reused_result=False,
                job=f"9115_{i}",
                seconds=seconds + sum(p["seconds"] for p in phases[1:]) + 4,
                phases=phases,
            ),
        )
    write_json(
        tmp_path / "job-timings" / "failed.json",
        dict(
            stage="fit",
            task_id=1,
            status="failed",
            reused_result=False,
            job="9116_1",
            seconds=2,
            phases=[],
        ),
    )
    accounting = tmp_path / "accounting.psv"
    accounting.write_text("JobID|ElapsedRaw\n9115_0|30\n9115_0.batch|30\n")
    data = compute.timing_report(tmp_path, accounting=accounting)
    assert data["complete"] == 4
    assert data["arms"][0]["seconds_p10_p50_p90"][1] == 18  # 16 and 20 seconds.
    assert data["arms"][1]["seconds_p10_p50_p90"][1] == 26
    assert data["scenarios"][0]["scheduled_hours"] == 26 / 3600
    assert data["scenarios"][2]["scheduled_hours"] == 30 / 3600
    assert (tmp_path / "timing-report" / "cv-fits.csv").exists()
    scaled = compute.timing_report(tmp_path, fits_per_arm=4, accounting=accounting)
    assert scaled["scenarios"][0]["cv_fits"] == 8
    assert scaled["scenarios"][0]["scheduled_hours"] == 44 / 3600
    with pytest.raises(ValueError, match="concurrency"):
        compute.timing_report(tmp_path, concurrency=5)


def test_schedule_accounts_for_array_chunks():
    assert compute.schedule_seconds([1.0] * 1530, 4) == 383
    with pytest.raises(ValueError, match="chunks"):
        compute.schedule_seconds([1.0] * 5000, 4)


def test_copied_report_handles_real_parameter_free_fits_and_preserves_source(small_study):
    original_root, original = small_study
    root = original_root / "controls"
    settings = dict(
        original["settings"],
        objectives=["causal"],
        arms=["mean", "qcse-identity", "qcse-frozen", "mlp"],
        seeds=[42],
        fixed_epochs=True,
    )
    manifest = benchmark.prepare(root, settings)
    controls = []
    trained = None
    for i, task in enumerate(manifest["tasks"]):
        output = root / "cv" / f"task-{i:04d}"
        result = benchmark.fit(root, task, output)
        if result["trainable_parameters"] == 0:
            controls.append(i)
            assert result["stopped_epoch"] == 0
            assert not (output / "history.json").exists()
        else:
            trained = output
            assert (output / "history.json").is_file()
    assert len(controls) == 4 and trained is not None
    assert {manifest["tasks"][i]["arm"]["name"] for i in controls} == {
        "qcse-identity",
        "qcse-frozen",
    }
    # Copy outside the source-hash inputs, as on DGX; retain strict manifest checks.
    reporter = original_root / "report_benchmark_timing.py"
    shutil.copyfile(compute.__file__, reporter)
    before = benchmark.code_hashes()
    subprocess.run([sys.executable, str(reporter), str(root), "--concurrency", "4"], check=True)
    assert benchmark.code_hashes() == before == manifest["code_sha256"]
    summary = json.loads((root / "timing-report/summary.json").read_text())
    assert summary["complete"] == summary["planned"] == 8
    assert summary["report_source_sha256"] == hashlib.sha256(reporter.read_bytes()).hexdigest()
    with (root / "timing-report/cv-fits.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for i in controls:
        assert rows[i]["clipped_batch_fraction"] == ""
        assert rows[i]["epoch_seconds"] == ""
        assert float(rows[i]["epoch_cap_seconds"]) == float(rows[i]["seconds"])
    # Missing history for a trainable fit remains an error, not a silent fallback.
    (trained / "history.json").unlink()
    with pytest.raises(FileNotFoundError, match="history.json"):
        compute.timing_report(root)
