import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qcse import benchmark
from qcse.benchmark_models import (
    ANSATZES,
    BertWordModel,
    PaperQCSE,
    VariationalCircuit,
)
from qcse.benchmark_report import report
from qcse.circuit import encoding_circuit
from qcse.data import Corpus, build_vocabulary


@pytest.mark.parametrize("kind", ANSATZES)
def test_generators_match_qiskit_state_and_gradients(kind):
    torch.manual_seed(13)
    model = VariationalCircuit(3, 2, kind, scale=0.4).double()
    initial = np.random.default_rng(7).normal(size=8) + 1j * np.random.default_rng(8).normal(size=8)
    initial /= np.linalg.norm(initial)
    state = torch.tensor(np.stack((initial.real, initial.imag), -1))[None]
    circuit = QuantumCircuit(3)
    circuit.initialize(initial.tolist())
    if kind != "identity":
        for weights in model.weights.detach().numpy():
            for q in range(3):
                if kind == "ry-cx":
                    circuit.ry(weights[q], q)
                else:
                    circuit.rx(weights[q], q)
                circuit.rz(weights[3 + q], q)
            for index, (control, target) in enumerate(model.edges):
                if kind == "ry-cx":
                    circuit.cx(control, target)
                elif kind.startswith("crz"):
                    circuit.crz(weights[6 + index], control, target)
                elif kind == "zz-chain":
                    circuit.rzz(weights[6 + index], control, target)
                elif kind == "xx-chain":
                    circuit.rxx(weights[6 + index], control, target)
    actual = model(state)
    np.testing.assert_allclose(
        actual.detach()[0, :, 0] + 1j * actual.detach()[0, :, 1],
        Statevector.from_instruction(circuit).data,
        atol=1e-12,
    )
    if kind != "identity":
        model.register.features(actual).square().sum().backward()
        assert model.weights.grad is not None
        assert torch.isfinite(model.weights.grad).all()
        assert model.weights.grad.norm() > 0


def test_paper_encoder_marginals_and_dead_final_diagonal():
    torch.manual_seed(1)
    model = PaperQCSE(7, dict(layers=2)).double()
    contexts = torch.tensor([[7, 7, 1, 2], [3, 4, 5, 6]])
    probabilities = model.probabilities(contexts)
    for i, context in enumerate(contexts):
        matrix = benchmark.np.asarray(context[context != 7])
        from qcse.context import context_matrix, encoding_angles

        circuit = encoding_circuit(encoding_angles(context_matrix(matrix, 7), 3))
        for weights in model.circuit.weights.detach().numpy():
            for q in range(3):
                circuit.rx(weights[q], q)
                circuit.rz(weights[3 + q], q)
            for q in range(2):
                circuit.crz(weights[6 + q], q, q + 1)
        reference = Statevector.from_instruction(circuit)
        expected = [reference.probabilities([q])[1] for q in range(3)]
        np.testing.assert_allclose(probabilities[i].detach(), expected, atol=1e-12)
    logits = model(contexts)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits.softmax(-1).sum(-1), torch.ones(2, dtype=torch.double))
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2])).backward()
    grad = model.circuit.weights.grad
    assert grad is not None
    torch.testing.assert_close(grad[-1, 3:], torch.zeros_like(grad[-1, 3:]), atol=1e-12, rtol=0)
    assert grad[0].norm() > 0


def test_bert_mask_target_absent_and_same_word_output():
    model = BertWordModel(8, 4, "cbow", dict(width=16, layers=1))
    captured = {}

    def before(_module, args, kwargs):
        captured["inputs"] = args[0]
        captured["mask"] = kwargs["attention_mask"]

    hook = model.bert.register_forward_pre_hook(before, with_kwargs=True)
    logits = model(torch.tensor([[8, 2, 3, 8]]))
    hook.remove()
    assert captured["inputs"].tolist() == [[8, 2, 9, 3, 8]]
    assert captured["mask"].tolist() == [[False, True, True, True, False]]
    assert logits.shape == (1, 8)
    assert (
        model.bert.cls.predictions.decoder.weight
        is model.bert.bert.embeddings.word_embeddings.weight
    )


@pytest.fixture
def small_study(tmp_path, monkeypatch):
    sentences: list[list[str]] = [
        [str(word) for word in f"the animal {name} likes the food".split()]
        for name in (
            "cat",
            "dog",
            "bird",
            "bear",
            "horse",
            "rabbit",
            "fox",
            "wolf",
            "lion",
            "tiger",
            "duck",
            "deer",
            "mouse",
            "sheep",
            "goat",
            "cow",
        )
    ]
    corpus = Corpus(
        sentences, build_vocabulary(sentences), [["toy"]] * len(sentences), dict(toy=True)
    )
    monkeypatch.setattr(benchmark, "load_corpus", lambda **_kwargs: corpus)
    settings = dict(
        objectives=["causal", "cbow"],
        arms=["qcse-crz-chain", "hybrid-identity", "bert-16", "mlp"],
        max_sentences=None,
        window=4,
        folds=2,
        seeds=[42, 43],
        split_seed=2026,
        epochs=2,
        samples_per_epoch=9,
        batch_size=4,
        microbatch_size=2,
        eval_examples=6,
        final_eval_sentences=0,
        patience=2,
        min_epochs=1,
        threads=1,
    )
    root = tmp_path / "study"
    root.mkdir()
    manifest = benchmark.prepare(root, settings)
    return root, manifest


def test_prepared_splits_are_grouped_paired_and_cbow_has_no_target(small_study):
    root, manifest = small_study
    group_sets = []
    for objective in ("causal", "cbow"):
        with np.load(root / f"{objective}.npz") as data:
            groups = data["groups"]
            test = set(groups[data["test"]])
            dev = set(groups[data["development"]])
            assert not test & dev
            group_sets.append((dev, test))
            validation_groups = []
            for fold in range(2):
                train = set(groups[data[f"train_{fold}"]])
                val = set(groups[data[f"val_{fold}"]])
                assert not train & val and not train & test and not val & test
                assert train | val == dev
                validation_groups.append(val)
            assert not validation_groups[0] & validation_groups[1]
            assert validation_groups[0] | validation_groups[1] == dev
            if objective == "cbow":
                sentences = json.loads((root / "sentences.json").read_text())
                vocabulary = json.loads((root / "vocabulary.json").read_text())
                for i, context in enumerate(data["contexts"]):
                    words = [vocabulary[j] for j in context if j != len(vocabulary)]
                    position, group = data["positions"][i], data["groups"][i]
                    expected = (
                        sentences[group][max(0, position - 2) : position]
                        + sentences[group][position + 1 : position + 3]
                    )
                    assert words == expected
    assert group_sets[0] == group_sets[1]
    assert len(manifest["tasks"]) == 32
    with pytest.raises(ValueError, match="CV incomplete"):
        benchmark.selection(root)


def test_bert_interrupted_epoch_resume_restores_dropout_adam_and_sampler(small_study, monkeypatch):
    root, manifest = small_study
    task = next(t for t in manifest["tasks"] if t["arm"]["model"] == "bert")
    full, interrupted = root / "full", root / "interrupted"
    benchmark.fit(root, task, full)
    original = benchmark.training_loss
    calls = 0

    def stop(model, contexts, targets):
        nonlocal calls
        calls += 1
        if calls == 7:  # After the first complete 9-draw epoch, interrupt epoch two.
            raise KeyboardInterrupt
        return original(model, contexts, targets)

    monkeypatch.setattr(benchmark, "training_loss", stop)
    with pytest.raises(KeyboardInterrupt):
        benchmark.fit(root, task, interrupted)
    monkeypatch.setattr(benchmark, "training_loss", original)
    actual = benchmark.fit(root, task, interrupted, resume=True)
    expected = json.loads((full / "result.json").read_text())
    assert actual["metrics"] == expected["metrics"]
    assert actual["optimizer_updates"] == expected["optimizer_updates"] == 6
    assert len(list((interrupted / "runtime-attempts").glob("*.json"))) == 2
    assert actual["seconds"] > 0
    a = torch.load(full / "checkpoint.pt", weights_only=False)
    b = torch.load(interrupted / "checkpoint.pt", weights_only=False)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    for key in a["optimizer"]["state"]:
        for field in a["optimizer"]["state"][key]:
            torch.testing.assert_close(
                a["optimizer"]["state"][key][field],
                b["optimizer"]["state"][key][field],
                rtol=0,
                atol=0,
            )


def test_sentence_bootstrap_preserves_pairing_and_dependency():
    groups = np.array([0, 0, 0, 1])
    left = np.array([[[1.0, 1, 1], [1.0, 1, 1], [1.0, 1, 1], [0.0, 0, 0]]])
    result = benchmark.paired_bootstrap(groups, left, np.zeros_like(left), draws=500)
    assert result["difference"] == [0.75, 0.75, 0.75]
    assert result["low"] == [0.0, 0.0, 0.0]
    assert result["high"] == [1.0, 1.0, 1.0]
    identity = benchmark.paired_bootstrap(groups, left, left, draws=30)
    assert identity["low"] == identity["high"] == [0.0, 0.0, 0.0]
    bad = left.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid paired"):
        benchmark.paired_bootstrap(groups, bad, left)


def test_complete_shared_cv_selection_refits_report_and_artifact_guard(small_study):
    root, manifest = small_study
    for index, task in enumerate(manifest["tasks"]):
        benchmark.fit(root, task, root / "cv" / f"task-{index:04d}")
    circuit_record = json.loads((root / "cv" / "task-0000" / "history.json").read_text())[1]
    assert circuit_record["training_loss"] > 0
    assert circuit_record["circuit_gradients"]["optimizer_batches"] == 3
    assert circuit_record["circuit_gradients"]["structurally_dead_parameters"] > 0
    selected = benchmark.selection(root)
    assert benchmark.selection(root) == selected
    assert len(selected["refits"]) == 16
    for index, task in enumerate(selected["refits"]):
        benchmark.fit(root, task, root / "refit" / f"task-{index:04d}", refit_epochs=task["epochs"])
    report(root, draws=20)
    results = json.loads((root / "report" / "results.json").read_text())
    assert len(results["summaries"]) == 8
    assert all(r["draws"] == 20 for r in results["paired_intervals"])
    # Validation selection cannot consult held-out result files.
    refit_result = root / "refit" / "task-0000" / "result.json"
    changed = json.loads(refit_result.read_text())
    changed["metrics"]["top1"] = 1.0
    refit_result.write_text(json.dumps(changed))
    assert benchmark.selection(root) == selected
    with pytest.raises(ValueError, match="Aggregate metrics"):
        report(root, draws=20)
    (root / "vocabulary.json").write_text("[]")
    with pytest.raises(ValueError, match="Prepared artifact changed"):
        benchmark.load_manifest(root)


@pytest.mark.parametrize("kind", [k for k in ANSATZES if k != "identity"])
def test_gate_gradients_match_independent_finite_differences(kind):
    circuit = VariationalCircuit(2, 2, kind, scale=0.3).double()
    state = torch.randn(2, 4, 2, dtype=torch.double)
    state /= state.square().sum((1, 2), keepdim=True).sqrt()
    weights = circuit.weights.detach().clone().requires_grad_(True)

    def action(value):
        return torch.func.functional_call(circuit, {"weights": value}, (state,))

    assert torch.autograd.gradcheck(action, (weights,), eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "arm", [{"model": "qcse"}, {"model": "hybrid"}, {"model": "mlp"}, {"model": "bert"}]
)
def test_microbatch_gradient_matches_full_optimizer_batch(arm):
    from qcse.benchmark_models import create_benchmark_model, training_loss

    torch.manual_seed(618)
    full = create_benchmark_model(16, 4, "causal", arm).double().eval()
    split = create_benchmark_model(16, 4, "causal", arm).double().eval()
    split.load_state_dict(full.state_dict())
    contexts = torch.tensor([[16, 16, 1, 2], [3, 4, 5, 6], [16, 7, 8, 9]])
    targets = torch.tensor([2, 7, 10])
    training_loss(full, contexts, targets).backward()
    for start in (0, 2):
        inputs, labels = contexts[start : start + 2], targets[start : start + 2]
        (training_loss(split, inputs, labels) * len(inputs) / 3).backward()
    for a, b in zip(full.parameters(), split.parameters(), strict=True):
        if a.grad is not None:
            assert b.grad is not None
            torch.testing.assert_close(a.grad, b.grad, atol=1e-10, rtol=1e-8)


def test_cv_saved_predictions_and_budget_ledger_are_validated(small_study):
    root, manifest = small_study
    task = manifest["tasks"][0]
    output = root / "cv" / "task-0000"
    benchmark.fit(root, task, output)
    assert benchmark.validate_result(root, manifest, task, output, True)["validation"]
    result_path = output / "result.json"
    row = json.loads(result_path.read_text())
    row["optimizer_updates"] += 1
    result_path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="budget/update"):
        benchmark.validate_result(root, manifest, task, output, True)


def test_trainability_manifest_new_arms_survive_selection_and_refit(small_study):
    root, original = small_study
    new_root = root / "trainability"
    settings: dict = dict(
        original["settings"],
        trainability=True,
        seeds=[42],
        epochs=1,
        samples_per_epoch=3,
        arms=["hybrid-q2-depth2-init1", "hybrid-q8-depth8-init0.05"],
    )
    manifest = benchmark.prepare(new_root, settings)
    assert len(manifest["tasks"]) == 8
    for index, task in enumerate(manifest["tasks"]):
        benchmark.fit(new_root, task, new_root / "cv" / f"task-{index:04d}")
    selected = benchmark.selection(new_root)
    assert len(selected["candidates"]) == 4 and len(selected["refits"]) == 2
    assert {r["arm"]["name"] for r in selected["candidates"]} == set(settings["arms"])
    for index, task in enumerate(selected["refits"]):
        benchmark.fit(
            new_root, task, new_root / "refit" / f"task-{index:04d}", refit_epochs=task["epochs"]
        )
    report(new_root, draws=20)


@pytest.mark.parametrize("tasks,chunk_size", [(1230, 615), (1530, 765)])
def test_slurm_chunk_offsets_and_combined_concurrency(tasks, chunk_size):
    path = Path(__file__).resolve().parents[1] / "scripts" / "submit_robust_benchmark.py"
    spec = importlib.util.spec_from_file_location("submit_robust", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commands = module.array_commands(Path("/study"), "fit", tasks, 4, 1000, [])
    assert len(commands) == 2
    assert f"--array=0-{chunk_size - 1}%2" in commands[0]
    assert f"--array=0-{chunk_size - 1}%2" in commands[1]
    assert any(option.endswith(f"BENCHMARK_OFFSET={chunk_size}") for option in commands[1])
    refits = module.array_commands(Path("/study"), "refit", 33, 4, 1000, [])
    assert "--array=0-32%4" in refits[0]
    assert not any("--dependency" in option for command in commands for option in command)
    with pytest.raises(ValueError, match="Concurrency"):
        module.array_commands(Path("/study"), "fit", 1230, 1, 1000, [])
    with pytest.raises(ValueError, match="between 1 and 4"):
        module.array_commands(Path("/study"), "fit", tasks, 5, 1000, [])


def test_slurm_submission_receipts_and_resource_override_guard(tmp_path, monkeypatch):
    import subprocess
    import sys

    source = Path(__file__).resolve().parents[1] / "scripts" / "submit_robust_benchmark.py"
    # The deployment-only fix can run outside the study's frozen scripts/ directory.
    path = tmp_path / "outputs" / "submit_robust_benchmark.py"
    path.parent.mkdir()
    path.write_bytes(source.read_bytes())
    spec = importlib.util.spec_from_file_location("submit_receipts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ROOT == tmp_path
    monkeypatch.setattr(
        module,
        "load_manifest",
        lambda _: dict(
            tasks=[{}] * 1230, fingerprint="study-hash", settings=dict(objectives=["causal"])
        ),
    )
    commands = []
    monkeypatch.setenv("SBATCH_GRES", "gpu:nvidia_a100_1g.10gb:8")

    def submitted(command, **kwargs):
        assert kwargs["cwd"] == tmp_path
        if command[0] == "squeue":
            assert "--me" in command and "--partition=prod10" in command
            assert "--format=%i|%T" in command
            # The DGX controller retains the completed pilot in squeue.
            return "9115_0|COMPLETED\n9114_0|FAILED\n9113_0|CANCELLED\n"
        assert "--gres=gpu:nvidia_a100_1g.10gb:1" in command
        assert "--partition=prod10" in command and "--nodes=1" in command
        assert not any(key.startswith("SBATCH_") for key in kwargs["env"])
        commands.append(command)
        return f"{900 + len(commands)};cluster\n"

    monkeypatch.setattr(module.subprocess, "check_output", submitted)
    monkeypatch.setattr(sys, "argv", ["submit", str(tmp_path), "--sbatch-option=--time=24:00:00"])
    module.main()
    receipts = [json.loads(p.read_text()) for p in (tmp_path / "submissions").glob("*.json")]
    assert len(commands) == len(receipts) == 2
    assert {r["job"] for r in receipts} == {"901;cluster", "902;cluster"}
    assert all(r["manifest"] == "study-hash" and r["command"] in commands for r in receipts)
    assert all("--time=24:00:00" in command for command in commands)
    import hashlib

    assert all(
        r["launcher_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest() for r in receipts
    )
    snapshots = list((tmp_path / "submissions").glob("launcher-*.py"))
    assert len(snapshots) == 2
    assert all(p.read_bytes() == path.read_bytes() for p in snapshots)
    # Slurm accepts abbreviated options; these must not bypass protocol controls.
    for option in ("--dep=42", "--gres=gpu:nvidia_a100_1g.10gb:2", "--partition=other"):
        monkeypatch.setattr(sys, "argv", ["submit", str(tmp_path), f"--sbatch-option={option}"])
        with pytest.raises(SystemExit) as raised:
            module.main()
        assert raised.value.code == 2
    assert len(commands) == 2
    assert all(any(option.endswith("%2") for option in command) for command in commands)
    monkeypatch.setattr(sys, "argv", ["submit", str(tmp_path), "--stage=pilot"])
    module.main()
    assert len(commands) == 3
    assert "--array=0-0%1" in commands[-1]
    assert any("BENCHMARK_STAGE=pilot" in option for option in commands[-1])
    pilot_receipts = list((tmp_path / "submissions").glob("pilot-*.json"))
    assert len(pilot_receipts) == 1
    assert json.loads(pilot_receipts[0].read_text())["stage"] == "pilot"
    manifest_loader = module.load_manifest
    monkeypatch.setattr(module, "load_manifest", lambda _: dict(settings=dict(objectives=["cbow"])))
    with pytest.raises(SystemExit):
        module.main()
    assert len(commands) == 3
    monkeypatch.setattr(module, "load_manifest", manifest_loader)
    monkeypatch.setattr(sys, "argv", ["submit", str(tmp_path), "--concurrency=5"])
    with pytest.raises(SystemExit):
        module.main()
    assert len(commands) == 3

    monkeypatch.setattr(sys, "argv", ["submit", str(tmp_path), "--stage=pilot"])
    for state in (
        "PENDING",
        "RUNNING",
        "COMPLETING",
        "CONFIGURING",
        "SUSPENDED",
        "REQUEUED",
        "PREEMPTED",
        "SIGNALING",
        "STAGE_OUT",
        "UNKNOWN",
    ):

        def occupied(command, state=state, **kwargs):
            assert command[0] == "squeue"
            return f"9115_0|COMPLETED\n123_[0-764]|{state}\n"

        monkeypatch.setattr(module.subprocess, "check_output", occupied)
        with pytest.raises(SystemExit) as raised:
            module.main()
        assert raised.value.code == 2
        assert len(commands) == 3

    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "9115_0\nCOMPLETED\n")
    with pytest.raises(SystemExit):
        module.main()
    assert len(commands) == 3

    def unavailable(command, **kwargs):
        assert command[0] == "squeue"
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module.subprocess, "check_output", unavailable)
    with pytest.raises(subprocess.CalledProcessError):
        module.main()
    assert len(commands) == 3

    def locked(*args):
        raise BlockingIOError

    monkeypatch.setattr(module.fcntl, "flock", locked)
    with pytest.raises(SystemExit) as raised:
        module.main()
    assert raised.value.code == 2
    assert len(commands) == 3


@pytest.mark.parametrize("failure", ["", "scripts/profile_robust_benchmark.py"])
def test_slurm_pilot_stops_on_failure_and_preserves_device_mask(tmp_path, failure):
    import os
    import subprocess

    root = Path(__file__).resolve().parents[1]
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(
        "#!/bin/bash\n"
        '[[ "$CUDA_VISIBLE_DEVICES" == MIG-test ]] || exit 9\n'
        'printf "%s\\n" "$*" >> "$PILOT_COMMAND_LOG"\n'
        '[[ "$1" != "$PILOT_FAILURE" ]]\n'
    )
    python.chmod(0o755)
    nvidia = python.parent / "nvidia-smi"
    nvidia.write_text("#!/bin/bash\nexit 0\n")
    nvidia.chmod(0o755)
    log = tmp_path / "commands.txt"
    result = subprocess.run(
        ["bash", str(root / "slurm-robust-prod10.sbatch")],
        env=dict(
            os.environ,
            PATH=f"{python.parent}:{os.environ['PATH']}",
            SLURM_SUBMIT_DIR=str(tmp_path),
            SLURM_ARRAY_TASK_ID="0",
            SLURM_CPUS_PER_TASK="4",
            BENCHMARK_STAGE="pilot",
            BENCHMARK_STUDY="outputs/study",
            CUDA_VISIBLE_DEVICES="MIG-test",
            PILOT_COMMAND_LOG=str(log),
            PILOT_FAILURE=failure,
        ),
        capture_output=True,
        text=True,
    )
    calls = log.read_text().splitlines()
    assert len(calls) == (3 if failure else 5)
    assert (result.returncode != 0) == bool(failure)
    if not failure:
        assert "--concurrency 4" in calls[-2]
        assert calls[-1].endswith("fit outputs/study --task-id 0 --device cuda --resume")
        assert "completed successfully" in result.stdout
    else:
        assert "completed successfully" not in result.stdout


@pytest.mark.parametrize("concurrency", [2, 4])
def test_compute_projection_uses_requested_slot_limit(tmp_path, monkeypatch, concurrency):
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "estimate_benchmark_compute.py"
    spec = importlib.util.spec_from_file_location("estimate_slots", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arm = dict(name="constant", family="qcse")
    manifest = dict(
        fingerprint="study",
        settings=dict(
            objectives=["causal"],
            folds=2,
            seeds=[42],
            samples_per_epoch=5000,
            batch_size=64,
            eval_examples=4,
        ),
        tasks=[dict(arm=arm, objective="causal", fold=fold) for fold in range(2)],
    )
    profile = dict(
        manifest="study",
        batch=64,
        device="cpu",
        measurements=[
            dict(
                arm=arm,
                trainable_parameters=0,
                training_batch_seconds=0,
                evaluation_batch_seconds=64,
            )
        ],
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    np.savez(
        tmp_path / "causal.npz",
        val_eval_0=np.arange(5),
        val_eval_1=np.arange(7),
        test_eval=np.arange(11),
    )
    output = tmp_path / "compute.json"
    arguments = ["estimate", str(tmp_path), str(tmp_path / "profile.json"), "--output", str(output)]
    if concurrency != 4:
        arguments.extend(["--concurrency", str(concurrency)])
    monkeypatch.setattr(sys, "argv", arguments)
    module.main()
    result = json.loads(output.read_text())
    assert result["concurrency"] == concurrency
    # Parameter-free fits: (4 + 5) + (4 + 7) evaluation seconds; refit: 11 seconds.
    for row in result["scenarios"]:
        assert row["total_reference_hours"] == pytest.approx(31 / 3600)
        assert row["ideal_slot_reference_days"] == pytest.approx(31 / concurrency / 86400)
