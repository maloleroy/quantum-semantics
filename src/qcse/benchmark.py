"""Shared grouped benchmark protocol, checkpoints and sentence bootstrap."""

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch.nn import functional as F

from .benchmark_models import ANSATZES, create_benchmark_model, training_loss
from .benchmark_timing import FitTimer
from .context import METHODS
from .data import load_corpus, make_examples, split_examples, validation_folds
from .evaluation import evaluation_sample
from .outputs import atomic_path, save_npz, write_json
from .trainability import circuit_gradient_summary


def configurations(trainability=False):
    """A compact factorial circuit study plus equal LR grids for reference models."""
    arms = []
    for model in ("qcse", "hybrid"):
        for ansatz in ANSATZES:
            arms.append(dict(name=f"{model}-{ansatz}", model=model, ansatz=ansatz, layers=2))
        arms.append(dict(name=f"{model}-frozen", model=model, frozen=True, layers=2))
        for depth in (4, 8):
            arms.append(dict(name=f"{model}-depth-{depth}", model=model, layers=depth))
    for method in METHODS[1:]:
        arms.append(dict(name=f"qcse-context-{method}", model="qcse", context=method, layers=2))
    arms.extend(
        [
            dict(name="qcse-permuted-ids", model="qcse", permuted_ids=True, layers=2),
            dict(name="qcse-bit-bce", model="qcse", loss="bce", layers=2),
            dict(name="mean", model="mean"),
            dict(name="mlp", model="mlp"),
            dict(name="bert-16", model="bert", width=16, layers=2),
            dict(name="bert-32", model="bert", width=32, layers=2),
            dict(name="bert-16-depth-4", model="bert", width=16, layers=4),
        ]
    )
    # Same three-point LR search for each principal reference architecture.
    for model in ("qcse", "hybrid", "mlp", "bert"):
        for lr in (0.0003, 0.01):
            arms.append(dict(name=f"{model}-lr-{lr:g}", model=model, lr=lr, layers=2))
    for encoding in ("qcse", "classical"):
        for circuit in (True, False):
            arms.append(
                dict(
                    name=f"attention-{encoding}-{'circuit' if circuit else 'identity'}",
                    model="attention",
                    encoding=encoding,
                    circuit=circuit,
                    layers=2,
                )
            )
    if trainability:
        for lr in (0.0003, 0.01):
            arms.append(
                dict(name=f"hybrid-identity-lr-{lr:g}", model="hybrid", ansatz="identity", lr=lr)
            )
        for qubits in (2, 4, 8):
            if qubits != 4:
                arms.append(
                    dict(
                        name=f"hybrid-q{qubits}-identity",
                        model="hybrid",
                        qubits=qubits,
                        ansatz="identity",
                    )
                )
            for depth in (2, 8):
                for scale in (0.05, 1.0):
                    if qubits == 4 and scale == 0.05:
                        continue  # Existing hybrid chain/depth arms are these references.
                    arms.append(
                        dict(
                            name=f"hybrid-q{qubits}-depth{depth}-init{scale:g}",
                            model="hybrid",
                            qubits=qubits,
                            layers=depth,
                            init_scale=scale,
                        )
                    )
        # Follow-up: equal LR budgets at the sizes favored by the discovery screen.
        arms.extend(
            [
                dict(name="mlp-32", model="mlp", width=32),
                dict(name="hybrid-q8-local", model="hybrid", qubits=8, ansatz="local"),
                dict(name="hybrid-q8-xx", model="hybrid", qubits=8, ansatz="xx-chain"),
                dict(name="hybrid-q8-zz", model="hybrid", qubits=8, ansatz="zz-chain"),
                dict(name="hybrid-q8-frozen", model="hybrid", qubits=8, frozen=True),
                dict(name="mean-init-0.1", model="mean", embedding_std=0.1),
            ]
        )
        for lr in (0.0003, 0.01):
            arms.extend(
                [
                    dict(name=f"bert-32-lr-{lr:g}", model="bert", width=32, layers=2, lr=lr),
                    dict(name=f"mlp-32-lr-{lr:g}", model="mlp", width=32, lr=lr),
                    dict(name=f"hybrid-q8-lr-{lr:g}", model="hybrid", qubits=8, lr=lr),
                    dict(
                        name=f"hybrid-q8-identity-lr-{lr:g}",
                        model="hybrid",
                        qubits=8,
                        ansatz="identity",
                        lr=lr,
                    ),
                ]
            )
    for arm in arms:
        arm.setdefault("lr", 0.003)
        arm["family"] = arm["model"]
    return arms


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def repository_state():
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    return dict(commit=git("rev-parse", "HEAD"), dirty=git("status", "--porcelain"))


def code_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "src" / "qcse").glob("*.py")) + [
        root / "scripts" / "robust_benchmark.py",
        root / "scripts" / "check_benchmark_backend.py",
        root / "scripts" / "submit_robust_benchmark.py",
        root / "scripts" / "run_robust_benchmark_job.py",
        root / "scripts" / "profile_robust_benchmark.py",
        root / "scripts" / "estimate_benchmark_compute.py",
        root / "scripts" / "report_benchmark_timing.py",
        root / "scripts" / "probe_barren_plateaus.py",
        root / "scripts" / "probe_nlp_trainability.py",
        root / "slurm-robust-prod10.sbatch",
        root / "uv.lock",
    ]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def save_code_snapshot(output, hashes):
    repository = Path(__file__).resolve().parents[2]
    for name, checksum in hashes.items():
        data = (repository / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError("Code changed while recording the experiment")
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


def sentence_sample(ids, groups, limit, seed):
    """Bound final inference by complete sentence groups, preserving dependence."""
    unique = np.unique(groups[ids])
    selected = (
        unique
        if limit == 0 or len(unique) <= limit
        else np.sort(np.random.default_rng(seed).choice(unique, limit, replace=False))
    )
    return ids[np.isin(groups[ids], selected)]


def prepare(output, settings) -> dict:
    corpus = load_corpus(
        datasets=settings.get("datasets", ["phrases", "cleaned"]),
        max_sentences=settings["max_sentences"],
        seed=settings["split_seed"],
    )
    write_json(output / "vocabulary.json", corpus.vocabulary)
    write_json(output / "corpus.json", corpus.summary)
    # Save words once rather than copying the entire corpus into every checkpoint.
    with atomic_path(output / "sentences.json") as path:
        path.write_text(json.dumps(corpus.sentences), encoding="utf-8")
    task_list = []
    for objective in settings["objectives"]:
        examples = make_examples(corpus.sentences, corpus.vocabulary, settings["window"], objective)
        groups = np.array([e.sentence for e in examples], dtype=np.int64)
        contexts = np.full((len(examples), settings["window"]), len(corpus.vocabulary), np.int64)
        for index, example in enumerate(examples):
            if objective == "causal":
                contexts[index, -len(example.context) :] = example.context
            else:
                radius = settings["window"] // 2
                left = min(radius, example.position)
                right = len(example.context) - left
                contexts[index, radius - left : radius] = example.context[:left]
                contexts[index, radius : radius + right] = example.context[left:]
        dev, test = split_examples(examples, corpus.sentences, settings["split_seed"])
        arrays = dict(
            contexts=contexts,
            targets=np.array([e.target for e in examples]),
            groups=groups,
            positions=np.array([e.position for e in examples]),
            development=dev,
            test=test,
            test_eval=sentence_sample(test, groups, settings["final_eval_sentences"], 8128),
        )
        for fold, (train, val) in enumerate(
            validation_folds(
                examples,
                corpus.sentences,
                dev,
                settings["folds"],
                settings["split_seed"] + 1,
            )
        ):
            arrays[f"train_{fold}"] = train
            arrays[f"val_{fold}"] = val
            arrays[f"val_eval_{fold}"] = sentence_sample(
                val,
                groups,
                settings["final_eval_sentences"],
                9000 + fold,
            )
        save_npz(output / f"{objective}.npz", **arrays)
        for arm in configurations(settings.get("trainability", False)):
            if settings.get("arms") and arm["name"] not in settings["arms"]:
                continue
            if arm["model"] == "attention" and objective != "causal":
                continue
            for seed in settings["seeds"]:
                for fold in range(settings["folds"]):
                    task_list.append(dict(objective=objective, arm=arm, seed=seed, fold=fold))
    manifest: dict = dict(
        schema=1,
        settings=settings,
        tasks=task_list,
        vocabulary_size=len(corpus.vocabulary),
        corpus=corpus.summary,
        provenance=repository_state(),
        lock_sha256=hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        code_sha256=code_hashes(),
        artifact_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.iterdir())
            if p.suffix in (".json", ".npz") and p.name != "run_info.json"
        },
    )
    if not task_list:
        raise ValueError("No configurations apply to the selected objectives")
    for objective in settings["objectives"]:
        available = {t["arm"]["name"] for t in task_list if t["objective"] == objective}
        if not set(settings.get("refit_controls", [])) <= available:
            raise ValueError(f"A refit control is unavailable for {objective}")
    manifest["fingerprint"] = fingerprint(manifest)
    save_code_snapshot(output, manifest["code_sha256"])
    write_json(output / "manifest.json", manifest)
    return manifest


def load_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text())
    digest = manifest["fingerprint"]
    if fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}) != digest:
        raise ValueError("Manifest was modified after preparation")
    if code_hashes() != manifest["code_sha256"]:
        raise ValueError(
            "Code or uv.lock differs from preparation; use the original checkout or prepare again"
        )
    for name, checksum in manifest["artifact_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Prepared artifact changed: {name}")
    return manifest


def runtime(device):
    info: dict = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        platform=platform.platform(),
        device=device,
        device_name=torch.cuda.get_device_name() if device == "cuda" else device,
        cuda=torch.version.cuda,
        scheduler={
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        repository=repository_state(),
        threads=torch.get_num_threads(),
    )
    if device == "cuda":
        properties = torch.cuda.get_device_properties(device)
        info["cuda_device"] = dict(
            name=properties.name,
            total_memory_bytes=properties.total_memory,
            multiprocessors=properties.multi_processor_count,
            compute_capability=[properties.major, properties.minor],
            uuid=str(getattr(properties, "uuid", "unavailable")),
        )
    return info


@torch.no_grad()
def evaluate(model, contexts, targets, ids, microbatch, predictions=None):
    model.eval()
    rows = []
    for start in range(0, len(ids), microbatch):
        batch = ids[start : start + microbatch]
        scores = model(contexts[batch].to(next(model.parameters()).device))
        target = targets[batch].to(scores.device)
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite evaluation logits")
        ranks = scores.topk(min(5, scores.shape[-1]), -1).indices
        if predictions is not None:
            predictions.append(ranks.cpu().numpy())
        nll = F.cross_entropy(scores, target, reduction="none")
        rows.append(
            torch.stack(
                (
                    nll,
                    (ranks[:, 0] == target).float(),
                    (ranks == target[:, None]).any(-1).float(),
                ),
                -1,
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(rows)


def metrics(rows):
    return dict(
        nll=float(rows[:, 0].mean()),
        top1=float(rows[:, 1].mean()),
        top5=float(rows[:, 2].mean()),
        examples=len(rows),
    )


def validate_result(root, manifest, task, output, validation, data=None):
    """Recompute saved metrics against prepared IDs and decoded top-k predictions."""
    result = json.loads((output / "result.json").read_text())
    if (
        result["manifest"] != manifest["fingerprint"]
        or result["task"] != task
        or result["validation"] != validation
    ):
        raise ValueError(f"Result metadata mismatch: {output}")
    if data is None:
        with np.load(root / f"{task['objective']}.npz") as archive:
            data = {k: archive[k] for k in archive.files if k != "contexts"}
    fold = task.get("fold", 0)
    expected_ids = data[f"val_eval_{fold}"] if validation else data["test_eval"]
    train_ids = data[f"train_{fold}"] if validation else data["development"]
    with np.load(output / "predictions.npz") as archive:
        saved = {k: archive[k] for k in archive.files}
    for key, expected in (
        ("ids", expected_ids),
        ("targets", data["targets"][expected_ids]),
        ("groups", data["groups"][expected_ids]),
        ("positions", data["positions"][expected_ids]),
    ):
        if not np.array_equal(saved[key], expected):
            raise ValueError(f"Prediction {key} differs from the prepared split: {output}")
    values, top_ids = saved["metrics"], saved["top_ids"]
    vocabulary_size = manifest["vocabulary_size"]
    if (
        values.shape != (len(expected_ids), 3)
        or not np.isfinite(values).all()
        or (values[:, 0] < -1e-7).any()
        or top_ids.shape != (len(expected_ids), min(5, vocabulary_size))
        or ((top_ids < 0) | (top_ids >= vocabulary_size)).any()
    ):
        raise ValueError(f"Invalid prediction metrics/ranks: {output}")
    y = saved["targets"]
    for column, expected in ((1, top_ids[:, 0] == y), (2, (top_ids == y[:, None]).any(-1))):
        if not np.array_equal(values[:, column], expected):
            raise ValueError(f"Saved retrieval scores disagree with predicted IDs: {output}")
    counts = np.bincount(data["targets"][train_ids], minlength=vocabulary_size) + 1
    frequencies = counts / counts.sum()
    ranked = np.argsort(-frequencies, kind="stable")[:5]
    baseline = np.column_stack((-np.log(frequencies[y]), y == ranked[0], np.isin(y, ranked)))
    if not np.allclose(saved["frequency"], baseline, atol=1e-12, rtol=1e-12):
        raise ValueError(f"Frequency baseline is not computed from training IDs: {output}")
    if (np.diff(np.sort(top_ids, axis=1), axis=1) == 0).any():
        raise ValueError(f"Duplicate IDs in saved prediction ranks: {output}")
    for label, metric_values in (("metrics", values), ("frequency", baseline)):
        expected = metrics(metric_values)
        if any(
            not np.isclose(result[label][k], expected[k], atol=1e-7, rtol=1e-7) for k in expected
        ):
            raise ValueError(f"Aggregate {label} differs from saved predictions: {output}")
    budget = manifest["settings"]["epochs"] if validation else task["epochs"]
    epoch = result["stopped_epoch"]
    steps = math.ceil(
        manifest["settings"]["samples_per_epoch"] / manifest["settings"]["batch_size"]
    )
    if (
        not 0 <= result["selected_epoch"] <= epoch <= budget
        or result["optimizer_updates"] != epoch * steps
        or result["training_draws"] != epoch * manifest["settings"]["samples_per_epoch"]
        or (
            manifest["settings"].get("fixed_epochs", False)
            and result["trainable_parameters"]
            and (epoch != budget or result["selected_epoch"] != budget)
        )
    ):
        raise ValueError(f"Training budget/update ledger is inconsistent: {output}")
    return result


def fit(root, task, output, device="cpu", resume=False, refit_epochs=None):
    attempt_begun = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        manifest = load_manifest(root)
        return validate_result(root, manifest, task, output, refit_epochs is None)
    path = output / "checkpoint.pt"
    if path.exists() and not resume:
        raise ValueError("Unfinished checkpoint exists; use --resume")
    attempt = uuid4().hex
    timer = FitTimer(output / "timing-attempts" / f"{attempt}.json", device, attempt)
    with timer.phase("manifest", synchronize=False):
        manifest = load_manifest(root)
        settings = manifest["settings"]
    if device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(task["seed"])
    np.random.seed(task["seed"])
    if device == "cuda":
        torch.cuda.manual_seed_all(task["seed"])
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(settings["threads"])
    with timer.phase("setup"):
        model = create_benchmark_model(
            manifest["vocabulary_size"],
            settings["window"],
            task["objective"],
            task["arm"],
        ).to(device)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=task["arm"]["lr"]) if parameters else None
        with np.load(root / f"{task['objective']}.npz") as arrays:
            contexts = torch.from_numpy(arrays["contexts"].copy())
            targets = torch.from_numpy(arrays["targets"].copy())
            groups, positions = arrays["groups"].copy(), arrays["positions"].copy()
            if refit_epochs is None:
                fold = task["fold"]
                train, val = arrays[f"train_{fold}"].copy(), arrays[f"val_{fold}"].copy()
                final_ids = arrays[f"val_eval_{fold}"].copy()
                eligible_eval_examples = len(val)
            else:
                train, val = arrays["development"].copy(), None
                final_ids = arrays["test_eval"].copy()
                eligible_eval_examples = len(arrays["test"])
        rng = np.random.default_rng(task["seed"] + task.get("fold", 0) * 100)
        monitor = (
            None
            if val is None
            else evaluation_sample(
                val,
                settings["eval_examples"],
                settings["split_seed"],
                7000 + task["fold"],
            )
        )
    epochs = settings["epochs"] if refit_epochs is None else refit_epochs
    fixed_epochs = settings.get("fixed_epochs", False)
    history, best_state, best_epoch, best_key, stale = [], None, 0, None, 0
    completed_epochs, updates, elapsed = 0, 0, 0.0
    info = runtime(device)
    info.update(attempt=attempt, started_at=datetime.now(UTC).isoformat())
    write_json(output / "runtime-attempts" / f"{info['attempt']}.json", info)
    write_json(output / "runtime.json", info)
    if path.exists():
        with timer.phase("resume"):
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["manifest"] != manifest["fingerprint"] or saved["task"] != task:
                raise ValueError("Checkpoint differs from the saved manifest/task")
            model.load_state_dict(saved["model"])
            if optimizer is not None:
                optimizer.load_state_dict(saved["optimizer"])
            rng.bit_generator.state = saved["rng"]
            torch.set_rng_state(saved["torch_rng"])
            if device == "cuda" and saved["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(saved["cuda_rng"])
            if device == "mps" and saved.get("mps_rng") is not None:
                torch.mps.set_rng_state(saved["mps_rng"])
            history, best_state, best_epoch, best_key, stale = (
                saved[k] for k in ("history", "best_state", "best_epoch", "best_key", "stale")
            )
            completed_epochs, updates, elapsed = (saved[k] for k in ("epoch", "updates", "elapsed"))
    begun = time.monotonic()
    elapsed += begun - attempt_begun
    microbatch = settings["microbatch_size"]

    def save(epoch):
        with atomic_path(path) as temporary:
            torch.save(
                dict(
                    manifest=manifest["fingerprint"],
                    task=task,
                    epoch=epoch,
                    model=model.state_dict(),
                    optimizer=None if optimizer is None else optimizer.state_dict(),
                    rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else None,
                    mps_rng=torch.mps.get_rng_state() if device == "mps" else None,
                    history=history,
                    best_state=best_state,
                    best_epoch=best_epoch,
                    best_key=best_key,
                    stale=stale,
                    updates=updates,
                    elapsed=elapsed + time.monotonic() - begun,
                ),
                temporary,
            )
        write_json(output / "history.json", history)

    # Epoch 0 is recorded, but checkpoint selection considers trained epochs only.
    if not history and monitor is not None:
        with timer.phase("initial_validation", epoch=0):
            history.append(
                dict(
                    epoch=0,
                    validation=metrics(
                        evaluate(
                            model,
                            contexts,
                            targets,
                            monitor,
                            microbatch,
                        )
                    ),
                )
            )
    for epoch in range(completed_epochs + 1, epochs + 1):
        if (
            val is not None
            and not fixed_epochs
            and stale >= settings["patience"]
            and completed_epochs >= settings["min_epochs"]
        ):
            break
        if optimizer is None:
            # Parameter-free controls have no optimizer trajectory.
            best_epoch = 0
            break
        with timer.phase("training", epoch):
            order = rng.choice(train, settings["samples_per_epoch"], replace=True)
            model.train()
            circuit_gradients = []
            gradient_norms = []
            epoch_loss = torch.zeros((), device=device)
            for start in range(0, len(order), settings["batch_size"]):
                batch = order[start : start + settings["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0, len(batch), microbatch):
                    part = batch[offset : offset + microbatch]
                    loss = training_loss(model, contexts[part].to(device), targets[part].to(device))
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training loss")
                    epoch_loss += loss.detach() * len(part)
                    (loss * (len(part) / len(batch))).backward()
                diagnostic = circuit_gradient_summary(model)
                if diagnostic is not None:
                    circuit_gradients.append(diagnostic)
                norm = torch.nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
                gradient_norms.append(norm.detach())
                optimizer.step()
                updates += 1
        norm_values = torch.stack(gradient_norms)
        row: dict = dict(
            epoch=epoch,
            updates=updates,
            gradient_norm=float(norm),
            training_loss=float(epoch_loss / len(order)),
            gradient_norm_mean=float(norm_values.mean()),
            gradient_norm_max=float(norm_values.max()),
            clipped_batch_fraction=float((norm_values > 5).float().mean()),
        )
        if circuit_gradients:
            row["circuit_gradients"] = {
                key: np.asarray([record[key] for record in circuit_gradients]).mean(0).tolist()
                for key in ("layer_rms", "layer_max_abs", "layer_near_zero_fraction")
            }
            row["circuit_gradients"].update(
                active_parameters=circuit_gradients[0]["active_parameters"],
                structurally_dead_parameters=circuit_gradients[0]["structurally_dead_parameters"],
                optimizer_batches=len(circuit_gradients),
                near_zero_threshold=1e-10,
            )
        if monitor is not None:
            with timer.phase("validation", epoch):
                score = metrics(evaluate(model, contexts, targets, monitor, microbatch))
                row["validation"] = score
            with timer.phase("selection", epoch):
                key = (score["top1"], score["top5"], -score["nll"], -epoch)
                if fixed_epochs:
                    best_epoch = epoch
                elif best_key is None or key > best_key:
                    best_key, best_epoch = key, epoch
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                # NLL-only tie-breaks must not extend a flat retrieval plateau forever.
                retrieval = (score["top1"], score["top5"])
                previous = [r["validation"] for r in history if r["epoch"] > 0]
                stale = (
                    0
                    if not previous or retrieval > max((r["top1"], r["top5"]) for r in previous)
                    else stale + 1
                )
        history.append(row)
        completed_epochs = epoch
        with timer.phase("checkpoint", epoch):
            save(epoch)
        print(json.dumps(row), flush=True)
    if best_state is not None and val is not None:
        model.load_state_dict(best_state)
    if refit_epochs is not None:
        best_epoch = completed_epochs
    with timer.phase("final_evaluation", epoch=best_epoch):
        predictions = []
        final_rows = evaluate(model, contexts, targets, final_ids, microbatch, predictions)
        predicted_ids = np.concatenate(predictions)
    with timer.phase("artifacts", epoch=best_epoch):
        # Train-only frequency baseline, on exactly the same evaluated targets.
        counts = np.bincount(targets[train].numpy(), minlength=manifest["vocabulary_size"]) + 1
        frequencies = counts / counts.sum()
        ranked = np.argsort(-frequencies, kind="stable")[:5]
        y = targets[final_ids].numpy()
        baseline_rows = np.column_stack(
            (
                -np.log(frequencies[y]),
                y == ranked[0],
                np.isin(y, ranked),
            )
        )
        save_npz(
            output / "predictions.npz",
            ids=final_ids,
            groups=groups[final_ids],
            positions=positions[final_ids],
            targets=y,
            top_ids=predicted_ids,
            metrics=final_rows,
            frequency=baseline_rows,
        )
        vocabulary = json.loads((root / "vocabulary.json").read_text())
        write_json(
            output / "decoded_examples.json",
            [
                dict(
                    context=[
                        vocabulary[i] for i in contexts[example].tolist() if i < len(vocabulary)
                    ],
                    target=vocabulary[int(y[index])],
                    predictions=[vocabulary[i] for i in predicted_ids[index]],
                )
                for index, example in enumerate(final_ids[:8])
            ],
        )
        # Save a portable inference checkpoint separately from the continuation state.
        with atomic_path(output / "selected.pt") as temporary:
            torch.save(
                dict(
                    model={k: v.cpu() for k, v in model.state_dict().items()},
                    task=task,
                    manifest=manifest["fingerprint"],
                ),
                temporary,
            )
    result = dict(
        manifest=manifest["fingerprint"],
        task=task,
        validation=val is not None,
        metrics=metrics(final_rows),
        frequency=metrics(baseline_rows),
        selected_epoch=best_epoch,
        stopped_epoch=completed_epochs,
        optimizer_updates=updates,
        training_draws=completed_epochs * settings["samples_per_epoch"],
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in parameters),
        train_examples=len(train),
        eligible_eval_examples=eligible_eval_examples,
        evaluated_sentences=len(np.unique(groups[final_ids])),
        seconds=elapsed + time.monotonic() - begun,
        timing_attempt=attempt,
        cuda_peak_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if device == "cuda" else None,
    )
    write_json(output / "result.json", result)
    timer.finish()
    return result


def selection(root):
    manifest = load_manifest(root)
    rows = []
    prepared = {}
    for index, task in enumerate(manifest["tasks"]):
        path = root / "cv" / f"task-{index:04d}" / "result.json"
        if not path.exists():
            raise ValueError(f"CV incomplete: missing {path}; test selection is blocked")
        if task["objective"] not in prepared:
            with np.load(root / f"{task['objective']}.npz") as archive:
                prepared[task["objective"]] = {
                    k: archive[k] for k in archive.files if k != "contexts"
                }
        row = validate_result(root, manifest, task, path.parent, True, prepared[task["objective"]])
        rows.append(row)
    summaries = []
    winners = []
    for objective in manifest["settings"]["objectives"]:
        candidates = []
        for task in manifest["tasks"]:
            if task["objective"] == objective and task["arm"] not in candidates:
                candidates.append(task["arm"])
        for arm in candidates:
            fits = [
                r for r in rows if r["task"]["arm"] == arm and r["task"]["objective"] == objective
            ]
            values = np.array([[r["metrics"][k] for k in ("nll", "top1", "top5")] for r in fits])
            summaries.append(
                dict(
                    objective=objective,
                    arm=arm,
                    fits=len(fits),
                    mean=dict(zip(("nll", "top1", "top5"), values.mean(0).tolist(), strict=True)),
                    sd=dict(
                        zip(("nll", "top1", "top5"), values.std(0, ddof=1).tolist(), strict=True)
                    ),
                    refit_epochs=int(np.median([r["selected_epoch"] for r in fits])),
                )
            )
        for family in sorted({arm["family"] for arm in candidates}):
            ranked = [
                s for s in summaries if s["objective"] == objective and s["arm"]["family"] == family
            ]
            winner = sorted(
                ranked,
                key=lambda s: (
                    -s["mean"]["top1"],
                    -s["mean"]["top5"],
                    s["mean"]["nll"],
                    s["arm"]["name"],
                ),
            )[0]
            for seed in manifest["settings"]["seeds"]:
                winners.append(
                    dict(
                        objective=objective,
                        arm=winner["arm"],
                        seed=seed,
                        epochs=winner["refit_epochs"],
                    )
                )
        for name in manifest["settings"].get("refit_controls", []):
            control = next(
                s for s in summaries if s["objective"] == objective and s["arm"]["name"] == name
            )
            if any(t["objective"] == objective and t["arm"] == control["arm"] for t in winners):
                continue
            for seed in manifest["settings"]["seeds"]:
                winners.append(
                    dict(
                        objective=objective,
                        arm=control["arm"],
                        seed=seed,
                        epochs=control["refit_epochs"],
                    )
                )
    result = dict(manifest=manifest["fingerprint"], candidates=summaries, refits=winners)
    result["fingerprint"] = fingerprint(result)
    path = root / "selection.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("Existing selection differs; prepare a new study")
    write_json(path, result)
    return result


def load_selection(root):
    manifest = load_manifest(root)
    result = json.loads((root / "selection.json").read_text())
    if result["manifest"] != manifest["fingerprint"] or result["fingerprint"] != fingerprint(
        {k: v for k, v in result.items() if k != "fingerprint"}
    ):
        raise ValueError("Selection was modified or belongs to a different study")
    return result


def paired_bootstrap(groups, left, right, draws=2000, seed=42, confidence=0.95):
    """Paired cluster percentile interval, token-weighted, conditional on fitted models.

    Arrays are (seeds, examples, metrics). Every draw resamples seed pairs and
    sentence groups; sentence sizes are retained in the denominator.
    """
    if (
        draws < 1
        or not 0 < confidence < 1
        or left.shape != right.shape
        or left.ndim != 3
        or left.shape[1] != len(groups)
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("Invalid paired bootstrap arrays or budget")
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("Bootstrap needs at least two sentence groups")
    counts = np.bincount(inverse)
    delta = left - right
    sums = np.stack(
        [
            np.stack(
                [
                    np.bincount(inverse, weights=row[:, k], minlength=len(unique))
                    for k in range(left.shape[2])
                ],
                -1,
            )
            for row in delta
        ]
    )
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, left.shape[2]))
    for index in range(draws):
        group_ids = rng.integers(len(unique), size=len(unique))
        seeds = rng.integers(len(left), size=len(left))
        samples[index] = sums[seeds][:, group_ids].sum((0, 1)) / (
            counts[group_ids].sum() * len(seeds)
        )
    return dict(
        difference=delta.mean((0, 1)).tolist(),
        low=np.quantile(samples, (1 - confidence) / 2, axis=0).tolist(),
        high=np.quantile(samples, (1 + confidence) / 2, axis=0).tolist(),
        confidence=confidence,
        draws=draws,
        sentences=len(unique),
        seeds=len(left),
        metric_order=["nll", "top1", "top5"],
    )
