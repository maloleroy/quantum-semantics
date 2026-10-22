"""Audit exported CV diagnostics without training, test data or source-hash bypasses."""

import argparse
import csv
import hashlib
import json
import statistics as stats
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True).encode())


def expected_folds(root, manifest):
    """Reconstruct causal group membership and train-frequency scores from saved words.

    This checks expected counts/scores, not the omitted saved binary indices.
    It deliberately does not materialize the million overlapping token windows.
    """
    settings = manifest["settings"]
    assert settings["objectives"] == ["causal"] and settings["final_eval_sentences"] == 0
    sentences = json.loads((root / "sentences.json").read_text())
    vocabulary = json.loads((root / "vocabulary.json").read_text())
    lookup = {word: i for i, word in enumerate(vocabulary)}
    groups = sorted(tuple(words) for words in sentences)
    assert len(groups) == len(set(groups))
    test_count = round(len(groups) * 0.2)
    order = np.random.default_rng(settings["split_seed"]).permutation(len(groups))
    held_out = set(order[:test_count].tolist())
    development = [words for i, words in enumerate(groups) if i not in held_out]
    all_counts = Counter(word for words in development for word in words[1:])
    order = np.random.default_rng(settings["split_seed"] + 1).permutation(len(development))
    folds = []
    for f, ids in enumerate(np.array_split(order, settings["folds"])):
        val_counts = Counter(word for i in ids for word in development[i][1:])
        training = all_counts - val_counts
        counts = np.array([training[word] + 1 for word in vocabulary], dtype=float)
        probabilities = counts / counts.sum()
        ranks = np.argsort(-probabilities, kind="stable")[:5]
        top = {vocabulary[i] for i in ranks}
        total = val_counts.total()
        baseline = dict(
            nll=sum(
                -count * np.log(probabilities[lookup[word]]) for word, count in val_counts.items()
            )
            / total,
            top1=val_counts[vocabulary[ranks[0]]] / total,
            top5=sum(count for word, count in val_counts.items() if word in top) / total,
        )
        folds.append(
            dict(
                fold=f,
                train_groups=len(development) - len(ids),
                validation_groups=len(ids),
                test_groups=test_count,
                train_examples=training.total(),
                validation_examples=total,
                frequency=baseline,
            )
        )
    return folds


def audit(root, output, archive=None):
    def read(name):
        return json.loads((root / name).read_text())

    manifest, selection = read("manifest.json"), read("selection.json")
    for value in (manifest, selection):
        assert (
            fingerprint({k: v for k, v in value.items() if k != "fingerprint"})
            == value["fingerprint"]
        )
    assert selection["manifest"] == manifest["fingerprint"]
    commit = manifest["provenance"]["commit"]
    for name, checksum in manifest["code_sha256"].items():
        assert digest(subprocess.check_output(["git", "show", f"{commit}:{name}"])) == checksum
    included, omitted = [], []
    for name, checksum in manifest["artifact_sha256"].items():
        if (root / name).exists():
            assert digest((root / name).read_bytes()) == checksum
            included.append(name)
        else:
            omitted.append(name)
    settings = manifest["settings"]
    for source in manifest["corpus"]["sources"]:
        assert digest(Path(Path(source["path"]).name).read_bytes()) == source["sha256"]
    reconstructed = expected_folds(root, manifest)
    results, histories = [], []
    for i, task in enumerate(manifest["tasks"]):
        folder = root / "cv" / f"task-{i:04d}"
        result = read(folder.relative_to(root) / "result.json")
        assert result["manifest"] == manifest["fingerprint"] and result["task"] == task
        assert result["validation"]
        expected = reconstructed[task["fold"]]
        assert result["train_examples"] == expected["train_examples"]
        assert result["metrics"]["examples"] == expected["validation_examples"]
        assert result["eligible_eval_examples"] == expected["validation_examples"]
        assert result["evaluated_sentences"] == expected["validation_groups"]
        assert all(
            abs(result["frequency"][k] - expected["frequency"][k]) < 1e-12
            for k in ("nll", "top1", "top5")
        )
        expected_epochs = settings["epochs"] if result["trainable_parameters"] else 0
        assert result["stopped_epoch"] == result["selected_epoch"] == expected_epochs
        steps = -(-settings["samples_per_epoch"] // settings["batch_size"])
        assert result["optimizer_updates"] == expected_epochs * steps
        assert result["training_draws"] == expected_epochs * settings["samples_per_epoch"]
        history_path = folder / "history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        if expected_epochs:
            assert [v["epoch"] for v in history] == list(range(expected_epochs + 1))
            assert all(v["updates"] == v["epoch"] * steps for v in history[1:])
        else:
            assert not history
        timing = read(
            folder.relative_to(root) / "timing-attempts" / f"{result['timing_attempt']}.json"
        )
        assert timing["status"] == "complete"
        assert all(p["status"] == "complete" and p["seconds"] >= 0 for p in timing["phases"])
        runtime = read(folder.relative_to(root) / "runtime.json")
        assert runtime["device"] == "cuda" and runtime["repository"]["commit"] == commit
        assert runtime["threads"] == settings["threads"]
        results.append(result)
        histories.append(history)

    jobs = [json.loads(p.read_text()) for p in (root / "job-timings").glob("*.json")]
    assert all(j["status"] == "complete" for j in jobs)
    assert all(p["returncode"] == 0 for j in jobs for p in j["phases"])
    job_ids = {j["job"] for j in jobs}
    with (root / "slurm-accounting.psv").open() as stream:
        accounting = [r for r in csv.DictReader(stream, delimiter="|") if r["JobID"] in job_ids]
    assert {r["JobID"] for r in accounting} == job_ids
    assert len(accounting) == len(job_ids)
    assert all(r["State"] == "COMPLETED" and r["ExitCode"] == "0:0" for r in accounting)
    events = sorted([(r["Start"], 1) for r in accounting] + [(r["End"], -1) for r in accounting])
    active = peak = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    assert active == 0 and peak <= 4
    array_ids = {j["job"] for j in jobs if j["stage"] == "fit"}
    array = [r for r in accounting if r["JobID"] in array_ids]
    span = (
        max(datetime.fromisoformat(r["End"]) for r in array)
        - min(datetime.fromisoformat(r["Start"]) for r in array)
    ).total_seconds()
    elapsed = {r["JobID"]: float(r["ElapsedRaw"]) for r in accounting}
    with (root / "timing-report/cv-fits.csv").open() as stream:
        times = list(csv.DictReader(stream))
    assert len(times) == len(results)
    assert [int(row["task_id"]) for row in times] == list(range(len(results)))
    for row in times:
        i = int(row["task_id"])
        attempts = [j for j in jobs if j["task_id"] == i and not j["reused_result"]]
        expected = sum(
            elapsed[j["job"]]
            - sum(
                p["seconds"]
                for p in j["phases"]
                if p["name"]
                in ("profile", "estimate", "trainability_probe", "nlp_trainability_probe")
            )
            for j in attempts
        )
        assert abs(float(row["seconds"]) - expected) < 1e-7
        assert row["duration_source"] == "slurm"

    arms = []
    for candidate in selection["candidates"]:
        name = candidate["arm"]["name"]
        indexes = [i for i, v in enumerate(results) if v["task"]["arm"]["name"] == name]
        fits = [results[i] for i in indexes]
        for key in ("nll", "top1", "top5"):
            values = [v["metrics"][key] for v in fits]
            assert abs(candidate["mean"][key] - stats.mean(values)) < 1e-12
            assert abs(candidate["sd"][key] - stats.stdev(values)) < 1e-12
        records = [row for i in indexes for row in histories[i][1:]]
        gradients = [v["circuit_gradients"] for v in records if "circuit_gradients" in v]
        curves = [
            dict(
                epoch=e,
                **{
                    key: stats.mean(histories[i][e]["validation"][key] for i in indexes)
                    for key in ("nll", "top1", "top5")
                },
            )
            for e in (0, 25, 50, 100)
            if all(len(histories[i]) > e for i in indexes)
        ]
        arms.append(
            dict(
                arm=candidate["arm"],
                mean=candidate["mean"],
                sd=candidate["sd"],
                parameters=fits[0]["parameters"],
                trainable_parameters=fits[0]["trainable_parameters"],
                median_seconds=stats.median(float(times[i]["seconds"]) for i in indexes),
                max_reserved_mib=max(v["cuda_peak_reserved_bytes"] for v in fits) / 2**20,
                clip_fraction=stats.mean(v["clipped_batch_fraction"] for v in records)
                if records
                else None,
                gradient_rms_range=[
                    min(v for g in gradients for v in g["layer_rms"]),
                    max(v for g in gradients for v in g["layer_rms"]),
                ]
                if gradients
                else None,
                structurally_dead_parameters=gradients[0]["structurally_dead_parameters"]
                if gradients
                else None,
                curves=curves,
            )
        )
    pairs = []
    for left, right in (
        ("bert-32", "mlp"),
        ("hybrid-q8-depth2-init0.05", "hybrid-q8-identity"),
        ("hybrid-crz-chain", "hybrid-identity"),
        ("hybrid-depth-8", "hybrid-crz-chain"),
        ("qcse-depth-8", "qcse-crz-chain"),
        ("qcse-context-angular", "qcse-crz-chain"),
    ):
        keyed = {
            name: {v["task"]["fold"]: v for v in results if v["task"]["arm"]["name"] == name}
            for name in (left, right)
        }
        delta = [
            100 * (keyed[left][f]["metrics"]["top1"] - keyed[right][f]["metrics"]["top1"])
            for f in range(settings["folds"])
        ]
        pairs.append(
            dict(
                left=left,
                right=right,
                top1_difference_pp=stats.mean(delta),
                fold_differences_pp=delta,
            )
        )
    baseline = {
        key: stats.mean(v["frequency"][key] for v in results) for key in ("nll", "top1", "top5")
    }
    summary = dict(
        archive_sha256=digest(archive.read_bytes()) if archive else None,
        study=Path(read("compute.json")["profile"]).parent.name,
        manifest=manifest["fingerprint"],
        source_commit=commit,
        source_dirty=manifest["provenance"]["dirty"],
        source_hashes_checked=len(manifest["code_sha256"]),
        included_artifacts=included,
        omitted_artifacts=omitted,
        reconstructed_folds=reconstructed,
        settings=settings,
        complete=len(results),
        parameter_free_fits=sum(not r["trainable_parameters"] for r in results),
        successful_allocations=len(accounting),
        allocation_stages=dict(Counter(j["stage"] for j in jobs)),
        peak_concurrent_allocations=peak,
        cv_array_start=min(r["Start"] for r in array),
        cv_array_end=max(r["End"] for r in array),
        cv_array_seconds=span,
        cv_allocation_seconds=sum(float(r["ElapsedRaw"]) for r in array),
        total_training_draws=sum(v["training_draws"] for v in results),
        total_optimizer_updates=sum(v["optimizer_updates"] for v in results),
        frequency=baseline,
        arms=arms,
        paired_descriptive_comparisons=pairs,
        scope="CV diagnostics only; omitted binary splits/predictions/checkpoints were not "
        "independently validated here. Three correlated folds and one seed are exploratory. "
        "No held-out test or bootstrap significance claim.",
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    arm_rows = [
        dict(
            arm=a["arm"]["name"],
            parameters=a["parameters"],
            trainable_parameters=a["trainable_parameters"],
            **{f"{k}_mean": a["mean"][k] for k in ("nll", "top1", "top5")},
            **{f"{k}_sd": a["sd"][k] for k in ("nll", "top1", "top5")},
            median_seconds=a["median_seconds"],
            max_reserved_mib=a["max_reserved_mib"],
            clip_fraction=a["clip_fraction"],
        )
        for a in sorted(arms, key=lambda a: -a["mean"]["top1"])
    ]
    with (output / "all-arms.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(arm_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(arm_rows)
    (output / "cv-fits.csv").write_text((root / "timing-report/cv-fits.csv").read_text())
    with (output / "accounting.psv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(accounting[0]), delimiter="|", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(sorted(accounting, key=lambda r: r["Start"]))
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "complete",
                    "source_hashes_checked",
                    "successful_allocations",
                    "peak_concurrent_allocations",
                    "cv_array_seconds",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("study", type=Path)
    cli.add_argument("--output", type=Path, default=Path("results/robust-screen-main"))
    cli.add_argument("--archive", type=Path)
    args = cli.parse_args()
    audit(args.study, args.output, args.archive)
