"""Prepare and report a development-only target/context control study.

Uses the frozen benchmark trainer unchanged. All three recipes share canonical
causal target IDs and saved fold-0 groups; only context slots change.
"""

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

from qcse.benchmark import (
    configurations,
    fingerprint,
    load_manifest,
    repository_state,
    save_code_snapshot,
    validate_result,
)
from qcse.benchmark_compute_report import schedule_seconds
from qcse.outputs import atomic_path, new_run_directory, save_npz, write_json

MAIN = ["mlp-32", "bert-32", "hybrid-q8-depth2-init0.05", "hybrid-q8-identity"]
EXTRA = ["hybrid-q8-local", "hybrid-q8-frozen", "hybrid-q8-xx", "hybrid-q8-zz"]
RECIPES = ("L4", "L2pad", "L2R2")


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def target_hash(data, ids):
    return sha(
        np.column_stack([data[k][ids] for k in ("groups", "positions", "targets")])
        .astype("<i8")
        .tobytes()
    )


def derive(causal, cbow, recipe, vocabulary):
    """Retain canonical causal IDs, including train/val/test membership."""
    common = np.flatnonzero(cbow["positions"] > 0)
    for key in ("groups", "positions", "targets"):
        np.testing.assert_array_equal(causal[key], cbow[key][common])
    if (causal["positions"] <= 0).any():
        raise ValueError("Canonical targets must exclude position zero")
    names = (
        "contexts",
        "targets",
        "groups",
        "positions",
        "development",
        "test",
        "test_eval",
        "train_0",
        "val_0",
        "val_eval_0",
    )
    data = {k: causal[k].copy() for k in names}
    if recipe in ("L2pad", "L2R2"):
        data["contexts"] = cbow["contexts"][common].copy()
        if recipe == "L2pad":
            data["contexts"][:, 2:] = vocabulary
    elif recipe != "L4":
        raise ValueError("Unknown context recipe")
    for left, right in (("train_0", "val_0"), ("development", "test")):
        if np.intersect1d(data["groups"][data[left]], data["groups"][data[right]]).size:
            raise ValueError("Sentence groups overlap")
    np.testing.assert_array_equal(data["val_0"], data["val_eval_0"])
    return data


def prepare(causal_root, cbow_root, output):
    parents = [load_manifest(p) for p in (causal_root, cbow_root)]
    if parents[0]["settings"]["objectives"] != ["causal"] or parents[1]["settings"][
        "objectives"
    ] != ["cbow"]:
        raise ValueError("Need the original causal and CBOW confirmation studies")
    assert parents[0]["code_sha256"] == parents[1]["code_sha256"]
    for name in ("vocabulary.json", "sentences.json"):
        assert parents[0]["artifact_sha256"][name] == parents[1]["artifact_sha256"][name]
    for p in parents:
        if p["settings"]["window"] != 4 or p["settings"]["final_eval_sentences"] != 0:
            raise ValueError("Need four-slot, full-validation parent studies")
    with np.load(causal_root / "causal.npz") as a, np.load(cbow_root / "cbow.npz") as b:
        causal, cbow = ({k: archive[k] for k in archive.files} for archive in (a, b))
    tools = Path(__file__).parent
    tool_hashes = {
        name: sha((tools / name).read_bytes())
        for name in ("target_control_study.py", "run_target_controls.sh")
    }
    with new_run_directory(output, "target-controls", dict(stage="prepare")) as cohort:
        expected_hashes = None
        for recipe in RECIPES:
            root = cohort / recipe
            root.mkdir()
            objective = "causal" if recipe == "L4" else "cbow"
            data = derive(causal, cbow, recipe, parents[0]["vocabulary_size"])
            save_npz(root / f"{objective}.npz", **data)
            for name in ("vocabulary.json", "sentences.json", "corpus.json"):
                shutil.copyfile(causal_root / name, root / name)
            names = MAIN + (EXTRA if recipe == "L2R2" else [])
            arms = {a["name"]: a for a in configurations(True)}
            settings = copy.deepcopy(parents[0]["settings"])
            settings.pop("followup", None)
            settings.update(
                objectives=[objective],
                arms=names,
                seeds=[47, 48],
                folds=1,
                epochs=100,
                fixed_epochs=True,
                samples_per_epoch=20000,
                batch_size=64,
                microbatch_size=8,
                eval_examples=2048,
                threads=4,
                refit_controls=[],
                primary_comparisons=[],
                planning_factor=1.1,
                target_control=dict(
                    recipe=recipe,
                    source_fold=0,
                    source_folds=parents[0]["settings"]["folds"],
                    selection="No selection/refit/test; development comparison only",
                    sampler="Original uniform-token replacement sampler",
                    full_eligible_pool=True,
                ),
            )
            tasks = [
                dict(objective=objective, arm=arms[name], seed=seed, fold=0)
                for name in names
                for seed in settings["seeds"]
            ]
            hashes = {
                key: target_hash(data, data[key])
                for key in ("train_0", "val_0", "development", "test")
            }
            if expected_hashes is None:
                expected_hashes = hashes
            assert hashes == expected_hashes
            manifest: dict = dict(
                schema=1,
                settings=settings,
                tasks=tasks,
                vocabulary_size=parents[0]["vocabulary_size"],
                corpus=parents[0]["corpus"],
                provenance=repository_state(),
                lock_sha256=parents[0]["lock_sha256"],
                code_sha256=parents[0]["code_sha256"],
                parent_manifests=[p["fingerprint"] for p in parents],
                extension_sha256=tool_hashes,
                target_sha256=hashes,
                artifact_sha256={p.name: sha(p.read_bytes()) for p in root.iterdir()},
            )
            manifest["fingerprint"] = fingerprint(manifest)
            save_code_snapshot(root, manifest["code_sha256"])
            write_json(root / "manifest.json", manifest)
            for name in tool_hashes:
                shutil.copyfile(tools / name, root / "source" / name)
        write_json(
            cohort / "protocol.json",
            dict(
                recipes=RECIPES,
                tasks=32,
                seeds=[47, 48],
                epochs=100,
                samples_per_epoch=20000,
                max_concurrent_mig=4,
                primary_development_question="L2R2 vs L2pad on common targets; top5 then NLL/top1",
                interpretation="Discovery, two seeds and one reused validation split; "
                "no new test claim",
                extension_sha256=tool_hashes,
            ),
        )
    return cohort


def verify_tools(cohort):
    expected = read(cohort / "protocol.json")["extension_sha256"]
    for recipe in read(cohort / "protocol.json")["recipes"]:
        assert read(cohort / recipe / "manifest.json")["extension_sha256"] == expected
    for name, checksum in expected.items():
        if sha((Path(__file__).parent / name).read_bytes()) != checksum:
            raise ValueError("Control-study tools differ from preparation")
    for name, checksum in read(cohort / "protocol.json").get("anchor_sha256", {}).items():
        if sha((cohort / name).read_bytes()) != checksum:
            raise ValueError("Saved parent evidence changed")


def prepare_causal_completion(parent, output):
    """Only the eight missing fits; preserve the original L4 arrays and runtime."""
    manifests = {r: load_manifest(parent / r) for r in ("L4", "L2R2")}
    left, both = manifests["L4"], manifests["L2R2"]
    for key in ("code_sha256", "target_sha256", "vocabulary_size"):
        if left[key] != both[key]:
            raise ValueError(f"Parent tracks differ in {key}")
    for m in manifests.values():
        for key, value in dict(
            epochs=100,
            samples_per_epoch=20000,
            batch_size=64,
            microbatch_size=8,
            seeds=[47, 48],
            fixed_epochs=True,
            final_eval_sentences=0,
        ).items():
            if m["settings"][key] != value:
                raise ValueError(f"Unexpected parent setting: {key}")
    if left["settings"]["objectives"] != ["causal"]:
        raise ValueError("Expected causal L4 parent")
    if both["settings"]["objectives"] != ["cbow"]:
        raise ValueError("Expected bidirectional parent")
    evidence = read(parent / "development-results.json")["rows"]
    for recipe, manifest in manifests.items():
        expected = {(t["arm"]["name"], t["seed"]) for t in manifest["tasks"]}
        observed = {(r["arm"], r["seed"]) for r in evidence if r["recipe"] == recipe}
        if observed != expected:
            raise ValueError("Parent development results are incomplete")
        for index, task in enumerate(manifest["tasks"]):
            row = next(
                r
                for r in evidence
                if (r["recipe"], r["arm"], r["seed"]) == (recipe, task["arm"]["name"], task["seed"])
            )
            result = read(parent / recipe / "cv" / f"task-{index:04d}" / "result.json")
            if (
                row["ledger"] != result
                or result["manifest"] != manifest["fingerprint"]
                or result["task"] != task
                or not result["validation"]
                or result["stopped_epoch"] != 100
                or result["selected_epoch"] != 100
                or result["training_draws"] != 2_000_000
                or result["optimizer_updates"] != 31_300
            ):
                raise ValueError("Parent result provenance or full budget differs")
    tools = Path(__file__).parent
    hashes = {
        name: sha((tools / name).read_bytes())
        for name in (
            "target_control_study.py",
            "run_causal_controls.sh",
            "profile_benchmark_microbatches.py",
        )
    }
    arms = {t["arm"]["name"]: t["arm"] for t in both["tasks"]}
    with new_run_directory(output, "causal-controls", dict(stage="prepare")) as cohort:
        root = cohort / "L4"
        root.mkdir()
        for name in left["artifact_sha256"]:
            shutil.copyfile(parent / "L4" / name, root / name)
        manifest = copy.deepcopy(left)
        manifest["tasks"] = [
            dict(objective="causal", arm=arms[name], seed=seed, fold=0)
            for name in EXTRA
            for seed in (47, 48)
        ]
        if any(t["arm"]["lr"] != 0.003 for t in manifest["tasks"]):
            raise ValueError("Parent controls do not use LR0.003")
        manifest["settings"].update(arms=EXTRA, primary_comparisons=[], refit_controls=[])
        manifest["settings"]["target_control"]["completion"] = "Missing causal mechanisms only"
        manifest["parent_manifests"] = [m["fingerprint"] for m in manifests.values()]
        manifest["extension_sha256"] = hashes
        manifest["provenance"] = repository_state()
        manifest.pop("fingerprint")
        manifest["fingerprint"] = fingerprint(manifest)
        save_code_snapshot(root, manifest["code_sha256"])
        for name in hashes:
            shutil.copyfile(tools / name, root / "source" / name)
        write_json(root / "manifest.json", manifest)
        anchors = cohort / "anchors"
        anchors.mkdir()
        shutil.copyfile(parent / "development-results.json", anchors / "development-results.json")
        for recipe in manifests:
            for name in ("manifest.json", "validation-clusters.npz"):
                shutil.copyfile(parent / recipe / name, anchors / f"{recipe}-{name}")
        write_json(
            cohort / "protocol.json",
            dict(
                recipes=["L4"],
                tasks=8,
                seeds=[47, 48],
                epochs=100,
                samples_per_epoch=20000,
                max_concurrent_mig=4,
                extension_sha256=hashes,
                parent=str(parent.resolve()),
                parent_manifests=manifest["parent_manifests"],
                pilot_tracks=["L4", "L2R2"],
                adopted_microbatch=8,
                anchor_sha256={
                    str(p.relative_to(cohort)): sha(p.read_bytes())
                    for p in sorted(anchors.iterdir())
                },
                interpretation="Development only; eight new causal fits paired with saved anchors. "
                "Pilot is diagnostic; no automatic microbatch change or test refits.",
            ),
        )
    return cohort


def verify_pilot(cohort):
    verify_tools(cohort)
    protocol = read(cohort / "protocol.json")
    report = read(cohort / "pilot" / "microbatch-report.json")
    if report["status"] != "complete" or report["device"] != "cuda":
        raise ValueError("Need a completed CUDA microbatch pilot")
    expected = {
        (recipe, name, batch, micro)
        for recipe in ("L4", "L2R2")
        for name in MAIN
        for batch in (64, 32)
        for micro in (8, 16, 32, 64)
    }
    observed = {(r["recipe"], r["arm"], r["batch"], r["microbatch"]) for r in report["rows"]}
    if observed != expected or len(report["rows"]) != len(expected):
        raise ValueError("Incomplete microbatch pilot")
    parent = Path(protocol["parent"])
    if report["manifests"] != {r: load_manifest(parent / r)["fingerprint"] for r in ("L4", "L2R2")}:
        raise ValueError("Pilot uses different parents")
    if report["effective_source_sha256"] != load_manifest(cohort / "L4")["code_sha256"]:
        raise ValueError("Pilot uses different training sources")
    if report["probe_sha256"] != protocol["extension_sha256"]["profile_benchmark_microbatches.py"]:
        raise ValueError("Pilot probe changed")
    if report["runtime"]["cuda_device"]["total_memory_bytes"] >= 11 * 1024**3:
        raise ValueError("Pilot did not use a 10GB MIG")
    if not report["all_parity_passed"]:
        print(
            "Some microbatch parity checks differ; retaining the original microbatch8.", flush=True
        )


def complete(root):
    manifest = load_manifest(root)
    return all(
        (root / "cv" / f"task-{i:04d}" / "result.json").exists()
        for i in range(len(manifest["tasks"]))
    )


def estimate(cohort):
    verify_tools(cohort)
    rows = []
    for recipe in read(cohort / "protocol.json")["recipes"]:
        root = cohort / recipe
        manifest = load_manifest(root)
        (objective,) = manifest["settings"]["objectives"]
        profile = read(root / f"profile-{objective}.json")
        if profile["manifest"] != manifest["fingerprint"] or profile["device"] != "cuda":
            raise ValueError("Need a matching CUDA profile")
        if profile["effective_source_sha256"] != manifest["code_sha256"]:
            raise ValueError("CUDA profile uses different sources")
        assert profile["objective"] == objective
        assert (profile["batch"], profile["microbatch"], profile["threads"]) == (64, 8, 4)
        measurements = {r["arm"]["name"]: r for r in profile["measurements"]}
        settings = manifest["settings"]
        with np.load(root / f"{objective}.npz") as data:
            final_count = len(data["val_eval_0"])
        costs = []
        for task in manifest["tasks"]:
            row = measurements[task["arm"]["name"]]
            steps = math.ceil(settings["samples_per_epoch"] / settings["batch_size"])
            train = steps * settings["epochs"] * row["training_batch_seconds"]
            monitored = (settings["epochs"] + 1) * min(settings["eval_examples"], final_count)
            evaluation = (monitored + final_count) * row["evaluation_batch_seconds"] / 64
            seconds = 1.1 * (train + evaluation) + 20
            if not math.isfinite(seconds) or seconds <= 0 or seconds > 23 * 3600:
                raise ValueError("Invalid cost or fit exceeds the 24-hour allocation allowance")
            costs.append(seconds)
        rows.append(
            dict(
                recipe=recipe,
                tasks=len(costs),
                allocation_hours=sum(costs) / 3600,
                scheduled_hours=schedule_seconds(costs, 4) / 3600,
                longest_fit_hours=max(costs) / 3600,
            )
        )
    result = dict(
        studies=rows,
        total_tasks=sum(r["tasks"] for r in rows),
        refits=0,
        allocation_hours=sum(r["allocation_hours"] for r in rows),
        sequential_arrays_hours=sum(r["scheduled_hours"] for r in rows),
        assumptions="Fresh CUDA batch medians, 1.10 planning multiplier plus 20s/job; "
        "100x20000 draws, final full validation, four slots. No held-out refits. "
        "Queueing, checkpoint I/O and restarts omitted; multiplier is not a confidence bound.",
    )
    write_json(cohort / "compute.json", result)
    print(json.dumps(result, indent=2), flush=True)


def report(cohort):
    verify_tools(cohort)
    rows = []
    for recipe in read(cohort / "protocol.json")["recipes"]:
        root = cohort / recipe
        manifest = load_manifest(root)
        (objective,) = manifest["settings"]["objectives"]
        with np.load(root / f"{objective}.npz") as a:
            data = {k: a[k] for k in a.files if k != "contexts"}
        clusters, labels = [], []
        for i, task in enumerate(manifest["tasks"]):
            folder = root / "cv" / f"task-{i:04d}"
            result = validate_result(root, manifest, task, folder, True, data)
            with np.load(folder / "predictions.npz") as saved:
                values = saved["metrics"].astype(np.float64)
                groups = saved["groups"]
                matches = saved["top_ids"] == saved["targets"][:, None]
                ranks = np.where(matches.any(1), matches.argmax(1) + 1, 6)
                unique, inverse = np.unique(groups, return_inverse=True)
                clusters.append(
                    np.stack([np.bincount(inverse, weights=v) for v in values.T], axis=1)
                )
                if i == 0:
                    counts = np.bincount(inverse)
                else:
                    np.testing.assert_array_equal(counts, np.bincount(inverse))
            labels.append([task["arm"]["name"], str(task["seed"])])
            rows.append(
                dict(
                    recipe=recipe,
                    arm=task["arm"]["name"],
                    seed=task["seed"],
                    metrics=dict(
                        zip(("nll", "top1", "top5"), values.mean(0).tolist(), strict=True)
                    ),
                    rank_counts_1_to_5_then_miss=np.bincount(ranks, minlength=7)[1:].tolist(),
                    predictions_sha256=sha((folder / "predictions.npz").read_bytes()),
                    ledger=result,
                )
            )
        save_npz(
            root / "validation-clusters.npz",
            sums=np.stack(clusters),
            groups=unique,
            counts=counts,
            labels=np.array(labels),
        )
    write_json(
        cohort / "development-results.json",
        dict(
            rows=rows,
            scope="Validation only; "
            "two training seeds, reused fold 0. No independent held-out inference. "
            "Float64 aggregates of float32 token values; full raw predictions remain on DGX.",
        ),
    )
    selected, omitted = [], []
    for file in sorted(cohort.rglob("*")):
        if not file.is_file():
            continue
        relative = str(file.relative_to(cohort))
        compact = file.name.endswith("validation-clusters.npz") or file.name.startswith(
            "microbatch-inputs-"
        )
        if file.suffix in (".pt", ".npz") and not compact:
            omitted.append(relative)
        else:
            selected.append(file)
    export = cohort / "export.json"
    write_json(
        export,
        dict(
            included_sha256={
                str(f.relative_to(cohort)): sha(f.read_bytes()) for f in selected if f != export
            },
            omitted=omitted,
            scope="Validation summaries, raw sentence aggregates and rank counts. "
            "Full prepared arrays, token predictions and checkpoints retained on DGX.",
        ),
    )
    archive = cohort.parent / (cohort.name + "-diagnostics.zip")
    with atomic_path(archive) as temporary:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as destination:
            for file in selected + [export] if export not in selected else selected:
                destination.write(file, str(file.relative_to(cohort.parent)))
    print(archive, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=(
            "prepare",
            "prepare-causal",
            "estimate",
            "report",
            "verify",
            "verify-pilot",
            "complete",
        ),
    )
    p.add_argument("cohort", type=Path, nargs="?")
    args = p.parse_args()
    if args.stage == "prepare-causal":
        parent = Path(Path("outputs/robust-target-controls-study.txt").read_text().strip())
        cohort = prepare_causal_completion(parent, Path("outputs/robust-causal-controls"))
        with atomic_path(Path("outputs/robust-causal-controls-study.txt")) as temporary:
            temporary.write_text(str(cohort) + "\n")
        print(cohort)
    elif args.stage == "prepare":
        base = Path("outputs/robust-followup")
        causal = Path((base / "confirm-causal-study.txt").read_text().strip())
        cbow = Path((base / "confirm-cbow-study.txt").read_text().strip())
        cohort = prepare(causal, cbow, Path("outputs/robust-target-controls"))
        with atomic_path(Path("outputs/robust-target-controls-study.txt")) as temporary:
            temporary.write_text(str(cohort) + "\n")
        print(cohort)
    elif args.cohort is None:
        p.error("This stage requires the cohort directory")
    elif args.stage == "estimate":
        estimate(args.cohort)
    elif args.stage == "report":
        report(args.cohort)
    elif args.stage == "complete":
        sys.exit(0 if complete(args.cohort) else 1)
    elif args.stage == "verify-pilot":
        verify_pilot(args.cohort)
    else:
        verify_tools(args.cohort)
        for recipe in read(args.cohort / "protocol.json")["recipes"]:
            load_manifest(args.cohort / recipe)


if __name__ == "__main__":
    main()
