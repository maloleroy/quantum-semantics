"""Audit the causal completion and combine its paired anchors in both tracks.

Data analysis only. Run with one BLAS thread. Omitted raw predictions and
checkpoints cannot be independently reconstructed from compact statistics.
"""

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from audit_target_controls import MAIN, METRICS, effect, peak_overlap, read, sha, split_keys
from review_prediction_clusters import bootstrap

EXTRA = ("hybrid-q8-local", "hybrid-q8-frozen", "hybrid-q8-xx", "hybrid-q8-zz")
ARMS = MAIN + EXTRA


def load_clusters(path, groups, counts):
    with np.load(path, allow_pickle=False) as saved:
        data = {k: saved[k] for k in saved.files}
    np.testing.assert_array_equal(data["groups"], groups)
    np.testing.assert_array_equal(data["counts"], counts)
    values = data["sums"]
    assert values.dtype == np.float64 and np.isfinite(values).all() and (values >= 0).all()
    assert (values[..., 1] <= values[..., 2]).all() and (values[..., 2] <= counts).all()
    np.testing.assert_array_equal(values[..., 1:], values[..., 1:].round())
    return data


def check_record(record, ledger, sums, counts):
    assert record["ledger"] == ledger
    assert ledger["validation"] and ledger["selected_epoch"] == ledger["stopped_epoch"] == 100
    assert ledger["training_draws"] == 2_000_000 and ledger["optimizer_updates"] == 31_300
    assert ledger["train_examples"] == 1_014_085
    assert ledger["metrics"]["examples"] == counts.sum() == 252_120
    metrics = sums.sum(0) / counts.sum()
    np.testing.assert_allclose(metrics, [record["metrics"][k] for k in METRICS], atol=1e-11, rtol=0)
    np.testing.assert_allclose(metrics, [ledger["metrics"][k] for k in METRICS], atol=1e-5, rtol=0)
    ranks = record["rank_counts_1_to_5_then_miss"]
    assert len(ranks) == 6 and sum(ranks) == counts.sum()
    np.testing.assert_allclose(
        metrics[1:], np.asarray([ranks[0], sum(ranks[:5])]) / counts.sum(), atol=1e-12, rtol=0
    )
    return metrics


def pilot_summary(pilot):
    assert pilot["status"] == "complete" and pilot["device"] == "cuda"
    assert pilot["microbatches"] == [8, 16, 32, 64] and pilot["batch_sizes"] == [64, 32]
    assert pilot["adopted_microbatch"] == 8 and pilot["repeats"] == 5
    expected = {
        (recipe, arm, batch, micro)
        for recipe in ("L4", "L2R2")
        for arm in MAIN
        for batch in (64, 32)
        for micro in (8, 16, 32, 64)
    }
    keys = [(r["recipe"], r["arm"], r["batch"], r["microbatch"]) for r in pilot["rows"]]
    assert len(keys) == len(expected) == len(set(keys)) and set(keys) == expected
    for row in pilot["rows"]:
        assert len(row["training_seconds"]) == 5
        assert all(np.isfinite(t) and t > 0 for t in row["training_seconds"])
        assert row["median_training_seconds"] == float(np.median(row["training_seconds"]))
        assert row["parity"]["passed"] == all(c["passed"] for c in row["parity"]["checks"].values())
        assert all(
            np.isfinite(c["max_abs"]) and c["max_abs"] >= 0
            for c in row["parity"]["checks"].values()
        )
    assert pilot["all_parity_passed"] == all(r["parity"]["passed"] for r in pilot["rows"])
    speeds = []
    for recipe in ("L4", "L2R2"):
        for arm in MAIN:
            for count in (64, 32):
                rows = {
                    r["microbatch"]: r
                    for r in pilot["rows"]
                    if (r["recipe"], r["arm"], r["batch"]) == (recipe, arm, count)
                }
                speeds.append(
                    dict(
                        recipe=recipe,
                        arm=arm,
                        batch=count,
                        speedup_8_to_64=rows[8]["median_training_seconds"]
                        / rows[64]["median_training_seconds"],
                        milliseconds={
                            str(m): r["median_training_seconds"] * 1000 for m, r in rows.items()
                        },
                    )
                )
    return dict(
        conditions=len(keys),
        seconds=pilot["seconds"],
        all_parity_passed=pilot["all_parity_passed"],
        speedups=speeds,
        max_abs={
            k: max(r["parity"]["checks"][k]["max_abs"] for r in pilot["rows"])
            for k in ("loss", "logits", "gradients", "parameters", "moments")
        },
        peak_torch_reserved_bytes=max(r["cuda_peak_reserved"] for r in pilot["rows"]),
        scope="Training-step medians on one fixed batch, one MIG. No evaluation or "
        "whole-fit speedup measurement; no multi-update/dropout trajectory equivalence.",
    )


def audit(root: Path, draws: int, parent=None) -> dict:
    export, protocol = read(root / "export.json"), read(root / "protocol.json")
    assert protocol["tasks"] == 8 and protocol["recipes"] == ["L4"]
    assert {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} == (
        set(export["included_sha256"]) | {"export.json"}
    )
    for checksums in (export["included_sha256"], protocol["anchor_sha256"]):
        for name, checksum in checksums.items():
            assert sha((root / name).read_bytes()) == checksum, name
    assert len(export["omitted"]) == 25
    study = root / "L4"
    manifest = read(study / "manifest.json")
    parents = {r: read(root / "anchors" / f"{r}-manifest.json") for r in ("L4", "L2R2")}
    for m in [manifest, *parents.values()]:
        assert m["fingerprint"] == sha(
            json.dumps({k: v for k, v in m.items() if k != "fingerprint"}, sort_keys=True).encode()
        )
        assert m["code_sha256"] == manifest["code_sha256"]
        assert m["target_sha256"] == manifest["target_sha256"]
        for key, value in dict(
            epochs=100,
            samples_per_epoch=20000,
            fixed_epochs=True,
            batch_size=64,
            microbatch_size=8,
            seeds=[47, 48],
            folds=1,
            final_eval_sentences=0,
            refit_controls=[],
        ).items():
            assert m["settings"][key] == value
        for name in ("sentences.json", "vocabulary.json", "corpus.json"):
            assert sha((study / name).read_bytes()) == m["artifact_sha256"][name]
    assert protocol["parent_manifests"] == [m["fingerprint"] for m in parents.values()]
    assert manifest["parent_manifests"] == protocol["parent_manifests"]
    assert manifest["artifact_sha256"] == parents["L4"]["artifact_sha256"]
    for name, checksum in manifest["code_sha256"].items():
        saved = (study / "source" / name).read_bytes()
        assert sha(saved) == checksum
        assert saved == subprocess.check_output(["git", "show", f"483bea6:{name}"])
    assert manifest["extension_sha256"] == protocol["extension_sha256"]
    for name, checksum in protocol["extension_sha256"].items():
        saved = (study / "source" / name).read_bytes()
        assert sha(saved) == checksum
        assert saved == subprocess.check_output(["git", "show", f"6d8ca19:scripts/{name}"])
    launcher = (study / "source/run_causal_controls.sh").read_text()
    assert (root / "profile-job.sbatch").read_text() == (
        launcher.split("<<'SBATCH'\n", 1)[1].split("\nSBATCH\n", 1)[0] + "\n"
    )
    if parent is not None:
        for recipe in parents:
            for name in ("manifest.json", "validation-clusters.npz"):
                assert (parent / recipe / name).read_bytes() == (
                    root / "anchors" / f"{recipe}-{name}"
                ).read_bytes()
        assert (parent / "development-results.json").read_bytes() == (
            root / "anchors/development-results.json"
        ).read_bytes()
    sentences, vocabulary = read(study / "sentences.json"), read(study / "vocabulary.json")
    keys = split_keys(sentences, vocabulary)
    assert {k: sha(v.tobytes()) for k, v in keys.items()} == manifest["target_sha256"]
    groups, counts = np.unique(keys["val_0"][:, 0], return_counts=True)
    new = load_clusters(study / "validation-clusters.npz", groups, counts)
    anchors = {
        r: load_clusters(root / "anchors" / f"{r}-validation-clusters.npz", groups, counts)
        for r in parents
    }
    paired_data = np.stack(
        (np.concatenate((anchors["L4"]["sums"], new["sums"])), anchors["L2R2"]["sums"])
    )
    labels = np.array([[arm, str(seed)] for arm in ARMS for seed in (47, 48)])
    np.testing.assert_array_equal(np.concatenate((anchors["L4"]["labels"], new["labels"])), labels)
    np.testing.assert_array_equal(anchors["L2R2"]["labels"], labels)
    new_records = read(root / "development-results.json")["rows"]
    old_records = read(root / "anchors/development-results.json")["rows"]
    assert len(new_records) == 8 and len(old_records) == 32
    rows, contrasts = [], {}
    for ri, recipe in enumerate(parents):
        for index, (arm, seed) in enumerate(labels):
            seed = int(seed)
            records = new_records if recipe == "L4" and arm in EXTRA else old_records
            (record,) = [
                r for r in records if (r["recipe"], r["arm"], r["seed"]) == (recipe, arm, seed)
            ]
            metrics = check_record(record, record["ledger"], paired_data[ri, index], counts)
            source_manifest = manifest if records is new_records else parents[recipe]
            assert record["ledger"]["manifest"] == source_manifest["fingerprint"]
            task = record["ledger"]["task"]
            assert task in source_manifest["tasks"]
            assert (task["arm"]["name"], task["seed"], task["fold"]) == (arm, seed, 0)
            assert task["objective"] == ("causal" if recipe == "L4" else "cbow")
            rows.append(
                dict(
                    recipe=recipe,
                    arm=arm,
                    seed=seed,
                    **dict(zip(METRICS, metrics.tolist(), strict=True)),
                    fit_seconds=record["ledger"]["seconds"],
                    parameters=record["ledger"]["parameters"],
                    trainable_parameters=record["ledger"]["trainable_parameters"],
                )
            )
    histories = []
    assert len(manifest["tasks"]) == 8
    for index, task in enumerate(manifest["tasks"]):
        (reference,) = [
            t
            for t in parents["L2R2"]["tasks"]
            if (t["arm"]["name"], t["seed"]) == (task["arm"]["name"], task["seed"])
        ]
        assert task["arm"] == reference["arm"] and task["arm"]["lr"] == 0.003
        folder = study / "cv" / f"task-{index:04d}"
        result, runtime, history = (
            read(folder / f) for f in ("result.json", "runtime.json", "history.json")
        )
        assert result["task"] == task and result["manifest"] == manifest["fingerprint"]
        assert runtime["device"] == "cuda" and "1g.10gb" in runtime["device_name"]
        assert runtime["repository"]["commit"].startswith("483bea6")
        assert runtime["cuda_device"]["total_memory_bytes"] < 11 * 1024**3
        assert [h["epoch"] for h in history] == list(range(101))
        assert all(h["updates"] == h["epoch"] * 313 for h in history[1:])
        assert all(h["validation"]["examples"] == 2048 for h in history)
        (record,) = [
            r for r in new_records if (r["arm"], r["seed"]) == (task["arm"]["name"], task["seed"])
        ]
        check_record(record, result, new["sums"][index], counts)
        gradients = [h["circuit_gradients"] for h in history[1:] if "circuit_gradients" in h]
        histories.append(
            dict(
                arm=task["arm"]["name"],
                seed=task["seed"],
                monitor_nll_50_100=history[100]["validation"]["nll"]
                - history[50]["validation"]["nll"],
                monitor_top5_50_100=history[100]["validation"]["top5"]
                - history[50]["validation"]["top5"],
                min_layer_gradient_rms=min((min(g["layer_rms"]) for g in gradients), default=None),
                mean_clipped_fraction=float(
                    np.mean([h["clipped_batch_fraction"] for h in history[1:]])
                ),
            )
        )
    data = paired_data.reshape(2, 8, 2, len(groups), 3)
    paired, sentence = bootstrap(data, np.stack([counts, counts]), draws=draws)
    for ri, recipe in enumerate(parents):
        for ai, arm in enumerate(ARMS):
            if ai != 3:
                contrasts[f"{recipe}:{arm}-minus-identity"] = effect(
                    data[ri], counts, paired[:, ri], sentence[:, ri], ai, 3
                )
        contrasts[f"{recipe}:CRZ-minus-frozen"] = effect(
            data[ri], counts, paired[:, ri], sentence[:, ri], 2, 5
        )
        contrasts[f"{recipe}:MLP-minus-BERT"] = effect(
            data[ri], counts, paired[:, ri], sentence[:, ri], 0, 1
        )
    jobs = [read(p) for p in (study / "job-timings").glob("*.json")]
    assert len(jobs) == 8 and {j["task_id"] for j in jobs} == set(range(8))
    for job in jobs:
        assert job["status"] == "complete" and not job["reused_result"]
        assert all(p["returncode"] == 0 for p in job["phases"])
        elapsed = (
            datetime.fromisoformat(job["finished_at"]) - datetime.fromisoformat(job["started_at"])
        ).total_seconds()
        assert abs(elapsed - job["seconds"]) < 0.01
    assert peak_overlap(jobs) <= 4
    for path in (study / "submissions").glob("*.json"):
        command = read(path)["command"]
        assert "--array=0-7%4" in command and "--gres=gpu:nvidia_a100_1g.10gb:1" in command
    pilot = read(root / "pilot/microbatch-report.json")
    assert (
        pilot["probe_sha256"] == protocol["extension_sha256"]["profile_benchmark_microbatches.py"]
    )
    assert pilot["effective_source_sha256"] == manifest["code_sha256"]
    assert pilot["manifests"] == {r: m["fingerprint"] for r, m in parents.items()}
    assert pilot["runtime"]["cuda_device"]["total_memory_bytes"] < 11 * 1024**3
    assert pilot["runtime"]["scheduler"]["SLURM_ARRAY_JOB_ID"] is None
    assert pilot["runtime"]["device"] == "cuda"
    preflight = read(root / "backend-preflight.json")
    assert preflight["device"] == "cuda" and preflight["checked"] == 71
    for arm in MAIN:
        assert (
            pilot["initialization_sha256"][f"L4/{arm}"]
            == pilot["initialization_sha256"][f"L2R2/{arm}"]
        )
    # Rebuild the shared canonical train-ID order before replaying the pilot sampler.
    all_groups = np.repeat(np.arange(len(sentences)), [len(s) - 1 for s in sentences])
    train_ids = np.flatnonzero(np.isin(all_groups, np.unique(keys["train_0"][:, 0])))
    replay = []
    for seed in (47, 48):
        rng, seen, digest = (
            np.random.default_rng(seed),
            np.zeros(len(all_groups), dtype=bool),
            hashlib.sha256(),
        )
        for _ in range(100):
            order = rng.choice(train_ids, 20000, replace=True)
            seen[order] = True
            digest.update(order.astype("<i8").tobytes())
        replay.append(
            dict(
                seed=seed,
                unique_targets=int(seen.sum()),
                eligible_targets=len(train_ids),
                coverage=float(seen.sum() / len(train_ids)),
                draw_sha256=digest.hexdigest(),
            )
        )
    sampled = np.random.default_rng(619).choice(train_ids, 64, replace=True)
    starts = np.concatenate(([0], np.cumsum([len(s) - 1 for s in sentences])))
    sampled_groups = np.searchsorted(starts, sampled, side="right") - 1
    sampled_positions = sampled - starts[sampled_groups] + 1
    lookup = {word: index for index, word in enumerate(vocabulary)}
    sampled_keys = [
        (int(g), int(p), lookup[sentences[g][p]])
        for g, p in zip(sampled_groups, sampled_positions, strict=True)
    ]
    for recipe in parents:
        with np.load(root / "pilot" / f"microbatch-inputs-{recipe}.npz") as inputs:
            np.testing.assert_array_equal(inputs["ids"], sampled)
            np.testing.assert_array_equal(inputs["targets"], [k[2] for k in sampled_keys])
            for i, (group, pos, _) in enumerate(sampled_keys):
                words = [lookup[w] for w in sentences[group]]
                left = words[max(0, pos - (4 if recipe == "L4" else 2)) : pos]
                context = [len(vocabulary)] * ((4 if recipe == "L4" else 2) - len(left)) + left
                if recipe == "L2R2":
                    context += words[pos + 1 : pos + 3]
                    context += [len(vocabulary)] * (4 - len(context))
                np.testing.assert_array_equal(inputs["contexts"][i], context)
    start, end = min(j["started_at"] for j in jobs), max(j["finished_at"] for j in jobs)
    assert pilot["finished_at"] < start
    forecast = read(root / "compute.json")
    profile = read(study / "profile-causal.json")
    assert profile["manifest"] == manifest["fingerprint"] and profile["device"] == "cuda"
    assert profile["effective_source_sha256"] == manifest["code_sha256"]
    assert (profile["batch"], profile["microbatch"], profile["threads"]) == (64, 8, 4)
    measures = {r["arm"]["name"]: r for r in profile["measurements"]}
    costs = [
        1.1
        * (
            31300 * measures[t["arm"]["name"]]["training_batch_seconds"]
            + (101 * 2048 + 252120) / 64 * measures[t["arm"]["name"]]["evaluation_batch_seconds"]
        )
        + 20
        for t in manifest["tasks"]
    ]
    np.testing.assert_allclose(sum(costs) / 3600, forecast["allocation_hours"], atol=1e-12, rtol=0)
    return dict(
        validation=dict(
            included_hashes=len(export["included_sha256"]),
            frozen_source_copies=len(manifest["code_sha256"]),
            extension_copies=3,
            new_fits=8,
            paired_core_records=32,
            groups=len(groups),
            targets=int(counts.sum()),
            target_hashes=manifest["target_sha256"],
            pilot_inputs_reconstructed=True,
            raw_token_predictions_checked=False,
        ),
        rows=rows,
        contrasts=contrasts,
        histories=histories,
        pilot=pilot_summary(pilot),
        sampler_replay=dict(
            rows=replay,
            draws_per_fit=2_000_000,
            expected_coverage=float(-np.expm1(2_000_000 * np.log1p(-1 / len(train_ids)))),
            scope="Replay implied by saved seed, canonical IDs and frozen sampler; "
            "actual DGX draw stream/checkpoints were not exported for independent verification.",
        ),
        resampling=dict(
            draws=draws,
            seed=42,
            trained_seeds=[47, 48],
            metrics=METRICS,
            scope="Paired seed/sentence and sentence-only nominal95%; "
            "two seeds, reused validation split, exploratory multiple comparisons",
        ),
        compute=dict(
            wrapper_mig_hours=sum(j["seconds"] for j in jobs) / 3600,
            wrapper_span_hours=(
                datetime.fromisoformat(end) - datetime.fromisoformat(start)
            ).total_seconds()
            / 3600,
            peak_overlap=peak_overlap(jobs),
            forecast_mig_hours=forecast["allocation_hours"],
            forecast_span_hours=forecast["sequential_arrays_hours"],
            first_wrapper=start,
            last_wrapper=end,
            scope="Fit-job wrappers only; pilot/preflight/profile, queueing "
            "and packaging excluded.",
        ),
    )


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/robust-causal-controls"))
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    result = audit(args.root, args.draws, args.parent_root)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(args.output / "core-fits.csv", result["rows"])
    means = []
    for recipe in ("L4", "L2R2"):
        for arm in ARMS:
            rows = [r for r in result["rows"] if (r["recipe"], r["arm"]) == (recipe, arm)]
            means.append(
                dict(
                    recipe=recipe,
                    arm=arm,
                    seeds="47;48",
                    **{k: sum(r[k] for r in rows) / 2 for k in METRICS},
                )
            )
    write_csv(args.output / "mean-scores.csv", means)
    write_csv(
        args.output / "pilot-step-times.csv",
        [
            dict(
                recipe=r["recipe"],
                arm=r["arm"],
                batch=r["batch"],
                microbatch=micro,
                median_step_ms=ms,
                speedup_8_to_64=r["speedup_8_to_64"],
            )
            for r in result["pilot"]["speedups"]
            for micro, ms in r["milliseconds"].items()
        ],
    )
    print(json.dumps(dict(validation=result["validation"], compute=result["compute"]), indent=2))


if __name__ == "__main__":
    main()
