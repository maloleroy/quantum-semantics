"""Audit compact target-control exports without importing models or training.

Run with one BLAS thread. Raw predictions/checkpoints remain on DGX; exported
sentence sums can validate aggregation, but cannot reconstruct token rankings.
"""

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from review_prediction_clusters import bootstrap, interval

RECIPES = ("L4", "L2pad", "L2R2")
METRICS = ("nll", "top1", "top5")
MAIN = ("mlp-32", "bert-32", "hybrid-q8-depth2-init0.05", "hybrid-q8-identity")


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def split_keys(sentences, vocabulary, seed=2026):
    """Rebuild original five-fold fold0, independently of prepared NPZ arrays."""
    keys = sorted(map(tuple, sentences))
    assert len(keys) == len(set(keys))
    order = np.random.default_rng(seed).permutation(len(keys))
    test = {keys[i] for i in order[: round(0.2 * len(keys))]}
    dev = [key for key in keys if key not in test]
    order = np.random.default_rng(seed + 1).permutation(len(dev))
    val = {dev[i] for i in np.array_split(order, 5)[0]}
    lookup = {word: i for i, word in enumerate(vocabulary)}
    rows = {name: [] for name in ("train_0", "val_0", "development", "test")}
    for group, words in enumerate(sentences):
        key = tuple(words)
        names = ("test",) if key in test else ("development", "val_0" if key in val else "train_0")
        targets = [(group, pos, lookup[word]) for pos, word in enumerate(words) if pos > 0]
        for name in names:
            rows[name].extend(targets)
    return {name: np.asarray(values, dtype="<i8") for name, values in rows.items()}


def peak_overlap(jobs):
    events = []
    for job in jobs:
        events.extend(((job["started_at"], 1), (job["finished_at"], -1)))
    active = peak = 0
    for _, delta in sorted(events):  # End before start at identical timestamps.
        active += delta
        peak = max(peak, active)
        assert active >= 0
    assert active == 0
    return peak


def corpus_composition(root):
    """Check local source files before deriving source membership and coverage."""
    from qcse.data import load_corpus

    corpus = load_corpus()
    study = root / "L4"
    assert corpus.sentences == read(study / "sentences.json")
    assert corpus.vocabulary == read(study / "vocabulary.json")
    manifest = read(study / "manifest.json")
    for source in manifest["corpus"]["sources"]:
        assert sha(Path(Path(source["path"]).name).read_bytes()) == source["sha256"]
    keys = split_keys(corpus.sentences, corpus.vocabulary)
    result = {}
    for split, rows in keys.items():
        groups = np.unique(rows[:, 0])
        memberships = {}
        for label in ("phrases", "cleaned", "both"):
            ids = [
                int(g)
                for g in groups
                if (
                    len(corpus.sources[g]) == 2 if label == "both" else corpus.sources[g] == [label]
                )
            ]
            memberships[label] = dict(
                groups=len(ids), positive_targets=sum(len(corpus.sentences[g]) - 1 for g in ids)
            )
        result[split] = memberships
    counts = np.bincount(keys["train_0"][:, 2], minlength=len(corpus.vocabulary))
    result["train_vocabulary"] = dict(
        words=len(corpus.vocabulary),
        words_with_training_targets=int((counts > 0).sum()),
        val_unseen_targets=int((counts[keys["val_0"][:, 2]] == 0).sum()),
        val_targets=len(keys["val_0"]),
    )
    return result


def effect(values, counts, paired, sentence, left, right):
    """A paired contrast, retaining both seeds and both averaging estimands."""
    a, b = left, right
    per_seed = values[a].sum(1) / counts.sum() - values[b].sum(1) / counts.sum()
    return dict(
        per_seed=per_seed.tolist(),
        mean=per_seed.mean(0).tolist(),
        sentence_macro=((values[a] - values[b]) / counts[None, :, None]).mean((0, 1)).tolist(),
        seed_sentence_nominal95=interval(paired[:, a] - paired[:, b], 0.95),
        sentence_only_nominal95=interval(sentence[:, a] - sentence[:, b], 0.95),
    )


def audit(root: Path, draws: int) -> dict:
    export = read(root / "export.json")
    included = export["included_sha256"]
    assert {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} == (
        set(included) | {"export.json"}
    )
    for name, checksum in included.items():
        assert sha((root / name).read_bytes()) == checksum, name
    assert len(export["omitted"]) == 99
    originals = {}
    clusters, rows, jobs, recipe_times = {}, [], [], {}
    reported = read(root / "development-results.json")["rows"]
    source_copies = extension_copies = 0
    expected = None
    for recipe in RECIPES:
        study = root / recipe
        m = read(study / "manifest.json")
        assert m["fingerprint"] == sha(
            json.dumps({k: v for k, v in m.items() if k != "fingerprint"}, sort_keys=True).encode()
        )
        assert m["settings"]["seeds"] == [47, 48]
        assert m["settings"]["fixed_epochs"]
        for key, value in dict(
            epochs=100,
            samples_per_epoch=20000,
            batch_size=64,
            microbatch_size=8,
            folds=1,
            refit_controls=[],
        ).items():
            assert m["settings"][key] == value
        for name in ("sentences.json", "vocabulary.json", "corpus.json"):
            assert sha((study / name).read_bytes()) == m["artifact_sha256"][name]
        if expected is None:
            expected = split_keys(read(study / "sentences.json"), read(study / "vocabulary.json"))
        assert {k: sha(v.tobytes()) for k, v in expected.items()} == m["target_sha256"]
        for revision, hashes in (("483bea6", m["code_sha256"]), ("555739e", m["extension_sha256"])):
            for name, checksum in hashes.items():
                key = (revision, name)
                if key not in originals:
                    path = f"scripts/{name}" if revision == "555739e" else name
                    originals[key] = subprocess.check_output(["git", "show", f"{revision}:{path}"])
                data = (study / "source" / name).read_bytes()
                assert data == originals[key] and sha(data) == checksum
                source_copies += revision == "483bea6"
                extension_copies += revision == "555739e"
        with np.load(study / "validation-clusters.npz", allow_pickle=False) as archive:
            a = {k: archive[k] for k in archive.files}
        groups, counts = np.unique(expected["val_0"][:, 0], return_counts=True)
        np.testing.assert_array_equal(a["groups"], groups)
        np.testing.assert_array_equal(a["counts"], counts)
        assert a["sums"].dtype == np.float64 and np.isfinite(a["sums"]).all()
        assert (a["sums"] >= 0).all()
        assert (a["sums"][..., 1] <= a["sums"][..., 2]).all()
        assert (a["sums"][..., 2] <= counts).all()
        np.testing.assert_array_equal(a["sums"][..., 1:], a["sums"][..., 1:].round())
        clusters[recipe] = a
        local_jobs = [read(f) for f in sorted((study / "job-timings").glob("*.json"))]
        assert len(local_jobs) == len(m["tasks"])
        assert {j["task_id"] for j in local_jobs} == set(range(len(m["tasks"])))
        for job in local_jobs:
            assert job["status"] == "complete" and not job["reused_result"]
            assert all(p["returncode"] == 0 for p in job["phases"])
            elapsed = (
                datetime.fromisoformat(job["finished_at"])
                - datetime.fromisoformat(job["started_at"])
            ).total_seconds()
            assert abs(elapsed - job["seconds"]) < 0.01
        jobs.extend(local_jobs)
        for index, task in enumerate(m["tasks"]):
            folder = study / "cv" / f"task-{index:04d}"
            result, runtime, history = (
                read(folder / f) for f in ("result.json", "runtime.json", "history.json")
            )
            arm, seed = task["arm"]["name"], task["seed"]
            assert a["labels"][index].tolist() == [arm, str(seed)]
            assert result["task"] == task and result["manifest"] == m["fingerprint"]
            assert result["validation"] and task["arm"]["lr"] == 0.003
            assert result["selected_epoch"] == result["stopped_epoch"] == 100
            assert result["training_draws"] == 2_000_000
            assert result["optimizer_updates"] == 31_300
            assert result["metrics"]["examples"] == counts.sum() == 252_120
            assert result["train_examples"] == len(expected["train_0"]) == 1_014_085
            assert runtime["device"] == "cuda" and "1g.10gb" in runtime["device_name"]
            assert runtime["repository"]["commit"].startswith("483bea6")
            assert runtime["cuda_device"]["total_memory_bytes"] < 11 * 1024**3
            assert [h["epoch"] for h in history] == list(range(101))
            assert all(h["updates"] == h["epoch"] * 313 for h in history[1:])
            assert all(h["validation"]["examples"] == 2048 for h in history)
            matching = [
                r for r in reported if (r["recipe"], r["arm"], r["seed"]) == (recipe, arm, seed)
            ]
            assert len(matching) == 1
            record = matching[0]
            assert record["ledger"] == result
            means = a["sums"][index].sum(0) / counts.sum()
            np.testing.assert_allclose(
                means, [record["metrics"][k] for k in METRICS], rtol=0, atol=1e-11
            )
            np.testing.assert_allclose(
                means, [result["metrics"][k] for k in METRICS], rtol=0, atol=1e-5
            )
            ranks = record["rank_counts_1_to_5_then_miss"]
            assert len(ranks) == 6 and sum(ranks) == counts.sum()
            np.testing.assert_allclose(
                means[1:], np.asarray([ranks[0], sum(ranks[:5])]) / counts.sum(), rtol=0, atol=1e-12
            )
            gradients = [h["circuit_gradients"] for h in history[1:] if "circuit_gradients" in h]
            rows.append(
                dict(
                    recipe=recipe,
                    arm=arm,
                    seed=seed,
                    **dict(zip(METRICS, means.tolist(), strict=True)),
                    fit_seconds=result["seconds"],
                    parameters=result["parameters"],
                    trainable_parameters=result["trainable_parameters"],
                    cuda_peak_bytes=result["cuda_peak_bytes"],
                    cuda_peak_reserved_bytes=result["cuda_peak_reserved_bytes"],
                    monitor_nll_50=history[50]["validation"]["nll"],
                    monitor_nll_100=history[100]["validation"]["nll"],
                    monitor_top5_change_50_100=history[100]["validation"]["top5"]
                    - history[50]["validation"]["top5"],
                    mean_clipped_fraction=float(
                        np.mean([h["clipped_batch_fraction"] for h in history[1:]])
                    ),
                    circuit_layer_rms_min=min(
                        (min(g["layer_rms"]) for g in gradients), default=None
                    ),
                    final_circuit_gradient=gradients[-1] if gradients else None,
                )
            )
        start = min(j["started_at"] for j in local_jobs)
        end = max(j["finished_at"] for j in local_jobs)
        recipe_times[recipe] = dict(
            tasks=len(local_jobs),
            array=sorted({j["scheduler"]["SLURM_ARRAY_JOB_ID"] for j in local_jobs}),
            wrapper_slot_hours=sum(j["seconds"] for j in local_jobs) / 3600,
            wrapper_span_hours=(
                datetime.fromisoformat(end) - datetime.fromisoformat(start)
            ).total_seconds()
            / 3600,
            peak=peak_overlap(local_jobs),
        )
        submissions = [read(f) for f in (study / "submissions").glob("*.json")]
        assert submissions
        for submission in submissions:
            command = submission["command"]
            assert "--gres=gpu:nvidia_a100_1g.10gb:1" in command
            assert "--ntasks=1" in command and "--nodes=1" in command
            assert f"--array=0-{len(m['tasks']) - 1}%4" in command
    contrasts = {}
    # One shared resampling across all recipes retains cross-context pairing.
    main = np.stack([clusters[r]["sums"][:8].reshape(4, 2, len(groups), 3) for r in RECIPES])
    paired, sentence = bootstrap(main, np.stack([counts] * 3), draws=draws)
    for model, arm in enumerate(MAIN):
        contrasts[f"L2R2-minus-L2pad:{arm}"] = effect(
            main[:, model], counts, paired[:, :, model], sentence[:, :, model], 2, 1
        )
    for recipe_index, recipe in enumerate(RECIPES):
        for name, left, right in (("CRZ-minus-identity", 2, 3), ("MLP-minus-BERT", 0, 1)):
            contrasts[f"{recipe}:{name}"] = effect(
                main[recipe_index],
                counts,
                paired[:, recipe_index],
                sentence[:, recipe_index],
                left,
                right,
            )
    all_models = clusters["L2R2"]["sums"].reshape(8, 2, len(groups), 3)
    paired_all, sentence_all = bootstrap(all_models[None], counts[None], draws=draws)
    for index, name in enumerate(clusters["L2R2"]["labels"][::2, 0]):
        if index != 3:
            contrasts[f"L2R2:{name}-minus-identity"] = effect(
                all_models, counts, paired_all[:, 0], sentence_all[:, 0], index, 3
            )
    contrasts["L2R2:CRZ-minus-frozen"] = effect(
        all_models, counts, paired_all[:, 0], sentence_all[:, 0], 2, 5
    )
    np.testing.assert_allclose(paired_all[:, 0, :4], paired[:, 2], atol=1e-12, rtol=0)
    start, end = min(j["started_at"] for j in jobs), max(j["finished_at"] for j in jobs)
    assert peak_overlap(jobs) <= 4
    forecast = read(root / "compute.json")
    assert expected is not None
    return dict(
        validation=dict(
            included_hashes=len(included),
            source_copies=source_copies,
            extension_copies=extension_copies,
            fits=len(rows),
            target_hashes={k: sha(v.tobytes()) for k, v in expected.items()},
            validation_groups=len(groups),
            validation_targets=int(counts.sum()),
            omitted_files=len(export["omitted"]),
            raw_predictions_independently_checked=False,
        ),
        resampling=dict(
            draws=draws,
            seed=42,
            metrics=METRICS,
            seeds=[47, 48],
            scope="Exploratory nominal intervals; reused validation "
            "split, only two training seeds; no multiplicity correction or new test",
        ),
        rows=rows,
        contrasts=contrasts,
        compute=dict(
            recipes=recipe_times,
            peak_wrapper_overlap=peak_overlap(jobs),
            wrapper_slot_hours=sum(j["seconds"] for j in jobs) / 3600,
            first_wrapper=start,
            last_wrapper=end,
            wrapper_span_hours=(
                datetime.fromisoformat(end) - datetime.fromisoformat(start)
            ).total_seconds()
            / 3600,
            forecast_slot_hours=forecast["allocation_hours"],
            forecast_sequential_array_hours=forecast["sequential_arrays_hours"],
            scope="Fit-job wrapper records only. Not Slurm allocation accounting; "
            "excludes earlier profile job, queueing and final report packaging.",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--check-local-corpus", action="store_true")
    args = parser.parse_args()
    result = audit(args.root, args.draws)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.check_local_corpus:
        (args.output / "source-composition.json").write_text(
            json.dumps(corpus_composition(args.root), indent=2) + "\n"
        )
    scalar_rows = [
        {k: v for k, v in row.items() if k != "final_circuit_gradient"} for row in result["rows"]
    ]
    with (args.output / "fits.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scalar_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(scalar_rows)
    lines = [
        "| Context | Model | NLL | Top-1 % | Top-5 % | Fit minutes |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for recipe in RECIPES:
        rows = [r for r in result["rows"] if r["recipe"] == recipe]
        for arm in dict.fromkeys(r["arm"] for r in rows):
            selected = [r for r in rows if r["arm"] == arm]
            means = {k: np.mean([r[k] for r in selected]) for k in (*METRICS, "fit_seconds")}
            lines.append(
                f"| {recipe} | {arm} | {means['nll']:.4f} | "
                f"{100 * means['top1']:.3f} | {100 * means['top5']:.3f} | "
                f"{means['fit_seconds'] / 60:.2f} |"
            )
    (args.output / "scores.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(validation=result["validation"], compute=result["compute"]), indent=2))


if __name__ == "__main__":
    main()
