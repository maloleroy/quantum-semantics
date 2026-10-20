"""Audit the completed follow-up JSON export without loading or training models.

Binary predictions, checkpoints and prepared indices are deliberately outside
this audit. Reported intervals are retained, not independently regenerated.
"""

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

METRICS = ("nll", "top1", "top5")
PAIR = ("hybrid-q8-depth2-init0.05", "hybrid-q8-identity")
CALIBRATION = {"profile", "estimate", "trainability_probe", "nlp_trainability_probe"}


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, sort_keys=True).encode())


def split_counts(root, manifest) -> dict:
    """Independent expected sentence membership/counts, not saved binary indices."""
    settings = manifest["settings"]
    vocabulary = read(root / "vocabulary.json")
    sentences = sorted(tuple(s) for s in read(root / "sentences.json"))
    assert len(sentences) == len(set(sentences)) == 202172
    assert len(vocabulary) == manifest["vocabulary_size"] == 10864
    order = np.random.default_rng(settings["split_seed"]).permutation(len(sentences))
    ntest = round(len(sentences) * 0.2)
    heldout = set(order[:ntest].tolist())
    development = [s for i, s in enumerate(sentences) if i not in heldout]
    test = [s for i, s in enumerate(sentences) if i in heldout]
    causal = settings["objectives"] == ["causal"]

    def counts(pool):
        return Counter(word for sentence in pool for word in (sentence[1:] if causal else sentence))

    dev_counts = counts(development)

    def record(training, evaluated, train_groups, eval_groups):
        weights = np.array([training[w] + 1 for w in vocabulary], dtype=float)
        probability = weights / weights.sum()
        ranking = np.argsort(-probability, kind="stable")[:5]
        total = evaluated.total()
        return dict(
            train_examples=training.total(),
            examples=total,
            train_groups=train_groups,
            evaluated_groups=eval_groups,
            frequency=dict(
                nll=sum(
                    -evaluated[w] * np.log(p) for w, p in zip(vocabulary, probability, strict=True)
                )
                / total,
                top1=evaluated[vocabulary[ranking[0]]] / total,
                top5=sum(evaluated[vocabulary[i]] for i in ranking) / total,
            ),
        )

    order = np.random.default_rng(settings["split_seed"] + 1).permutation(len(development))
    folds = []
    for ids in np.array_split(order, settings["folds"]):
        validation = counts(development[i] for i in ids)
        folds.append(
            record(dev_counts - validation, validation, len(development) - len(ids), len(ids))
        )
    return dict(
        folds=folds,
        test=record(dev_counts, counts(test), len(development), len(test)),
    )


def paired_records(records, left, right, cv):
    def key(r):
        return (r["task"]["seed"], r["task"].get("fold", -1))

    a = {key(r): r for r in records if r["task"]["arm"]["name"] == left}
    b = {key(r): r for r in records if r["task"]["arm"]["name"] == right}
    assert a.keys() == b.keys()
    delta = {k: 100 * (a[k]["metrics"]["top1"] - b[k]["metrics"]["top1"]) for k in a}
    seed_means = [stats.mean(v for k, v in delta.items() if k[0] == s) for s in (44, 45, 46)]
    return dict(
        left=left,
        right=right,
        cv=cv,
        mean_top1_pp=stats.mean(delta.values()),
        positive_pairs=sum(v > 0 for v in delta.values()),
        pairs=len(delta),
        min_top1_pp=min(delta.values()),
        max_top1_pp=max(delta.values()),
        seed_mean_top1_pp=seed_means,
        leave_one_seed_out_top1_pp=[
            stats.mean(seed_means[:i] + seed_means[i + 1 :]) for i in range(3)
        ],
    )


def audit_study(root, source_hashes):
    manifest, selection = read(root / "manifest.json"), read(root / "selection.json")
    for value in (manifest, selection):
        assert value["fingerprint"] == fingerprint(
            {k: v for k, v in value.items() if k != "fingerprint"}
        )
    assert selection["manifest"] == manifest["fingerprint"]
    commit = manifest["provenance"]["commit"]
    assert manifest["code_sha256"] == source_hashes
    for source in manifest["corpus"]["sources"]:
        assert sha(Path(Path(source["path"]).name).read_bytes()) == source["sha256"]
    omitted = []
    for name, expected in manifest["artifact_sha256"].items():
        if (root / name).exists():
            assert sha((root / name).read_bytes()) == expected
        else:
            omitted.append(name)
    settings = manifest["settings"]
    confirm = settings["followup"]["stage"] == "confirm"
    for name, expected in dict(
        epochs=100,
        fixed_epochs=True,
        samples_per_epoch=20000,
        batch_size=64,
        microbatch_size=8,
        threads=4,
        window=4,
        split_seed=2026,
        final_eval_sentences=0,
        max_sentences=None,
    ).items():
        assert settings[name] == expected, name
    assert settings["datasets"] == ["phrases", "cleaned"]
    assert settings["seeds"] == ([44, 45, 46] if confirm else [43])
    assert settings["folds"] == (5 if confirm else 3)
    expected_counts = split_counts(root, manifest)
    records, histories = {}, {}
    for stage, tasks in (
        ("cv", manifest["tasks"]),
        ("refit", selection["refits"] if confirm else []),
    ):
        records[stage], histories[stage] = [], []
        assert len(list((root / stage).glob("task-*/result.json"))) == len(tasks)
        identities = [(t["arm"]["name"], t["seed"], t.get("fold", -1)) for t in tasks]
        assert len(set(identities)) == len(tasks)
        for i, task in enumerate(tasks):
            folder = root / stage / f"task-{i:04d}"
            result = read(folder / "result.json")
            assert result["manifest"] == manifest["fingerprint"] and result["task"] == task
            assert result["validation"] == (stage == "cv")
            assert result["trainable_parameters"] > 0
            assert result["selected_epoch"] == result["stopped_epoch"] == 100
            assert result["optimizer_updates"] == 31300 and result["training_draws"] == 2000000
            expected = (
                expected_counts["folds"][task["fold"]] if stage == "cv" else expected_counts["test"]
            )
            assert result["train_examples"] == expected["train_examples"]
            assert (
                result["metrics"]["examples"]
                == result["eligible_eval_examples"]
                == expected["examples"]
            )
            assert result["evaluated_sentences"] == expected["evaluated_groups"]
            for metric in METRICS:
                assert abs(result["frequency"][metric] - expected["frequency"][metric]) < 1e-12
                assert np.isfinite(result["metrics"][metric])
            history = read(folder / "history.json")
            assert [v["epoch"] for v in history] == list(range(0 if stage == "cv" else 1, 101))
            assert all(v["updates"] == v["epoch"] * 313 for v in history if v["epoch"])
            timing = read(folder / "timing-attempts" / f"{result['timing_attempt']}.json")
            assert timing["status"] == "complete"
            assert all(
                p["status"] == "complete" and np.isfinite(p["seconds"]) and p["seconds"] >= 0
                for p in timing["phases"]
            )
            runtime = read(folder / "runtime.json")
            assert runtime["device"] == "cuda" and runtime["threads"] == 4
            assert runtime["repository"]["commit"] == commit
            assert runtime["cuda_device"]["total_memory_bytes"] == 10200547328
            assert runtime["scheduler"]["CUDA_VISIBLE_DEVICES"].startswith("MIG-")
            records[stage].append(result)
            histories[stage].append(history)
    arms = []
    for candidate in selection["candidates"]:
        name = candidate["arm"]["name"]
        indexes = [i for i, r in enumerate(records["cv"]) if r["task"]["arm"]["name"] == name]
        fits = [records["cv"][i] for i in indexes]
        for metric in METRICS:
            values = [r["metrics"][metric] for r in fits]
            assert abs(candidate["mean"][metric] - stats.mean(values)) < 1e-12
            assert abs(candidate["sd"][metric] - stats.stdev(values)) < 1e-12
        curves = {
            e: {
                k: stats.mean(histories["cv"][i][e]["validation"][k] for i in indexes)
                for k in METRICS
            }
            for e in (0, 25, 50, 100)
        }
        arms.append(dict(**candidate, curves=curves, parameters=fits[0]["parameters"]))
    jobs = [read(p) for p in (root / "job-timings").glob("*.json")]
    assert len(jobs) == len(manifest["tasks"]) + 1 + (15 if confirm else 0)
    assert all(j["status"] == "complete" for j in jobs)
    assert all(
        p["status"] == "complete" and p["returncode"] == 0 for j in jobs for p in j["phases"]
    )
    ids = {j["job"] for j in jobs}
    with (root / "slurm-accounting.psv").open() as stream:
        accounting = [r for r in csv.DictReader(stream, delimiter="|") if r["JobID"] in ids]
    assert len(accounting) == len(ids) and {r["JobID"] for r in accounting} == ids
    assert all(r["State"] == "COMPLETED" and r["ExitCode"] == "0:0" for r in accounting)
    elapsed = {r["JobID"]: float(r["ElapsedRaw"]) for r in accounting}
    with (root / "timing-report/cv-fits.csv").open() as stream:
        costs = list(csv.DictReader(stream))
    assert [int(r["task_id"]) for r in costs] == list(range(len(manifest["tasks"])))
    for i, row in enumerate(costs):
        attempts = [
            j
            for j in jobs
            if j["stage"] in ("pilot", "fit") and j["task_id"] == i and not j["reused_result"]
        ]
        expected = sum(
            elapsed[j["job"]] - sum(p["seconds"] for p in j["phases"] if p["name"] in CALIBRATION)
            for j in attempts
        )
        assert abs(float(row["seconds"]) - expected) < 1e-7
        assert row["duration_source"] == "slurm"
    for arm in arms:
        values = [float(r["seconds"]) for r in costs if r["arm"] == arm["arm"]["name"]]
        arm["median_cv_seconds"] = stats.median(values)
    probes = read(next(root.glob("nlp-trainability-probe/*/summary.json")))
    assert probes["manifest"] == manifest["fingerprint"] and probes["code_sha256"] == source_hashes
    assert probes["runtime"]["device"] == "cuda" and len(probes["rows"]) == 16
    for q in (2, 4, 8, 12):
        rows = [r for r in probes["rows"] if r["qubits"] == q]
        assert len({r["common_state_sha256"] for r in rows}) == 1
        assert {(r["layers"], r["init_scale"], r["seeds"]) for r in rows} == {
            (depth, scale, 32) for depth in (2, 8) for scale in (0.05, 1.0)
        }
        assert all(np.isfinite(r["first_rx_gradient_variance"]) for r in rows)
    summary = dict(
        manifest=manifest["fingerprint"],
        commit=commit,
        settings=settings,
        omitted_artifacts=omitted,
        expected_counts=expected_counts,
        arms=arms,
        cv_fits=len(records["cv"]),
        refits=len(records["refit"]),
        allocations=len(jobs),
        cv_fit_hours=sum(float(r["seconds"]) for r in costs) / 3600,
        cv_projection_hours=read(root / "compute.json")["scenarios"][0]["cv_reference_hours"],
        nlp_probe=probes["rows"],
    )
    if confirm:
        report = read(root / "report/results.json")
        assert (
            report["manifest"] == manifest["fingerprint"]
            and report["selection"] == selection["fingerprint"]
        )
        assert len(report["refits"]) == 15 and len(report["summaries"]) == 5
        for r in report["refits"]:
            result = next(
                v
                for v in records["refit"]
                if v["task"]["arm"]["name"] == r["arm"] and v["task"]["seed"] == r["seed"]
            )
            assert all(r[k] == result["metrics"][k] for k in METRICS)
        summary["heldout"] = []
        for arm in report["summaries"]:
            fitted = [v for v in records["refit"] if v["task"]["arm"]["name"] == arm["arm"]]
            mu = {k: stats.mean(v["metrics"][k] for v in fitted) for k in METRICS}
            assert all(abs(arm["mean"][i] - mu[k]) < 1e-7 for i, k in ((1, "top1"), (2, "top5")))
            summary["heldout"].append(
                dict(
                    arm=arm["arm"],
                    ledger_mean=mu,
                    seed_sd={k: stats.stdev(v["metrics"][k] for v in fitted) for k in METRICS},
                    seed_metrics=[v["metrics"] for v in fitted],
                    exported_mean=arm["mean"],
                    nll_summary_minus_ledger=arm["mean"][0] - mu["nll"],
                )
            )
        summary["pairs"] = [
            paired_records(records[stage], left, right, stage == "cv")
            for stage in ("cv", "refit")
            for left, right in (PAIR, ("mlp-32", "bert-32"))
        ]
        summary["reported_intervals"] = report["paired_intervals"]
        for pair in report["paired_intervals"]:
            assert pair["draws"] == 2000 and pair["seeds"] == 3 and pair["sentences"] == 40434
            assert pair["confidence"] == (0.975 if pair["primary"] else 0.95)
        primary = [p for p in report["paired_intervals"] if p["primary"]]
        assert len(primary) == 1 and (primary[0]["left"], primary[0]["right"]) == PAIR
        summary["test_frequency"] = records["refit"][0]["frequency"]
    return summary, accounting, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Extracted outputs/robust-followup folder")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/robust-followup"))
    args = parser.parse_args()
    roots = {
        stage: next((args.root / stage).glob("*/manifest.json")).parent
        for stage in ("refine-causal", "refine-cbow", "confirm-causal", "confirm-cbow")
    }
    first = read(roots["refine-causal"] / "manifest.json")
    sources = first["code_sha256"]
    for name, expected in sources.items():
        assert (
            sha(subprocess.check_output(["git", "show", f"{first['provenance']['commit']}:{name}"]))
            == expected
        )
    studies, allocations, all_records = {}, {}, []
    for stage, root in roots.items():
        studies[stage], accounting, records = audit_study(root, sources)
        for row in accounting:
            assert row["JobID"] not in allocations
            allocations[row["JobID"]] = dict(study=stage, **row)
        all_records += [
            dict(study=stage, stage=k, **r) for k, rows in records.items() for r in rows
        ]
    # The actual LR winners must follow the parent selection, including the common paired LR.
    lr_names = [
        [
            "bert-32-lr-0.0003",
            "mlp-32-lr-0.0003",
            "hybrid-q8-lr-0.0003",
            "hybrid-q8-identity-lr-0.0003",
        ],
        ["bert-32", "mlp-32", *PAIR],
        ["bert-32-lr-0.01", "mlp-32-lr-0.01", "hybrid-q8-lr-0.01", "hybrid-q8-identity-lr-0.01"],
    ]
    for objective in ("causal", "cbow"):
        refine = read(roots[f"refine-{objective}"] / "selection.json")
        confirm = studies[f"confirm-{objective}"]["settings"]
        assert confirm["followup"]["parent_selection"] == refine["fingerprint"]
        candidates = {c["arm"]["name"]: c["mean"] for c in refine["candidates"]}

        def rank(names, candidates=candidates):
            return (
                -sum(candidates[n]["top1"] for n in names),
                -sum(candidates[n]["top5"] for n in names),
                sum(candidates[n]["nll"] for n in names),
                tuple(names),
            )

        pair = min((names[2:] for names in lr_names), key=rank)
        bert = min(([names[0]] for names in lr_names), key=rank)[0]
        mlp = min(([names[1]] for names in lr_names), key=rank)[0]
        assert confirm["arms"] == ["qcse-context-angular", *pair, mlp, bert]
        assert confirm["refit_controls"] == confirm["arms"]
        assert confirm["primary_comparisons"] == [list(PAIR)]
    driver = (args.root / "driver.log").read_text()
    assert driver.endswith(
        "Follow-up complete: 108 refinement CV fits, 150 confirmation CV fits, 30 refits.\n"
    )
    assert (
        driver.split("BENCHMARK_STAGE=refit", 1)[0].count("Frozen validation-only selection;") == 4
    )
    events = sorted(
        [(r["Start"], 1) for r in allocations.values()]
        + [(r["End"], -1) for r in allocations.values()]
    )
    active = peak = 0
    for _, change in events:
        active += change
        peak = max(peak, active)
    assert active == 0 and peak == 4 and len(allocations) == 292 and len(all_records) == 288
    start, end = (
        min(r["Start"] for r in allocations.values()),
        max(r["End"] for r in allocations.values()),
    )
    summary = dict(
        archive_sha256=sha(args.archive.read_bytes()),
        archive_bytes=args.archive.stat().st_size,
        source_hashes_checked=len(sources),
        studies=studies,
        complete_fits=len(all_records),
        training_draws=sum(r["training_draws"] for r in all_records),
        optimizer_updates=sum(r["optimizer_updates"] for r in all_records),
        successful_allocations=len(allocations),
        peak_concurrent_allocations=peak,
        slurm_local_start=start,
        slurm_local_end=end,
        wall_hours=(datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
        / 3600,
        allocation_hours=sum(float(r["ElapsedRaw"]) for r in allocations.values()) / 3600,
        scope="JSON ledgers, expected splits/frequency, source/parent/selection "
        "and accounting audited. "
        "Binary indices, checkpoints and predictions omitted; intervals retained, not regenerated. "
        "NLL summary float32 aggregation drift reproduced; ledger means reported provisionally.",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = [
        dict(
            study=r["study"],
            stage=r["stage"],
            arm=r["task"]["arm"]["name"],
            seed=r["task"]["seed"],
            fold=r["task"].get("fold"),
            **r["metrics"],
            parameters=r["parameters"],
            seconds=r["seconds"],
        )
        for r in all_records
    ]
    for filename, values, delimiter in (
        ("fits.csv", rows, ","),
        ("accounting.psv", list(allocations.values()), "|"),
    ):
        with (args.output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(values[0]), delimiter=delimiter, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(values)
    print(json.dumps({k: v for k, v in summary.items() if k != "studies"}, indent=2))


if __name__ == "__main__":
    main()
