"""Describe measured CV costs and replay the same arms under a fixed slot limit."""

import argparse
import csv
import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np

from qcse.benchmark import load_manifest
from qcse.outputs import atomic_path, write_json

EPOCH_PHASES = {"training", "validation", "selection", "checkpoint"}


def schedule_seconds(costs, concurrency):
    """Same balanced array chunks and combined slot cap as the submitter."""
    chunks = math.ceil(len(costs) / 1000)
    slots = concurrency // chunks
    if slots < 1:
        raise ValueError("Too many array chunks for the concurrency budget; split the study")
    size = math.ceil(len(costs) / chunks)
    finishes = []
    for offset in range(0, len(costs), size):
        queue = [0.0] * slots
        for seconds in costs[offset : offset + size]:
            heapq.heapreplace(queue, queue[0] + seconds)
        finishes.append(max(queue))
    return max(finishes)


def timing_report(root, concurrency=4, fits_per_arm=None, accounting=None):
    if not 1 <= concurrency <= 4 or (fits_per_arm is not None and fits_per_arm < 1):
        raise ValueError("Use concurrency 1..4 and a positive fit count")
    manifest = load_manifest(root)
    jobs = [json.loads(p.read_text()) for p in (root / "job-timings").glob("*.json")]
    elapsed = {}
    if accounting is not None:
        with accounting.open(newline="") as stream:
            reader = csv.DictReader(stream, delimiter="|")
            if not {"JobID", "ElapsedRaw"} <= set(reader.fieldnames or []):
                raise ValueError("Accounting needs pipe-separated JobID and ElapsedRaw columns")
            for row in reader:
                if row["ElapsedRaw"]:
                    elapsed[row["JobID"]] = float(row["ElapsedRaw"])
    rows, devices = [], set()
    for index, task in enumerate(manifest["tasks"]):
        output = root / "cv" / f"task-{index:04d}"
        if not (output / "result.json").exists():
            continue
        result = json.loads((output / "result.json").read_text())
        if result["manifest"] != manifest["fingerprint"] or result["task"] != task:
            raise ValueError(f"Result differs from the prepared task: {output}")
        runtime = json.loads((output / "runtime.json").read_text())
        devices.add(
            (
                runtime["device"],
                runtime.get("device_name", runtime["device"]),
                runtime.get("cuda_device", {}).get("total_memory_bytes", 0),
                runtime.get("threads", 0),
                runtime.get("torch", "unknown"),
            )
        )
        phases = [
            phase
            for p in (output / "timing-attempts").glob("*.json")
            for phase in json.loads(p.read_text())["phases"]
            if phase["status"] == "complete"
        ]
        if any(
            p["seconds"] is None or not math.isfinite(p["seconds"]) or p["seconds"] < 0
            for p in phases
        ):
            raise ValueError(f"Invalid completed phase timing: {output}")
        totals = {
            name: sum(p["seconds"] for p in phases if p["name"] == name)
            for name in {p["name"] for p in phases}
        }
        epochs = sum(p["name"] == "training" for p in phases)
        epoch_seconds = (
            sum(v for k, v in totals.items() if k in EPOCH_PHASES) / epochs if epochs else None
        )
        attempts = [
            j
            for j in jobs
            if j["stage"] in ("pilot", "fit")
            and j["task_id"] == index
            and j["status"] in ("complete", "failed", "interrupted")
            and not j["reused_result"]
        ]
        seconds = (
            sum(
                elapsed.get(j["job"], j["seconds"])
                - sum(
                    p["seconds"]
                    for p in j["phases"]
                    if p["name"]
                    in ("profile", "estimate", "trainability_probe", "nlp_trainability_probe")
                    and p["seconds"] is not None
                )
                for j in attempts
            )
            if attempts
            else result["seconds"]
        )
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"Invalid fit duration: {output}")
        cap = manifest["settings"]["epochs"]
        cap_seconds = (
            seconds + max(0, cap - result["stopped_epoch"]) * epoch_seconds
            if epoch_seconds is not None
            else (seconds if not result["trainable_parameters"] else None)
        )
        history_path = output / "history.json"
        if history_path.exists() or result["trainable_parameters"] or result["stopped_epoch"]:
            history = json.loads(history_path.read_text())
        else:
            # Parameter-free fits never checkpoint and have no optimizer trajectory.
            history = []
        clipping = [r["clipped_batch_fraction"] for r in history if "clipped_batch_fraction" in r]
        rows.append(
            dict(
                task_id=index,
                objective=task["objective"],
                arm=task["arm"]["name"],
                fold=task["fold"],
                seed=task["seed"],
                selected_epoch=result["selected_epoch"],
                stopped_epoch=result["stopped_epoch"],
                seconds=seconds,
                duration_source="slurm"
                if attempts and all(j["job"] in elapsed for j in attempts)
                else ("job_timer" if attempts else "fit_timer"),
                epoch_seconds=epoch_seconds,
                epoch_cap_seconds=cap_seconds,
                training_seconds=totals.get("training"),
                validation_seconds=totals.get("validation"),
                checkpoint_seconds=totals.get("checkpoint"),
                clipped_batch_fraction=float(np.mean(clipping)) if clipping else None,
                cuda_peak_bytes=result.get("cuda_peak_bytes"),
                cuda_peak_reserved_bytes=result.get("cuda_peak_reserved_bytes"),
                cuda_total_memory_bytes=runtime.get("cuda_device", {}).get("total_memory_bytes"),
                nll=result["metrics"]["nll"],
                top1=result["metrics"]["top1"],
            )
        )
    if not rows:
        raise ValueError("No completed CV fits yet")
    arms = list(dict.fromkeys((t["objective"], t["arm"]["name"]) for t in manifest["tasks"]))
    summaries, scenarios = [], []
    for objective, arm in arms:
        group = [r for r in rows if (r["objective"], r["arm"]) == (objective, arm)]
        if not group:
            continue
        summaries.append(
            dict(
                objective=objective,
                arm=arm,
                complete=len(group),
                planned=sum(
                    (t["objective"], t["arm"]["name"]) == (objective, arm)
                    for t in manifest["tasks"]
                ),
                seconds_p10_p50_p90=np.quantile(
                    [r["seconds"] for r in group], [0.1, 0.5, 0.9]
                ).tolist(),
                median_stopped_epoch=float(np.median([r["stopped_epoch"] for r in group])),
                phases_seconds={
                    k: float(np.mean([r[k] for r in group if r[k] is not None]))
                    for k in ("training_seconds", "validation_seconds", "checkpoint_seconds")
                    if any(r[k] is not None for r in group)
                },
            )
        )
    if len(summaries) == len(arms) and len(devices) == 1:
        for label, key, quantile in (
            ("observed_lengths_median", "seconds", 0.5),
            ("observed_lengths_p90", "seconds", 0.9),
            ("epoch_cap_median", "epoch_cap_seconds", 0.5),
        ):
            costs = []
            for objective, arm in arms:
                group = [r[key] for r in rows if (r["objective"], r["arm"]) == (objective, arm)]
                if any(v is None for v in group):
                    break
                count = fits_per_arm or sum(
                    (t["objective"], t["arm"]["name"]) == (objective, arm)
                    for t in manifest["tasks"]
                )
                costs.extend([float(np.quantile(group, quantile))] * count)
            else:
                scenarios.append(
                    dict(
                        name=label,
                        cv_fits=len(costs),
                        job_hours=sum(costs) / 3600,
                        scheduled_hours=schedule_seconds(costs, concurrency) / 3600,
                    )
                )
    output = root / "timing-report"
    output.mkdir(exist_ok=True)
    data = dict(
        manifest=manifest["fingerprint"],
        report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        complete=len(rows),
        planned=len(manifest["tasks"]),
        concurrency=concurrency,
        devices=sorted(devices),
        arms=summaries,
        scenarios=scenarios,
        scope=(
            "Additional CV fits of these same arms and protocol only; excludes selected refits, "
            "queue waits and human handoff. P10/P90 are descriptive quantiles, not confidence "
            "or prediction bounds. Assumes slots stay available and task-order scheduling; "
            "contention may change. Epoch-cap projection holds per-epoch cost fixed. "
            "Fit-timer fallback excludes allocation preflights/startup. "
            "Partial/interrupted attempts can have incomplete timing records."
        ),
    )
    write_json(output / "summary.json", data)
    with atomic_path(output / "cv-fits.csv") as p, p.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "# Measured benchmark compute",
        "",
        f"Completed CV fits: {len(rows)}/{len(manifest['tasks'])}; concurrency: {concurrency}.",
        "",
        "| Track | Arm | Fits | Fit seconds P10 / median / P90 | Median epochs |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for r in summaries:
        times = " / ".join(f"{v:.1f}" for v in r["seconds_p10_p50_p90"])
        text.append(
            f"| {r['objective']} | {r['arm']} | {r['complete']}/{r['planned']} | "
            f"{times} | {r['median_stopped_epoch']:g} |"
        )
    text.extend(["", "## Same-arm CV budget scenarios", ""])
    for r in scenarios:
        text.append(
            f"- {r['name']}: {r['cv_fits']} fits, {r['job_hours']:.2f} job-hours, "
            f"{r['scheduled_hours']:.2f} hours with {concurrency} slots."
        )
    text.extend(
        [
            "",
            data["scope"],
            "",
            "See cv-fits.csv for measured phase costs, clipping frequency and GPU memory. "
            "Inspect history.json and timing-attempts/ before attributing a plateau or "
            "bottleneck. A larger microbatch is a profiling candidate when memory permits; "
            "speedup and stochastic equivalence require measurement. "
            "New circuits/qubit counts require fresh pilots.",
        ]
    )
    with atomic_path(output / "report.md") as p:
        p.write_text("\n".join(text) + "\n")
    return data


def main():
    """Allow a copied reporter to analyze an unchanged, source-locked study."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("study", type=Path)
    cli.add_argument("--concurrency", type=int, choices=range(1, 5), default=4)
    cli.add_argument("--fits-per-arm", type=int)
    cli.add_argument("--slurm-accounting", type=Path)
    args = cli.parse_args()
    timing_report(args.study.resolve(), args.concurrency, args.fits_per_arm, args.slurm_accounting)
    print(args.study / "timing-report" / "report.md")


if __name__ == "__main__":
    main()
