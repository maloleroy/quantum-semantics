"""Extrapolate measured batch costs; GPU speed factors are explicit scenarios."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from qcse.benchmark_compute_report import schedule_seconds
from qcse.outputs import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-audit/compute.json"))
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=4)
    parser.add_argument(
        "--job-overhead-seconds",
        type=float,
        default=20.0,
        help="Fixed per-job allowance; the completed focused DGX study measured about 20 seconds",
    )
    args = parser.parse_args()
    if not math.isfinite(args.job_overhead_seconds) or args.job_overhead_seconds < 0:
        parser.error("Job overhead must be finite and nonnegative")
    manifest = json.loads((args.study / "manifest.json").read_text())
    profile = json.loads(args.profile.read_text())
    if profile["manifest"] != manifest["fingerprint"]:
        parser.error("Profile and prepared study differ")
    rows = {r["arm"]["name"]: r for r in profile["measurements"]}
    settings = manifest["settings"]
    if profile["batch"] != settings["batch_size"]:
        parser.error("Profile must use the planned optimizer batch size")
    if (
        profile["microbatch"] != settings["microbatch_size"]
        or profile["threads"] != settings["threads"]
        or profile["effective_source_sha256"] != manifest["code_sha256"]
        or profile["objective"] not in settings["objectives"]
    ):
        parser.error("Profile must match the prepared source, microbatch, threads and objective")
    if not {t["arm"]["name"] for t in manifest["tasks"]} <= rows.keys():
        parser.error("Profile must include every prepared arm")
    steps = math.ceil(settings["samples_per_epoch"] / profile["batch"])
    final_counts = {}
    for objective in settings["objectives"]:
        with np.load(args.study / f"{objective}.npz") as data:
            final_counts[objective] = dict(
                validation=[len(data[f"val_eval_{f}"]) for f in range(settings["folds"])],
                test=len(data["test_eval"]),
            )
    scenarios = []
    epoch_budgets = (
        (settings["epochs"],)
        if settings.get("fixed_epochs", False)
        else sorted({10, 25, settings["epochs"]})
    )
    for epochs in epoch_budgets:
        fits, family_costs = [], {}
        configuration_costs = {}
        for task in manifest["tasks"]:
            row = rows[task["arm"]["name"]]
            trained = epochs if row["trainable_parameters"] else 0
            evaluation = row["evaluation_batch_seconds"] / profile["batch"]
            training = steps * row["training_batch_seconds"] * trained
            counts = final_counts[task["objective"]]
            monitor = (trained + 1) * settings["eval_examples"] * evaluation
            final = counts["validation"][task["fold"]] * evaluation
            cost = training + monitor + final + args.job_overhead_seconds
            fits.append(cost)
            configuration_costs.setdefault((task["objective"], task["arm"]["name"]), []).append(
                cost
            )
            # Winner unknown: use the costliest config in each family for refits.
            key = (task["objective"], task["arm"]["family"])
            refit = training + counts["test"] * evaluation + args.job_overhead_seconds
            family_costs[key] = max(family_costs.get(key, 0), refit)
            if task["arm"]["name"] in settings.get("refit_controls", []):
                control_key = (task["objective"], "control:" + task["arm"]["name"])
                family_costs[control_key] = max(family_costs.get(control_key, 0), refit)
        refit_costs = [cost for _, cost in sorted(family_costs.items()) for _ in settings["seeds"]]
        refit_seconds = sum(refit_costs)
        total = sum(fits) + refit_seconds
        chunks = math.ceil(len(fits) / 1000)
        limit = args.concurrency // chunks
        if limit < 1:
            parser.error("Concurrency must cover every independent array chunk")
        scheduled_cv = schedule_seconds(fits, args.concurrency)
        scheduled_refit = schedule_seconds(refit_costs, args.concurrency)
        scenarios.append(
            dict(
                trained_epochs=epochs,
                cv_reference_hours=sum(fits) / 3600,
                conservative_refit_reference_hours=refit_seconds / 3600,
                conservative_refit_tasks=len(family_costs) * len(settings["seeds"]),
                total_reference_hours=total / 3600,
                longest_cv_fit_reference_hours=max(fits) / 3600,
                configurations=[
                    dict(
                        objective=objective,
                        arm=arm,
                        cv_fits=len(costs),
                        median_fit_reference_hours=float(np.median(costs)) / 3600,
                        max_fit_reference_hours=max(costs) / 3600,
                        total_cv_reference_hours=sum(costs) / 3600,
                    )
                    for (objective, arm), costs in sorted(configuration_costs.items())
                ],
                ideal_slot_reference_days=total / 3600 / args.concurrency / 24,
                balanced_array_reference_days=(scheduled_cv + scheduled_refit) / 86400,
                pilot_then_cv_reference_hours=(
                    fits[0]
                    + schedule_seconds([args.job_overhead_seconds] + fits[1:], args.concurrency)
                )
                / 3600,
                assumed_mig_scenarios=[
                    dict(
                        speedup_vs_profile=factor,
                        slot_hours=total / 3600 / factor,
                        ideal_slot_days=total / 3600 / factor / args.concurrency / 24,
                    )
                    for factor in ((1, 2, 5, 10) if profile["device"] == "cpu" else (1,))
                ],
            )
        )
    result = dict(
        manifest=manifest["fingerprint"],
        profile=str(args.profile),
        reference_device=profile["device"],
        reference_objective=profile.get("objective", "causal"),
        concurrency=args.concurrency,
        job_overhead_seconds=args.job_overhead_seconds,
        scenarios=scenarios,
        assumptions=(
            "Profiled objective used for requested tracks (same tensor dimensions); batch medians, "
            "linear evaluation scaling; conservative unknown refit winners and extra controls "
            "(a control winning its family may be counted twice); "
            "stated per-job overhead allowance. "
            "Checkpoint I/O, lost work, queue waits and scheduling tails omitted. "
            "MIG speed factors are unmeasured planning scenarios, not confidence bounds. "
            "Balanced arrays assume maximum size 1000, available slots and task-order scheduling. "
            "The pilot-then-CV estimate includes one reused-task allowance but excludes pilot "
            "profiling/estimation and human handoff."
        ),
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
