"""Extrapolate measured batch costs; GPU speed factors are explicit scenarios."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from qcse.outputs import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-audit/compute.json"))
    args = parser.parse_args()
    manifest = json.loads((args.study / "manifest.json").read_text())
    profile = json.loads(args.profile.read_text())
    if profile["manifest"] != manifest["fingerprint"]:
        parser.error("Profile and prepared study differ")
    rows = {r["arm"]["name"]: r for r in profile["measurements"]}
    settings = manifest["settings"]
    if profile["batch"] != settings["batch_size"]:
        parser.error("Profile must use the planned optimizer batch size")
    steps = math.ceil(settings["samples_per_epoch"] / profile["batch"])
    final_counts = {}
    for objective in settings["objectives"]:
        with np.load(args.study / f"{objective}.npz") as data:
            final_counts[objective] = dict(
                validation=[len(data[f"val_eval_{f}"]) for f in range(settings["folds"])],
                test=len(data["test_eval"]),
            )
    scenarios = []
    for epochs in (10, 25, 50):
        fits, family_costs = [], {}
        for task in manifest["tasks"]:
            row = rows[task["arm"]["name"]]
            trained = epochs if row["trainable_parameters"] else 0
            evaluation = row["evaluation_batch_seconds"] / profile["batch"]
            training = steps * row["training_batch_seconds"] * trained
            counts = final_counts[task["objective"]]
            monitor = (trained + 1) * settings["eval_examples"] * evaluation
            final = counts["validation"][task["fold"]] * evaluation
            fits.append(training + monitor + final)
            # Winner unknown: use the costliest config in each family for refits.
            key = (task["objective"], task["arm"]["family"])
            refit = training + counts["test"] * evaluation
            family_costs[key] = max(family_costs.get(key, 0), refit)
        refit_seconds = sum(family_costs.values()) * len(settings["seeds"])
        total = sum(fits) + refit_seconds
        chunks = [sum(fits[:615]) / 5, sum(fits[615:]) / 5]
        scenarios.append(
            dict(
                trained_epochs=epochs,
                cv_reference_hours=sum(fits) / 3600,
                conservative_refit_reference_hours=refit_seconds / 3600,
                total_reference_hours=total / 3600,
                longest_cv_fit_reference_hours=max(fits) / 3600,
                ideal_ten_slot_reference_days=total / 3600 / 10 / 24,
                balanced_array_reference_days=(max(chunks) + refit_seconds / 10) / 86400,
                assumed_mig_scenarios=[
                    dict(
                        speedup_vs_profile=factor,
                        slot_hours=total / 3600 / factor,
                        ideal_ten_slot_days=total / 3600 / factor / 10 / 24,
                    )
                    for factor in ((1, 2, 5, 10) if profile["device"] == "cpu" else (1,))
                ],
            )
        )
    result = dict(
        manifest=manifest["fingerprint"],
        profile=str(args.profile),
        reference_device=profile["device"],
        scenarios=scenarios,
        assumptions=(
            "Causal profile used for both tracks (same tensor dimensions); batch medians, "
            "linear evaluation scaling; conservative unknown refit winners; no startup, "
            "hashing, preflight, checkpoint I/O, lost work, queue or scheduling overhead. "
            "MIG speed factors are unmeasured planning scenarios, not confidence bounds. "
            "615+615 chunks and five slots each describe the default 1230-task plan."
        ),
    )
    if len(manifest["tasks"]) != 1230:
        parser.error("Array wall-time projection currently describes the default 1230-task plan")
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
