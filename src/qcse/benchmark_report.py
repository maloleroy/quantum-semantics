"""Report frozen validation selections and paired held-out sentence intervals."""

import csv
import json

import numpy as np

from .benchmark import load_manifest, load_selection, paired_bootstrap
from .outputs import atomic_path, write_json


def report(root, draws=2000):
    manifest, selected = load_manifest(root), load_selection(root)
    seeds = manifest["settings"]["seeds"]
    rows, predictions = [], {}
    for index, task in enumerate(selected["refits"]):
        output = root / "refit" / f"task-{index:04d}"
        path = output / "result.json"
        if not path.exists():
            raise ValueError(f"Refit incomplete: {path}")
        result = json.loads(path.read_text())
        if (
            result["manifest"] != manifest["fingerprint"]
            or result["task"] != task
            or result["validation"]
        ):
            raise ValueError(f"Invalid held-out result: {path}")
        with np.load(output / "predictions.npz") as archive:
            predictions[(task["objective"], task["arm"]["family"], task["seed"])] = {
                k: archive[k].copy() for k in archive.files
            }
        rows.append(
            dict(
                objective=task["objective"],
                family=task["arm"]["family"],
                arm=task["arm"]["name"],
                seed=task["seed"],
                **result["metrics"],
                epoch=result["selected_epoch"],
                parameters=result["parameters"],
                trainable_parameters=result["trainable_parameters"],
                updates=result["optimizer_updates"],
                seconds=result["seconds"],
                cuda_peak_bytes=result["cuda_peak_bytes"],
            )
        )
    comparisons, summaries = [], []
    for objective in manifest["settings"]["objectives"]:
        families = sorted({r["family"] for r in rows if r["objective"] == objective})
        for family in families:
            records = [predictions[(objective, family, s)] for s in seeds]
            first = records[0]
            for record in records[1:]:
                for key in ("ids", "groups", "positions", "targets"):
                    if not np.array_equal(record[key], first[key]):
                        raise ValueError("Seed predictions are not aligned")
            values = np.stack([r["metrics"] for r in records])
            summaries.append(
                dict(
                    objective=objective,
                    family=family,
                    mean=values.mean((0, 1)).tolist(),
                    seed_sd=values.mean(1).std(0, ddof=1).tolist() if len(seeds) > 1 else None,
                    metric_order=["nll", "top1", "top5"],
                )
            )
            frequency = np.stack([r["frequency"] for r in records])
            comparisons.append(
                dict(
                    objective=objective,
                    left=family,
                    right="frequency",
                    **paired_bootstrap(first["groups"], values, frequency, draws),
                )
            )
            if family != "bert" and "bert" in families:
                bert = [predictions[(objective, "bert", s)] for s in seeds]
                for left, right in zip(records, bert, strict=True):
                    for key in ("ids", "groups", "positions", "targets"):
                        if not np.array_equal(left[key], right[key]):
                            raise ValueError("Models are not paired on the same test examples")
                comparisons.append(
                    dict(
                        objective=objective,
                        left=family,
                        right="bert",
                        **paired_bootstrap(
                            first["groups"], values, np.stack([r["metrics"] for r in bert]), draws
                        ),
                    )
                )
    output = root / "report"
    output.mkdir(exist_ok=True)
    write_json(
        output / "results.json",
        dict(
            manifest=manifest["fingerprint"],
            selection=selected["fingerprint"],
            refits=rows,
            summaries=summaries,
            paired_intervals=comparisons,
            interval_scope="Exploratory 95% paired seed/sentence percentile intervals; "
            "no multiplicity correction; "
            "conditional on this fixed split, selected models and evaluated sentences.",
        ),
    )
    with atomic_path(output / "refits.csv") as path, path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "# QCSE/BERT held-out results",
        "",
        "Configurations and refit epochs were frozen using all development folds and seeds.",
        "Top-k uses vocabulary logits for every model; NLL is normalized over valid words.",
        "",
        "| Track | Family | Test NLL | Test top-1 | Test top-5 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        nll, top1, top5 = row["mean"]
        text.append(
            f"| {row['objective']} | {row['family']} | {nll:.4f} | {top1:.2%} | {top5:.2%} |"
        )
    text.extend(
        [
            "",
            "## Paired differences",
            "",
            "Exploratory 95% percentile intervals resample paired seeds and whole sentences.",
            "Positive top-1 favors the left model. Negative NLL favors the left model.",
            "Intervals are conditional on selection; three seeds do not establish "
            "stability across all training runs.",
            "No correction for multiple comparisons is applied. "
            "Fold SD is descriptive, not an independent-fold CI.",
            "",
            "| Track | Left − right | Top-1 difference | 95% interval |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in comparisons:
        text.append(
            f"| {row['objective']} | {row['left']} − {row['right']} | "
            f"{row['difference'][1]:+.2%} | [{row['low'][1]:+.2%}, {row['high'][1]:+.2%}] |"
        )
    text.extend(
        [
            "",
            "See results.json for NLL/top-5 intervals and refits.csv "
            "for capacity, updates, timing and memory.",
        ]
    )
    with atomic_path(output / "report.md") as path:
        path.write_text("\n".join(text) + "\n")
    print(output / "report.md")
