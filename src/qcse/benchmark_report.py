"""Report frozen validation selections and paired held-out sentence intervals."""

import csv
from itertools import combinations

import numpy as np

from .benchmark import load_manifest, load_selection, paired_bootstrap, validate_result
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
        result = validate_result(root, manifest, task, output, False)
        with np.load(output / "predictions.npz") as archive:
            predictions[(task["objective"], task["arm"]["name"], task["seed"])] = {
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
                cuda_peak_reserved_bytes=result.get("cuda_peak_reserved_bytes"),
            )
        )
    comparisons, summaries = [], []
    for objective in manifest["settings"]["objectives"]:
        names = sorted({r["arm"] for r in rows if r["objective"] == objective})
        records_by_name = {}
        for name in names:
            records = [predictions[(objective, name, s)] for s in seeds]
            first = records[0]
            for record in records[1:]:
                for key in ("ids", "groups", "positions", "targets"):
                    if not np.array_equal(record[key], first[key]):
                        raise ValueError("Seed predictions are not aligned")
            records_by_name[name] = records
            values = np.stack([r["metrics"] for r in records])
            family = next(
                r["family"] for r in rows if r["objective"] == objective and r["arm"] == name
            )
            summaries.append(
                dict(
                    objective=objective,
                    family=family,
                    arm=name,
                    mean=values.mean((0, 1)).tolist(),
                    seed_sd=values.mean(1).std(0, ddof=1).tolist() if len(seeds) > 1 else None,
                    metric_order=["nll", "top1", "top5"],
                )
            )
            comparisons.append(
                dict(
                    objective=objective,
                    left=name,
                    right="frequency",
                    primary=False,
                    **paired_bootstrap(
                        first["groups"], values, np.stack([r["frequency"] for r in records]), draws
                    ),
                )
            )
        primary = manifest["settings"].get("primary_comparisons", [])
        pairs = [tuple(pair) for pair in primary]
        pairs += [
            pair for pair in combinations(names, 2) if not any(set(pair) == set(p) for p in primary)
        ]
        for left, right in pairs:
            a, b = records_by_name[left], records_by_name[right]
            for x, y in zip(a, b, strict=True):
                for key in ("ids", "groups", "positions", "targets"):
                    if not np.array_equal(x[key], y[key]):
                        raise ValueError("Models are not paired on the same test examples")
            is_primary = [left, right] in primary
            comparisons.append(
                dict(
                    objective=objective,
                    left=left,
                    right=right,
                    primary=is_primary,
                    **paired_bootstrap(
                        a[0]["groups"],
                        np.stack([r["metrics"] for r in a]),
                        np.stack([r["metrics"] for r in b]),
                        draws,
                        confidence=manifest["settings"].get("primary_confidence", 0.95)
                        if is_primary
                        else 0.95,
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
            interval_scope="Paired seed/sentence percentile intervals at the recorded confidence. "
            "Primary confidence is prespecified; secondary intervals are exploratory. "
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
        "| Track | Configuration | Test NLL | Test top-1 | Test top-5 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        nll, top1, top5 = row["mean"]
        text.append(f"| {row['objective']} | {row['arm']} | {nll:.4f} | {top1:.2%} | {top5:.2%} |")
    text.extend(
        [
            "",
            "## Paired differences",
            "",
            "Percentile intervals resample paired seeds and whole sentences.",
            "Positive top-1 favors the left model. Negative NLL favors the left model.",
            "Intervals are conditional on selection; three seeds do not establish "
            "stability across all training runs.",
            (
                "Prespecified primary contrasts use "
                f"{manifest['settings'].get('primary_confidence', 0.95):.1%} intervals; "
                "secondary 95% intervals are uncorrected. "
                if manifest["settings"].get("primary_comparisons")
                else "No primary contrast was prespecified; all 95% intervals are exploratory. "
            )
            + "Fold SD is descriptive, not an independent-fold CI.",
            "",
            "| Track | Left − right | Top-1 difference | Confidence | Interval |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in comparisons:
        text.append(
            f"| {row['objective']} | {row['left']} − {row['right']} | "
            f"{row['difference'][1]:+.2%} | {row['confidence']:.1%} | "
            f"[{row['low'][1]:+.2%}, {row['high'][1]:+.2%}] |"
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
