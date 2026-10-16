"""Prepare focused LR/circuit studies, then freeze their confirmation controls."""

import argparse
import json
from pathlib import Path

from qcse.benchmark import load_manifest, load_selection, prepare
from qcse.outputs import new_run_directory

LR_ARMS = {
    0.0003: [
        "bert-32-lr-0.0003",
        "mlp-32-lr-0.0003",
        "hybrid-q8-lr-0.0003",
        "hybrid-q8-identity-lr-0.0003",
    ],
    0.003: ["bert-32", "mlp-32", "hybrid-q8-depth2-init0.05", "hybrid-q8-identity"],
    0.01: ["bert-32-lr-0.01", "mlp-32-lr-0.01", "hybrid-q8-lr-0.01", "hybrid-q8-identity-lr-0.01"],
}


def confirmed_arms(root, objective):
    manifest, selected = load_manifest(root), load_selection(root)
    if manifest["settings"].get("followup", {}).get("stage") != "refine":
        raise ValueError("Confirmation requires the completed follow-up LR study")
    candidates = {
        c["arm"]["name"]: c for c in selected["candidates"] if c["objective"] == objective
    }

    def rank(names):
        values = [candidates[name]["mean"] for name in names]
        return (
            -sum(v["top1"] for v in values),
            -sum(v["top5"] for v in values),
            sum(v["nll"] for v in values),
            tuple(names),
        )

    # Choose one common LR for the paired circuit/identity contrast.
    pair = min((names[2:] for names in LR_ARMS.values()), key=rank)
    bert = min(([names[0]] for names in LR_ARMS.values()), key=rank)[0]
    mlp = min(([names[1]] for names in LR_ARMS.values()), key=rank)[0]
    return ["qcse-context-angular", *pair, mlp, bert], selected["fingerprint"]


def settings_for(stage, objective, source=None):
    if stage == "refine":
        arms = [name for names in LR_ARMS.values() for name in names]
        arms += [
            "hybrid-q8-local",
            "hybrid-q8-frozen",
            "hybrid-q8-xx",
            "hybrid-q8-zz",
            "mean-init-0.1",
            "mean",
        ]
        parent = None
    else:
        if source is None:
            raise ValueError("This stage needs --from-study with its frozen selection")
        arms, parent = confirmed_arms(source, objective)
    confirm = stage == "confirm"
    settings = dict(
        objectives=[objective],
        arms=arms,
        refit_controls=arms if confirm else [],
        trainability=True,
        max_sentences=None,
        datasets=["phrases", "cleaned"],
        window=4,
        folds=5 if confirm else 3,
        seeds=[44, 45, 46] if confirm else [43],
        split_seed=2026,
        epochs=100,
        fixed_epochs=True,
        samples_per_epoch=20000,
        batch_size=64,
        microbatch_size=8,
        eval_examples=2048,
        final_eval_sentences=0,
        patience=8,
        min_epochs=10,
        threads=4,
        followup=dict(
            stage=stage,
            parent_selection=parent,
            screen_manifest="9c6e04456c1db3d7857e5d62e398d4d5a343d5fe1c8e2e6f6361330c32fdb1ee",
        ),
        primary_comparisons=[arms[1:3]] if confirm else [],
        primary_confidence=0.975,
        planning_factor=1.10,
    )
    if stage in ("window-2", "window-8"):
        settings["window"] = int(stage.split("-")[1])
    elif stage in ("phrases", "cleaned"):
        settings["datasets"] = [stage]
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("refine", "confirm", "window-2", "window-8", "phrases", "cleaned")
    )
    parser.add_argument("--objective", choices=("causal", "cbow"), default="causal")
    parser.add_argument("--from-study", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/robust-followup"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        settings = settings_for(args.stage, args.objective, args.from_study)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(settings, indent=2))
        return
    folder = args.output / f"{args.stage}-{args.objective}"
    with new_run_directory(folder, "qcse-bert", dict(stage="prepare")) as root:
        manifest = prepare(root, settings)
        print(f"Prepared {len(manifest['tasks'])} CV tasks at {root}")


if __name__ == "__main__":
    main()
