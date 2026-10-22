"""Independently audit prediction aggregates and resample paired seeds/sentences.

Uses compact exported sufficient statistics; no model imports or training.
Raw logits, top-k IDs and circuit gradients remain outside this review.
"""

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from itertools import combinations, product
from pathlib import Path

import numpy as np

METRICS = ("nll", "top1", "top5")


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bootstrap(sums, counts, draws=2000, seed=42, chunk=32):
    """sums=(populations, models, seeds, groups, metrics); counts=(populations, groups).

    Multinomial group/seed multiplicities replace repeated row indexing.
    Return paired and sentence-only samples using the same group draws.
    All populations must share ordered group and seed identities.
    """
    populations, models, seeds, groups, metrics = sums.shape
    if counts.shape != (populations, groups) or draws < 1 or chunk < 1:
        raise ValueError("Invalid dimensions or resampling budget")
    if not np.isfinite(sums).all() or not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("Need finite sums and positive group counts")
    flat = np.ascontiguousarray(np.moveaxis(sums, 3, 0).reshape(groups, -1))
    paired = np.empty((draws, populations, models, metrics))
    sentences = np.empty_like(paired)
    rng = np.random.default_rng(seed)
    for start in range(0, draws, chunk):
        size = min(chunk, draws - start)
        weights, seed_weights = np.empty((size, groups)), np.empty((size, seeds))
        for i in range(size):
            weights[i] = np.bincount(rng.integers(groups, size=groups), minlength=groups)
            seed_weights[i] = np.bincount(rng.integers(seeds, size=seeds), minlength=seeds)
        totals = (weights @ flat).reshape(size, populations, models, seeds, metrics)
        denominator = (weights @ counts.T)[:, :, None, None] * seeds
        paired[start : start + size] = (totals * seed_weights[:, None, None, :, None]).sum(
            3
        ) / denominator
        sentences[start : start + size] = totals.sum(3) / denominator
    return paired, sentences


def interval(samples, confidence):
    return dict(
        low=np.quantile(samples, (1 - confidence) / 2, axis=0).tolist(),
        high=np.quantile(samples, (1 + confidence) / 2, axis=0).tolist(),
    )


def expected_population(study, arrays, objective):
    sentences, vocabulary = read(study / "sentences.json"), read(study / "vocabulary.json")
    lookup = {w: i for i, w in enumerate(vocabulary)}
    keys = sorted(tuple(s) for s in sentences)
    order = np.random.default_rng(2026).permutation(len(keys))
    heldout = {keys[i] for i in order[: round(0.2 * len(keys))]}
    ids = [i for i, s in enumerate(sentences) if tuple(s) in heldout]
    assert np.array_equal(arrays["groups"], ids)
    lengths = np.array([len(sentences[i]) for i in ids])
    assert np.array_equal(arrays["counts"], lengths - (objective == "causal"))
    assert np.array_equal(arrays["noninitial_counts"], lengths - 1)
    targets = np.array(
        [(i, j, lookup[sentences[i][j]]) for i in ids for j in range(1, len(sentences[i]))],
        dtype="<i8",
    )
    assert sha(targets.tobytes()) == str(arrays["noninitial_keys_sha256"])
    train = Counter(
        w
        for s in sentences
        if tuple(s) not in heldout
        for w in (s[1:] if objective == "causal" else s)
    )
    counts = np.array([train[w] + 1 for w in vocabulary], dtype=float)
    probabilities = counts / counts.sum()
    top = np.argsort(-probabilities, kind="stable")[:5]
    expected, common = [], []
    for group in ids:
        target = np.array([lookup[w] for w in sentences[group]])
        values = np.column_stack(
            (-np.log(probabilities[target]), target == top[0], np.isin(target, top))
        )
        expected.append(values[1:].sum(0) if objective == "causal" else values.sum(0))
        common.append(values[1:].sum(0))
    for name, values in (("frequency_sums", expected), ("noninitial_frequency_sums", common)):
        for seed in arrays[name]:
            np.testing.assert_allclose(seed, values, atol=1e-11, rtol=1e-12)
    train_first = Counter(s[0] for s in sentences if tuple(s) not in heldout)
    test_first = Counter(sentences[i][0] for i in ids)
    first_probability = np.array([train_first[w] + 1 for w in vocabulary], dtype=float)
    first_probability /= first_probability.sum()
    first_top = np.argsort(-first_probability, kind="stable")[:5]
    first_prior = dict(
        nll=sum(
            -test_first[w] * np.log(p) for w, p in zip(vocabulary, first_probability, strict=True)
        )
        / len(ids),
        top1=test_first[vocabulary[first_top[0]]] / len(ids),
        top5=sum(test_first[vocabulary[i]] for i in first_top) / len(ids),
    )
    return dict(
        test_sentences=len(ids),
        common_targets=len(targets),
        common_target_sha256=sha(targets.tobytes()),
        first_words=Counter(sentences[i][0] for i in ids).most_common(10),
        first_position_frequency=first_prior,
    )


def load_export(root, old_root, objective, analyzer):
    cluster = next(root.rglob(f"clusters-{objective}.npz"))
    report = read(cluster.with_name("results.json"))
    study = next((old_root / f"confirm-{objective}").glob("*/manifest.json")).parent
    original = read(study / "report/results.json")
    manifest = read(study / "manifest.json")
    assert manifest["fingerprint"] == sha(
        json.dumps(
            {k: v for k, v in manifest.items() if k != "fingerprint"}, sort_keys=True
        ).encode()
    )
    for name in ("sentences.json", "vocabulary.json"):
        assert sha((study / name).read_bytes()) == manifest["artifact_sha256"][name]
    assert report["manifest"] == original["manifest"] == manifest["fingerprint"]
    assert report["selection"] == original["selection"]
    assert report["refits"] == original["refits"]
    assert report["analyzer_sha256"] == analyzer
    assert report["accumulator_dtype"] == "float64" and report["cluster_export"]
    assert report["cluster_sha256"][cluster.name] == sha(cluster.read_bytes())
    with np.load(cluster, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    assert arrays["seeds"].tolist() == [44, 45, 46]
    assert arrays["arms"].tolist() == [r["arm"] for r in report["summaries"]]
    pairs = [frozenset((r["left"], r["right"])) for r in report["paired_intervals"]]
    assert len(pairs) == 15 and set(pairs) == {
        frozenset(pair) for pair in combinations(arrays["arms"].tolist() + ["frequency"], 2)
    }
    for prefix in ("", "noninitial_"):
        sums, counts = arrays[prefix + "metric_sums"], arrays[prefix + "counts"]
        assert sums.shape == (5, 3, 40434, 3) and sums.dtype == np.float64
        assert np.isfinite(sums).all() and (sums >= 0).all()
        assert np.array_equal(sums[..., 1:], np.round(sums[..., 1:]))
        assert (sums[..., 1] <= sums[..., 2]).all()
        assert (sums[..., 2] <= counts[None, None, :]).all()
    assert (arrays["metric_sums"] >= arrays["noninitial_metric_sums"] - 1e-10).all()
    if objective == "causal":
        np.testing.assert_array_equal(arrays["metric_sums"], arrays["noninitial_metric_sums"])
    else:
        np.testing.assert_array_equal(
            arrays["counts"] - arrays["noninitial_counts"], np.ones(40434)
        )
    means = arrays["metric_sums"].sum(2) / arrays["counts"].sum()
    for i, row in enumerate(report["summaries"]):
        np.testing.assert_allclose(means[i].mean(0), row["mean"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(means[i].std(0, ddof=1), row["seed_sd"], atol=1e-10, rtol=0)
    return arrays, report, expected_population(study, arrays, objective)


def validation_selection(study_root):
    """Check whether top-5 would change the existing three-LR development winners."""
    groups = {
        "bert": ("bert-32", "bert-32-lr-0.0003", "bert-32-lr-0.01"),
        "mlp": ("mlp-32", "mlp-32-lr-0.0003", "mlp-32-lr-0.01"),
        "crz": ("hybrid-q8-depth2-init0.05", "hybrid-q8-lr-0.0003", "hybrid-q8-lr-0.01"),
        "identity": (
            "hybrid-q8-identity",
            "hybrid-q8-identity-lr-0.0003",
            "hybrid-q8-identity-lr-0.01",
        ),
    }
    result = {}
    for objective in ("causal", "cbow"):
        path = next((study_root / f"refine-{objective}").glob("*/selection.json"))
        selection = read(path)
        assert selection["fingerprint"] == sha(
            json.dumps(
                {k: v for k, v in selection.items() if k != "fingerprint"}, sort_keys=True
            ).encode()
        )
        assert selection["manifest"] == read(path.with_name("manifest.json"))["fingerprint"]
        rows = []
        for group, names in groups.items():
            candidates = [r for r in selection["candidates"] if r["arm"]["name"] in names]
            assert len(candidates) == 3
            assert {r["arm"]["lr"] for r in candidates} == {0.0003, 0.003, 0.01}
            winners = {}
            for metric, other in (("top1", "top5"), ("top5", "top1")):
                winner = min(
                    candidates,
                    key=lambda r: (
                        -r["mean"][metric],
                        r["mean"]["nll"],
                        -r["mean"][other],
                        r["arm"]["name"],
                    ),
                )
                winners[metric] = winner["arm"]["lr"]
            rows.append(dict(group=group, winners=winners, candidates=candidates))
        result[objective] = dict(selection_sha256=sha(path.read_bytes()), groups=rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Extracted prediction-review archive")
    parser.add_argument(
        "--study-root", type=Path, required=True, help="Original diagnostic studies"
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/robust-followup-prediction-review")
    )
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()
    if not 2000 <= args.draws <= 50000:
        parser.error("Use 2000..50000 draws; 2000 reproduces the original report")
    validation = validation_selection(args.study_root)
    analyzer = sha(subprocess.check_output(["git", "show", "369790b:src/qcse/benchmark_report.py"]))
    arrays, reports, checks = {}, {}, {}
    for objective in ("causal", "cbow"):
        arrays[objective], reports[objective], checks[objective] = load_export(
            args.root, args.study_root, objective, analyzer
        )
    np.testing.assert_array_equal(arrays["causal"]["groups"], arrays["cbow"]["groups"])
    np.testing.assert_array_equal(arrays["causal"]["counts"], arrays["cbow"]["noninitial_counts"])
    assert checks["causal"]["common_target_sha256"] == checks["cbow"]["common_target_sha256"]
    names = arrays["causal"]["arms"].tolist() + ["frequency"]
    assert names[:-1] == arrays["cbow"]["arms"].tolist()
    populations = [
        ("causal/all", "causal", ""),
        ("cbow/all", "cbow", ""),
        ("cbow/common", "cbow", "noninitial_"),
    ]
    values, counts = [], []
    for _, objective, prefix in populations:
        a = arrays[objective]
        values.append(
            np.concatenate((a[prefix + "metric_sums"], a[prefix + "frequency_sums"][None]), axis=0)
        )
        counts.append(a[prefix + "counts"])
    sums, counts = np.stack(values), np.stack(counts)
    print(f"Validated exports and corpus; resampling {args.draws} paired draws.", flush=True)
    paired, sentence_only = bootstrap(sums, counts, args.draws)
    seed_means = sums.sum(3) / counts.sum(1)[:, None, None, None]
    contrasts, scores = [], []
    for p, (population, objective, _) in enumerate(populations):
        for i, name in enumerate(names):
            scores.append(
                dict(
                    population=population,
                    arm=name,
                    mean=seed_means[p, i].mean(0).tolist(),
                    seed_metrics=seed_means[p, i].tolist(),
                    seed_sd=seed_means[p, i].std(0, ddof=1).tolist(),
                    sentence_macro=(sums[p, i] / counts[p][None, :, None]).mean((0, 1)).tolist(),
                )
            )
        for row in reports[objective]["paired_intervals"]:
            left, right = names.index(row["left"]), names.index(row["right"])
            delta = seed_means[p, left] - seed_means[p, right]
            samples = paired[:, p, left] - paired[:, p, right]
            reproduction = interval(samples[:2000], row["confidence"])
            if population.endswith("/all"):
                for k in ("low", "high"):
                    np.testing.assert_allclose(reproduction[k], row[k], atol=1e-10, rtol=0)
                np.testing.assert_allclose(delta.mean(0), row["difference"], atol=1e-10, rtol=0)
            exact_seed_only = np.array(
                [delta[list(ids)].mean(0) for ids in product(range(3), repeat=3)]
            )
            contrasts.append(
                dict(
                    population=population,
                    left=row["left"],
                    right=row["right"],
                    primary=row["primary"] and population.endswith("/all"),
                    confidence=row["confidence"],
                    difference=delta.mean(0).tolist(),
                    seed_differences=delta.tolist(),
                    leave_one_seed_out=np.stack(
                        [np.delete(delta, i, axis=0).mean(0) for i in range(3)]
                    ).tolist(),
                    paired_2000=reproduction,
                    paired=interval(samples, row["confidence"]),
                    sentences_only=interval(
                        sentence_only[:, p, left] - sentence_only[:, p, right], row["confidence"]
                    ),
                    empirical_seeds_only=interval(exact_seed_only, row["confidence"]),
                    macro_difference=((sums[p, left] - sums[p, right]) / counts[p][None, :, None])
                    .mean((0, 1))
                    .tolist(),
                )
            )
    # Sentence-initial CBOW targets: descriptive subgroup selected after seeing outcomes.
    a = arrays["cbow"]
    first = (
        np.concatenate(
            (
                a["metric_sums"] - a["noninitial_metric_sums"],
                (a["frequency_sums"] - a["noninitial_frequency_sums"])[None],
            ),
            axis=0,
        ).sum(2)
        / (a["counts"] - a["noninitial_counts"]).sum()
    )
    for i, name in enumerate(names):
        scores.append(
            dict(
                population="cbow/initial",
                arm=name,
                mean=first[i].mean(0).tolist(),
                seed_metrics=first[i].tolist(),
                seed_sd=first[i].std(0, ddof=1).tolist(),
            )
        )
    result = dict(
        archive_sha256=sha(args.archive.read_bytes()),
        analyzer_sha256=analyzer,
        review_script_sha256=sha(Path(__file__).read_bytes()),
        validation_selection=validation,
        checks=checks,
        corrected_reports=reports,
        draws=args.draws,
        seed=42,
        reproduced_native_intervals=30,
        scores=scores,
        contrasts=contrasts,
        metric_order=METRICS,
        scope="Compact aggregates, corpus group/target hash and training-only per-group frequency "
        "verified. All 30 original paired intervals independently reproduced. Common-position, "
        "macro and subgroup findings are post hoc. Raw contexts/logits/ranks are not included. "
        "Bootstrap conditions on three trained seeds and one split, "
        "not full tuning/split variance.",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = [
        dict(population=r["population"], arm=r["arm"], **dict(zip(METRICS, r["mean"], strict=True)))
        for r in scores
    ]
    with (args.output / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for c in contrasts:
        if c["left"] == "hybrid-q8-depth2-init0.05" and c["right"] == "hybrid-q8-identity":
            print(c["population"], c["difference"], c["paired"], flush=True)
    print(args.output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
