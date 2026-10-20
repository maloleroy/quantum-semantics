"""Verify replicated raw gradients, saved sources and independently rebuilt inputs."""

import argparse
import hashlib
import json
import subprocess
from itertools import product
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def check_inputs(study, run, objective):
    manifest = read(study / "manifest.json")
    sentences = read(study / "sentences.json")
    vocabulary = read(study / "vocabulary.json")
    for name in ("sentences.json", "vocabulary.json"):
        assert sha((study / name).read_bytes()) == manifest["artifact_sha256"][name]
    keys = sorted(map(tuple, sentences))
    assert len(keys) == len(set(keys))
    order = np.random.default_rng(2026).permutation(len(keys))
    heldout = {keys[i] for i in order[: round(0.2 * len(keys))]}
    development = [k for k in keys if k not in heldout]
    order = np.random.default_rng(2027).permutation(len(development))
    validation = {development[i] for i in np.array_split(order, 5)[0]}
    lengths = np.array([len(s) - (objective == "causal") for s in sentences])
    starts = np.concatenate(([0], lengths.cumsum()))
    groups = np.repeat(np.arange(len(sentences)), lengths)
    training = np.flatnonzero(
        [
            tuple(sentences[g]) not in heldout and tuple(sentences[g]) not in validation
            for g in range(len(sentences))
        ]
    )
    train_ids = np.flatnonzero(np.isin(groups, training))
    lookup = {w: i for i, w in enumerate(vocabulary)}
    for seed in (619, 620, 621):
        with np.load(run / f"inputs-batch{seed}.npz", allow_pickle=False) as a:
            ids = np.random.default_rng(seed).choice(train_ids, 8, replace=False)
            np.testing.assert_array_equal(a["ids"], ids)
            for index, g in enumerate(groups[ids]):
                words = [lookup[w] for w in sentences[g]]
                pos = ids[index] - starts[g] + (objective == "causal")
                if objective == "causal":
                    context = [len(vocabulary)] * (4 - min(4, pos)) + words[max(0, pos - 4) : pos]
                else:
                    left, right = words[max(0, pos - 2) : pos], words[pos + 1 : pos + 3]
                    context = [len(vocabulary)] * (2 - len(left)) + left + right
                    context += [len(vocabulary)] * (4 - len(context))
                np.testing.assert_array_equal(a["contexts"][index], context)
                assert a["targets"][index] == words[pos]


def audit(root, studies):
    launcher = subprocess.check_output(
        ["git", "show", "d8f8295:scripts/submit_gradient_replications.sh"], text=True
    )
    expected_job = launcher.split("<<'SBATCH'\n", 1)[1].rsplit("\nSBATCH", 1)[0] + "\n"
    assert (root / "submitted.sbatch").read_text() == expected_job
    rows = []
    for objective in ("causal", "cbow"):
        (path,) = (root / objective).glob("*/summary.json")
        run, summary = path.parent, read(path)
        (study,) = (studies / f"confirm-{objective}").glob("*/manifest.json")
        manifest = read(study)
        assert manifest["fingerprint"] == sha(
            json.dumps(
                {k: v for k, v in manifest.items() if k != "fingerprint"}, sort_keys=True
            ).encode()
        )
        assert summary["manifest"] == manifest["fingerprint"]
        assert summary["code_sha256"] == manifest["code_sha256"]
        assert summary["runtime"]["device"] == "cuda"
        assert summary["runtime"]["scheduler"]["SLURM_JOB_ID"] == "10324"
        assert summary["runtime"]["cuda_device"]["total_memory_bytes"] < 11 * 1024**3
        for name, checksum in summary["code_sha256"].items():
            data = (run / "source" / name).read_bytes()
            assert sha(data) == checksum
            assert data == subprocess.check_output(["git", "show", f"483bea6:{name}"])
        executed = (run / "executed_probe.py").read_bytes()
        assert executed == (root / "probe.py").read_bytes()
        assert summary["analyzer_sha256"] == sha(executed)
        assert executed == subprocess.check_output(
            ["git", "show", "d43cc96:scripts/probe_nlp_trainability.py"]
        )
        check_inputs(study.parent, run, objective)
        expected = set(
            product((619, 620, 621), (619, 719, 819), (2, 4, 8, 12), (2, 8), (0.05, 1.0))
        )
        assert len(summary["rows"]) == 144
        assert {
            (r["batch_seed"], r["head_seed"], r["qubits"], r["layers"], r["init_scale"])
            for r in summary["rows"]
        } == expected
        (old,) = (study.parent / "nlp-trainability-probe").glob("*/summary.json")
        old_rows = {(r["qubits"], r["layers"], r["init_scale"]): r for r in read(old)["rows"]}
        hashes = {}
        for r in summary["rows"]:
            q, d, s = r["qubits"], r["layers"], r["init_scale"]
            file = run / f"batch{r['batch_seed']}-head{r['head_seed']}-q{q}-depth{d}-init{s:g}.npz"
            with np.load(file, allow_pickle=False) as a:
                g, loss = a["gradients"], a["losses"]
                assert g.shape == (32, d, 3 * q - 1) and loss.shape == (32,)
                assert np.isfinite(g).all() and np.isfinite(loss).all()
                np.testing.assert_array_equal(a["seeds"], np.arange(42, 74))
                checks = dict(
                    first_rx_gradient_variance=g[:, 0, 0].var(ddof=1),
                    mean_gradient_variance=g.var(0, ddof=1).mean(),
                    rms_gradient=np.sqrt(np.mean(g * g)),
                    loss_mean=loss.mean(),
                )
                for key, value in checks.items():
                    np.testing.assert_allclose(value, r[key], atol=0, rtol=1e-12)
                if r["batch_seed"] == r["head_seed"] == 619:
                    assert r["common_state_sha256"] == old_rows[q, d, s]["common_state_sha256"]
                    for key, value in checks.items():
                        np.testing.assert_allclose(
                            value, old_rows[q, d, s][key], rtol=1e-6, atol=1e-12
                        )
                hashes.setdefault((q, r["head_seed"]), set()).add(r["common_state_sha256"])
                rows.append(
                    dict(
                        objective=objective,
                        **r,
                        mean_gradient_rms=float(np.sqrt(np.mean(g.mean(0) ** 2))),
                        minimum_circuit_norm=float(np.linalg.norm(g, axis=(1, 2)).min()),
                        near_zero_fraction=float(np.mean(np.abs(g) < 1e-8)),
                        raw_sha256=sha(file.read_bytes()),
                    )
                )
        assert all(len(v) == 1 for v in hashes.values())
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--studies", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/gradient-replicated-10324"))
    a = p.parse_args()
    rows = audit(a.root, a.studies)
    preflight = read(a.root / "backend-preflight.json")
    assert preflight["device"] == "cuda" and len(preflight["results"]) == preflight["checked"] == 71
    summary = dict(
        archive_sha256=sha(a.archive.read_bytes()),
        conditions=288,
        circuit_gradient_evaluations=9216,
        near_zero_absolute_threshold=1e-8,
        rows=rows,
        timing_tsv=(a.root / "timing.tsv").read_text(),
        preflight=preflight,
        scope="Initialization only. Nine crossed batch/head conditions are not nine independent "
        "training replicates. Register size changes the classical encoder/head. No asymptotic "
        "barren-plateau or trained-landscape claim. Low near-identity variance "
        "is not zero mean gradient.",
    )
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(a.output / "summary.json")


if __name__ == "__main__":
    main()
