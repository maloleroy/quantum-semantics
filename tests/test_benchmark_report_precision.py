import numpy as np

from qcse.benchmark_report import export_clusters, prediction_summary


def test_large_float32_summary_matches_float64_seed_reductions():
    values = np.empty((3, 350000, 3), dtype=np.float32)
    for seed, nll in enumerate((5.54, 3.25, 2.18)):
        values[seed, :, 0] = nll
        values[seed, :, 1] = np.arange(350000) % 5 == seed
        values[seed, :, 2] = np.arange(350000) % 3 != seed
    expected = values.astype(np.float64).mean(1)
    result = prediction_summary(values)
    np.testing.assert_allclose(result["mean"], expected.mean(0), atol=1e-12, rtol=0)
    np.testing.assert_allclose(result["seed_sd"], expected.std(0, ddof=1), atol=1e-12, rtol=0)
    assert abs(values.mean((0, 1))[0] - result["mean"][0]) > 1e-3


def test_cluster_export_preserves_token_weighting_and_seed_model_pairing(tmp_path):
    groups = np.array([9, 9, 10, 11, 11])
    records = {}
    for arm, offset in (("circuit", 1), ("identity", 0)):
        for seed in (44, 45):
            values = np.arange(15, dtype=np.float32).reshape(5, 3) + offset + seed
            records[("cbow", arm, seed)] = dict(
                groups=groups,
                metrics=values,
                frequency=np.zeros_like(values),
                positions=np.array([0, 1, 0, 0, 1]),
                targets=np.array([1, 2, 3, 4, 5]),
            )
    export_clusters(tmp_path, "cbow", ["circuit", "identity"], [44, 45], records)
    with np.load(tmp_path / "clusters-cbow.npz") as data:
        np.testing.assert_array_equal(data["groups"], [9, 10, 11])
        np.testing.assert_array_equal(data["counts"], [2, 1, 2])
        np.testing.assert_array_equal(data["noninitial_counts"], [1, 0, 1])
        np.testing.assert_array_equal(data["arms"], ["circuit", "identity"])
        np.testing.assert_array_equal(data["seeds"], [44, 45])
        for i, arm in enumerate(data["arms"]):
            for j, seed in enumerate(data["seeds"]):
                expected = records[("cbow", arm, seed)]["metrics"].mean(0, dtype=np.float64)
                np.testing.assert_allclose(
                    data["metric_sums"][i, j].sum(0) / data["counts"].sum(), expected
                )
                expected_subset = records[("cbow", arm, seed)]["metrics"][[1, 4]].mean(0)
                np.testing.assert_allclose(
                    data["noninitial_metric_sums"][i, j].sum(0) / 2, expected_subset
                )
        np.testing.assert_allclose(
            data["metric_sums"][0] - data["metric_sums"][1],
            np.broadcast_to(data["counts"][None, :, None], (2, 3, 3)),
        )
