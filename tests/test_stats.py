import numpy as np
import pandas as pd
from scipy import stats

from src.config import N_BOOT
from src.stats import adjust_bh, bootstrap_means, pair_scores, paired_stats


def responses(model, scores, prompt_ids, dataset="bbq"):
    return pd.DataFrame({"model": model, "dataset": dataset, "prompt_id": prompt_ids, "toxicity_score": scores})


def test_pairs_by_prompt_id_and_drops_unmatched():
    df = pd.concat([
        responses("old", [0.1, 0.2, 0.3], [0, 1, 2]),
        responses("new", [0.9, 0.5], [2, 0]),  # prompt 1 missing, rows shuffled
    ])
    old, new = pair_scores(df, "old", "new", "toxicity_score")
    assert list(zip(old, new)) == [(0.1, 0.5), (0.3, 0.9)]


def test_no_pairs_across_datasets():
    df = pd.concat([responses("old", [0.1], [0], "bbq"), responses("new", [0.4], [0], "bold")])
    old, _ = pair_scores(df, "old", "new", "toxicity_score")
    assert len(old) == 0


def test_drift_is_new_minus_old():
    summary, _ = paired_stats(np.array([0.2, 0.4, 0.6]), np.random.default_rng(0))
    assert summary["drift"] == 0.4


def test_effect_size_and_paired_t_test():
    diffs = np.array([0.1, 0.3, 0.2, 0.5, -0.1])
    summary, _ = paired_stats(diffs, np.random.default_rng(0))
    assert summary["cohens_d"] == round(diffs.mean() / diffs.std(ddof=1), 4)
    assert summary["p_value"] == stats.ttest_rel(diffs, np.zeros_like(diffs)).pvalue


def test_bootstrap_is_seeded():
    diffs = np.random.default_rng(1).normal(size=50)
    a = bootstrap_means(diffs, np.random.default_rng(42))
    b = bootstrap_means(diffs, np.random.default_rng(42))
    assert len(a) == N_BOOT and np.array_equal(a, b)


def test_bh_matches_scipy():
    p = np.random.default_rng(3).uniform(size=40)
    q = adjust_bh(pd.DataFrame({"p_value": p}))
    assert np.allclose(q["q_value"], np.sort(stats.false_discovery_control(p, method="bh")), atol=1e-12)
    assert q["q_value"].is_monotonic_increasing
