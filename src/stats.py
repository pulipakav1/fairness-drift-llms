import numpy as np
from scipy import stats

from src.config import N_BOOT


def pair_scores(df, old_model, new_model, metric, keys=("dataset", "prompt_id")):
    """Paired scores on the prompts both models answered."""
    keys = list(keys)
    old = df[df["model"] == old_model][keys + [metric]].dropna()
    new = df[df["model"] == new_model][keys + [metric]].dropna()
    pairs = old.merge(new, on=keys, suffixes=("_old", "_new"))
    return pairs[f"{metric}_old"].values, pairs[f"{metric}_new"].values


def bootstrap_means(diffs, rng):
    n = len(diffs)
    idx = np.arange(n)
    return np.array([diffs[rng.choice(idx, size=n, replace=True)].mean() for _ in range(N_BOOT)])


def paired_stats(diffs, rng):
    """Drift, bootstrap CI, Cohen's d and t-test; returns (summary, boot)."""
    n = len(diffs)
    drift = float(diffs.mean())
    boot = bootstrap_means(diffs, rng)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    sd = diffs.std(ddof=1)
    effect = drift / sd if sd > 0 else 0.0
    # One-sample t-test on diffs = paired t-test
    t, p = stats.ttest_1samp(diffs, 0.0) if n > 1 and sd > 0 else (np.nan, 1.0)
    summary = {
        "n_paired": n,
        "drift": round(drift, 6),
        "ci_lower": round(float(ci_low), 6),
        "ci_upper": round(float(ci_high), 6),
        "cohens_d": round(float(effect), 4),
        "t_statistic": None if np.isnan(t) else round(float(t), 4),
        "p_value": float(p),
    }
    return summary, boot


def adjust_bh(df):
    """Sort by p-value and add BH q-values (rows = one test family)."""
    df = df.sort_values("p_value").reset_index(drop=True)
    m = len(df)
    q = (df["p_value"] * m / np.arange(1, m + 1)).clip(upper=1.0)
    df["q_value"] = q[::-1].cummin()[::-1]
    return df
