"""
Revision re-analysis: paired bootstrap drift, addressing reviewer aH6z's
core statistical objection (independent resampling was used despite the
same prompt_id being evaluated across model versions).

Outputs (all under outputs/tables/):
  - revision_paired_drift_by_benchmark.csv   (per transition x benchmark x metric)
  - revision_paired_aggregate.csv            (Table 2 replacement, all 8 release-drift transitions)
  - revision_paired_aggregate_strict.csv     (Table 2 replacement, strict same-tier successors only)
  - revision_effect_sizes.csv                (paired Cohen's d for every transition x metric, BH-corrected q-values)
"""
import numpy as np
import pandas as pd
from scipy import stats

SCORED_PATH = "outputs/results/results_with_metrics_20251115_035107.csv"
METRICS = ["sentiment_score", "toxicity_score", "stereotype_score"]
N_BOOT = 1000
SEED = 42

RELEASE_CHAINS = {
    "openai": ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
    "claude_sonnet": ["claude-sonnet-4-20250514", "claude-sonnet-4-5-20250929"],
    "gemini": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
}

# Same product-tier progression only ("strict successor" per AzMk's request).
STRICT_TRANSITIONS = {
    ("openai", "gpt-4-turbo", "gpt-4o"),
    ("claude_sonnet", "claude-sonnet-4-20250514", "claude-sonnet-4-5-20250929"),
    ("gemini", "gemini-2.0-flash", "gemini-2.5-flash"),
}


def paired_bootstrap_drift(df, model_t, model_t1, dataset, metric, rng):
    a = df[(df["model"] == model_t) & (df["dataset"] == dataset)][["prompt_id", metric]].dropna()
    b = df[(df["model"] == model_t1) & (df["dataset"] == dataset)][["prompt_id", metric]].dropna()
    merged = a.merge(b, on="prompt_id", suffixes=("_t", "_t1"))
    n = len(merged)
    if n == 0:
        return None
    vt = merged[f"{metric}_t"].values
    vt1 = merged[f"{metric}_t1"].values
    diffs = vt1 - vt
    drift = float(diffs.mean())

    idx = np.arange(n)
    boot_drifts = np.array([
        diffs[rng.choice(idx, size=n, replace=True)].mean()
        for _ in range(N_BOOT)
    ])
    ci_lower, ci_upper = np.percentile(boot_drifts, [2.5, 97.5])

    sd = diffs.std(ddof=1)
    cohens_d = drift / sd if sd > 0 else 0.0
    # paired t-test as the confirmatory complement to the bootstrap p-value
    if sd > 0 and n > 1:
        t_stat, p_value = stats.ttest_rel(vt1, vt)
    else:
        t_stat, p_value = np.nan, 1.0

    return {
        "n_paired": n,
        "drift": round(drift, 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "cohens_d": round(float(cohens_d), 4),
        "t_statistic": round(float(t_stat), 4) if not np.isnan(t_stat) else None,
        "p_value": float(p_value),
        "_boot_drifts": boot_drifts,  # kept for pooled aggregate bootstrap; dropped before CSV export
    }


def main():
    df = pd.read_csv(SCORED_PATH, low_memory=False)
    rng = np.random.default_rng(SEED)

    per_cell_records = []
    for family, chain in RELEASE_CHAINS.items():
        for i in range(len(chain) - 1):
            m_t, m_t1 = chain[i], chain[i + 1]
            is_strict = (family, m_t, m_t1) in STRICT_TRANSITIONS
            for dataset in sorted(df["dataset"].unique()):
                for metric in METRICS:
                    result = paired_bootstrap_drift(df, m_t, m_t1, dataset, metric, rng)
                    if result is None:
                        continue
                    per_cell_records.append({
                        "family": family, "model_t": m_t, "model_t1": m_t1,
                        "dataset": dataset, "metric": metric,
                        "strict_successor": is_strict,
                        **result,
                    })

    boot_cache = {i: rec.pop("_boot_drifts") for i, rec in enumerate(per_cell_records)}
    cell_df = pd.DataFrame(per_cell_records)
    cell_df["_cache_key"] = list(range(len(per_cell_records)))

    # Benjamini-Hochberg correction across all transition x benchmark x metric tests
    cell_df = cell_df.sort_values("p_value").reset_index(drop=True)
    m = len(cell_df)
    cell_df["bh_rank"] = np.arange(1, m + 1)
    cell_df["bh_critical"] = cell_df["bh_rank"] / m * 0.05
    cell_df["q_value"] = (cell_df["p_value"] * m / cell_df["bh_rank"]).clip(upper=1.0)
    cell_df["q_value"] = cell_df["q_value"][::-1].cummin()[::-1]  # enforce monotonicity
    cell_df["significant_raw_p001"] = cell_df["p_value"] < 0.001
    cell_df["significant_bh_q05"] = cell_df["q_value"] < 0.05

    cell_df_export = cell_df.drop(columns=["_cache_key"])
    cell_df_export.to_csv("outputs/tables/revision_paired_drift_by_benchmark.csv", index=False)
    print(f"Wrote {len(cell_df)} transition x benchmark x metric paired tests.")
    print(f"Total tests: {m}. Significant at raw p<0.001: {cell_df['significant_raw_p001'].sum()}. "
          f"Significant at BH q<0.05: {cell_df['significant_bh_q05'].sum()}.")

    # Aggregate (Table 2 equivalent): sample-size-weighted mean drift across
    # all release-drift transitions, per metric, with a *pooled* paired bootstrap
    # CI -- for each of the 1000 iterations, take each cell's own bootstrap draw
    # for that iteration and combine with sample-size weights, so the aggregate
    # CI is a genuine two-level (paired-example + across-cell) bootstrap, not a
    # normal approximation.
    def aggregate(subset, label):
        rows = []
        for metric in METRICS:
            sub = subset[subset["metric"] == metric]
            if sub.empty:
                continue
            weights = sub["n_paired"].values
            drift_w = np.average(sub["drift"].values, weights=weights)

            cell_boot_matrix = np.array([boot_cache[k] for k in sub["_cache_key"].values])  # (n_cells, N_BOOT)
            boot_means = np.average(cell_boot_matrix, axis=0, weights=weights)  # (N_BOOT,)
            ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

            rows.append({
                "aggregate_set": label,
                "metric": metric,
                "mean_drift": round(float(drift_w), 6),
                "ci_lower": round(float(ci_lower), 6),
                "ci_upper": round(float(ci_upper), 6),
                "n_transitions_x_benchmarks": len(sub),
                "total_paired_n": int(weights.sum()),
            })
        return pd.DataFrame(rows)

    agg_all = aggregate(cell_df, "all_release_drift_8_transitions")
    agg_strict = aggregate(cell_df[cell_df["strict_successor"]], "strict_same_tier_successors_3_transitions")

    agg_all.to_csv("outputs/tables/revision_paired_aggregate.csv", index=False)
    agg_strict.to_csv("outputs/tables/revision_paired_aggregate_strict.csv", index=False)

    print("\n=== Aggregate (all 8 release-drift transitions, paired) ===")
    print(agg_all.to_string(index=False))
    print("\n=== Aggregate (3 strict same-tier successors only, paired) ===")
    print(agg_strict.to_string(index=False))

    # Effect sizes table equivalent (largest signals) using paired d
    top = cell_df_export.reindex(cell_df["cohens_d"].abs().sort_values(ascending=False).index).head(15)
    top.to_csv("outputs/tables/revision_effect_sizes.csv", index=False)
    print("\n=== Top 15 |paired Cohen's d| ===")
    print(top[["family", "model_t", "model_t1", "dataset", "metric", "drift", "cohens_d", "p_value", "q_value"]].to_string(index=False))

    # --- Transition-level pooled-across-all-benchmarks effect sizes -----------
    # This mirrors what Table 8 in the original paper appears to report
    # ("Gemini Stereotype", "Toxicity (all)"): one pooled per-example diff
    # array per transition x metric, spanning every benchmark that transition
    # was evaluated on, computed with PAIRED per-example differences.
    pooled_records = []
    for family, chain in RELEASE_CHAINS.items():
        for i in range(len(chain) - 1):
            m_t, m_t1 = chain[i], chain[i + 1]
            for metric in METRICS:
                all_diffs = []
                for dataset in sorted(df["dataset"].unique()):
                    a = df[(df["model"] == m_t) & (df["dataset"] == dataset)][["prompt_id", metric]].dropna()
                    b = df[(df["model"] == m_t1) & (df["dataset"] == dataset)][["prompt_id", metric]].dropna()
                    merged = a.merge(b, on="prompt_id", suffixes=("_t", "_t1"))
                    if len(merged) == 0:
                        continue
                    all_diffs.append((merged[f"{metric}_t1"] - merged[f"{metric}_t"]).values)
                if not all_diffs:
                    continue
                diffs = np.concatenate(all_diffs)
                n = len(diffs)
                drift = float(diffs.mean())
                sd = diffs.std(ddof=1)
                cohens_d = drift / sd if sd > 0 else 0.0
                idx = np.arange(n)
                boot = np.array([diffs[rng.choice(idx, size=n, replace=True)].mean() for _ in range(N_BOOT)])
                ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
                t_stat, p_val = stats.ttest_1samp(diffs, 0.0) if n > 1 and sd > 0 else (np.nan, 1.0)
                pooled_records.append({
                    "family": family, "model_t": m_t, "model_t1": m_t1, "metric": metric,
                    "n_paired": n, "drift": round(drift, 6),
                    "ci_lower": round(float(ci_lo), 6), "ci_upper": round(float(ci_hi), 6),
                    "cohens_d": round(float(cohens_d), 4), "p_value": float(p_val),
                })
    pooled_df = pd.DataFrame(pooled_records).sort_values("p_value").reset_index(drop=True)
    mp = len(pooled_df)
    pooled_df["q_value"] = (pooled_df["p_value"] * mp / np.arange(1, mp + 1)).clip(upper=1.0)
    pooled_df["q_value"] = pooled_df["q_value"][::-1].cummin()[::-1]
    pooled_df.to_csv("outputs/tables/revision_effect_sizes_pooled_per_transition.csv", index=False)
    print("\n=== Per-transition pooled-across-benchmarks paired effect sizes (top 10 by |d|) ===")
    top_pooled = pooled_df.reindex(pooled_df["cohens_d"].abs().sort_values(ascending=False).index).head(10)
    print(top_pooled.to_string(index=False))

    # Fully-pooled-across-everything per metric ("Toxicity (all)"-style)
    full_pool_records = []
    for metric in METRICS:
        all_diffs = []
        for family, chain in RELEASE_CHAINS.items():
            for i in range(len(chain) - 1):
                m_t, m_t1 = chain[i], chain[i + 1]
                for dataset in sorted(df["dataset"].unique()):
                    a = df[(df["model"] == m_t) & (df["dataset"] == dataset)][["prompt_id", metric]].dropna()
                    b = df[(df["model"] == m_t1) & (df["dataset"] == dataset)][["prompt_id", metric]].dropna()
                    merged = a.merge(b, on="prompt_id", suffixes=("_t", "_t1"))
                    if len(merged) == 0:
                        continue
                    all_diffs.append((merged[f"{metric}_t1"] - merged[f"{metric}_t"]).values)
        diffs = np.concatenate(all_diffs)
        n = len(diffs)
        drift = float(diffs.mean())
        sd = diffs.std(ddof=1)
        cohens_d = drift / sd if sd > 0 else 0.0
        idx = np.arange(n)
        boot = np.array([diffs[rng.choice(idx, size=n, replace=True)].mean() for _ in range(N_BOOT)])
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
        full_pool_records.append({
            "metric": metric, "n_paired": n, "drift": round(drift, 6),
            "ci_lower": round(float(ci_lo), 6), "ci_upper": round(float(ci_hi), 6),
            "cohens_d": round(float(cohens_d), 4), "p_value": float(p_val),
        })
    full_pool_df = pd.DataFrame(full_pool_records)
    full_pool_df.to_csv("outputs/tables/revision_effect_sizes_fully_pooled.csv", index=False)
    print("\n=== Fully-pooled-across-all-8-transitions paired effect size, per metric ===")
    print(full_pool_df.to_string(index=False))


if __name__ == "__main__":
    main()
