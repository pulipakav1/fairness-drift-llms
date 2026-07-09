"""
Revision addition: demographic-slice drift, addressing AzMk's request that
the paper actually execute the axis-level analysis it recommends in
Section 6.4 rather than only recommending it. Uses the same paired
methodology as revision_paired_drift.py.
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

AXIS_MAP = {
    "gender": "Gender", "gender_identity": "Gender",
    "race": "Race/Ethnicity", "race-color": "Race/Ethnicity",
    "religion": "Religion",
    "age": "Age", "Age": "Age",
    "disability": "Disability", "ability": "Disability",
    "nationality": "Nationality",
    "socioeconomic": "Socioeconomic",
}


def main():
    df = pd.read_csv(SCORED_PATH, low_memory=False)
    df["demo_axis"] = df["axis"].map(AXIS_MAP)
    unmatched_rate = df["demo_axis"].isna().mean() * 100
    print(f"Rows mapped to the 7-axis taxonomy: {df['demo_axis'].notna().sum()}/{len(df)} "
          f"({100 - unmatched_rate:.1f}%); unmatched -> excluded, as in the paper's own methodology.")

    rng = np.random.default_rng(SEED)
    records = []
    for family, chain in RELEASE_CHAINS.items():
        for i in range(len(chain) - 1):
            m_t, m_t1 = chain[i], chain[i + 1]
            for axis in sorted(df["demo_axis"].dropna().unique()):
                for metric in METRICS:
                    a = df[(df["model"] == m_t) & (df["demo_axis"] == axis)][["prompt_id", "dataset", metric]].dropna()
                    b = df[(df["model"] == m_t1) & (df["demo_axis"] == axis)][["prompt_id", "dataset", metric]].dropna()
                    merged = a.merge(b, on=["prompt_id", "dataset"], suffixes=("_t", "_t1"))
                    n = len(merged)
                    if n < 5:
                        continue
                    vt, vt1 = merged[f"{metric}_t"].values, merged[f"{metric}_t1"].values
                    diffs = vt1 - vt
                    drift = float(diffs.mean())
                    sd = diffs.std(ddof=1)
                    idx = np.arange(n)
                    boot = np.array([diffs[rng.choice(idx, size=n, replace=True)].mean() for _ in range(N_BOOT)])
                    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
                    cohens_d = drift / sd if sd > 0 else 0.0
                    t_stat, p_val = stats.ttest_1samp(diffs, 0.0) if sd > 0 and n > 1 else (np.nan, 1.0)
                    records.append({
                        "family": family, "model_t": m_t, "model_t1": m_t1, "axis": axis, "metric": metric,
                        "n_paired": n, "drift": round(drift, 6),
                        "ci_lower": round(float(ci_lo), 6), "ci_upper": round(float(ci_hi), 6),
                        "cohens_d": round(float(cohens_d), 4), "p_value": float(p_val),
                    })

    rec_df = pd.DataFrame(records).sort_values("p_value").reset_index(drop=True)
    m = len(rec_df)
    rec_df["q_value"] = (rec_df["p_value"] * m / np.arange(1, m + 1)).clip(upper=1.0)
    rec_df["q_value"] = rec_df["q_value"][::-1].cummin()[::-1]
    rec_df.to_csv("outputs/tables/revision_demographic_slice_drift.csv", index=False)

    print(f"\n{m} axis x transition x metric tests.")
    print(f"Significant at BH q<0.05: {(rec_df['q_value'] < 0.05).sum()}")

    print("\n=== Aggregate drift by axis (pooled across all release-drift transitions, per metric) ===")
    agg = df.copy()
    for metric in METRICS:
        rows = []
        for axis in sorted(df["demo_axis"].dropna().unique()):
            sub = rec_df[(rec_df["axis"] == axis) & (rec_df["metric"] == metric)]
            if sub.empty:
                continue
            w = sub["n_paired"].values
            drift_w = np.average(sub["drift"].values, weights=w)
            rows.append((axis, round(float(drift_w), 5), int(w.sum())))
        rows.sort(key=lambda r: -abs(r[1]))
        print(f"-- {metric} --")
        for axis, drift, n in rows:
            print(f"   {axis:16s} drift={drift:+.5f}  n={n}")

    print("\n=== Top 15 axis-level effects by |paired Cohen's d| ===")
    top = rec_df.reindex(rec_df["cohens_d"].abs().sort_values(ascending=False).index).head(15)
    print(top[["family", "model_t", "model_t1", "axis", "metric", "drift", "cohens_d", "p_value", "q_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
