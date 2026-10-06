import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    ABLATION_EXCLUDED, ALPHA, AXIS_MAP, CONTEXT_TRANSITIONS, DATASETS, METRICS, MIN_GROUP_SIZE, PROVIDER_GROUPS,
    SEED, STRICT_TRANSITIONS, release_transitions,
)
from src.data import DROPPED, response_issue
from src.stats import adjust_bh, pair_scores, paired_stats

ALL_8 = "all_release_drift_8_transitions"
STRICT_3 = "strict_same_tier_successors_3_transitions"


def pooled_diffs(df, old_model, new_model, datasets, metric):
    """Paired differences for one transition, across datasets in order."""
    parts = []
    for dataset in datasets:
        old, new = pair_scores(df[df["dataset"] == dataset], old_model, new_model, metric)
        if len(old):
            parts.append(new - old)
    return np.concatenate(parts) if parts else None


def aggregate(cells, boot, label):
    """n-weighted mean drift, with a weighted cell-bootstrap CI."""
    rows = []
    for metric in METRICS:
        sub = cells[cells["metric"] == metric]
        weights = sub["n_paired"].values
        boot_means = np.average(np.array([boot[i] for i in sub["_boot"]]), axis=0, weights=weights)
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        rows.append({
            "aggregate_set": label,
            "metric": metric,
            "mean_drift": round(float(np.average(sub["drift"].values, weights=weights)), 6),
            "ci_lower": round(float(ci_low), 6),
            "ci_upper": round(float(ci_high), 6),
            "n_transitions_x_benchmarks": len(sub),
            "total_paired_n": int(weights.sum()),
        })
    return pd.DataFrame(rows)


def paired_drift(df, datasets):
    """Paired drift for the 8 release transitions (one seeded RNG, fixed order)."""
    rng = np.random.default_rng(SEED)

    # Per transition x dataset x metric
    rows, boot = [], []
    for family, old_model, new_model in release_transitions():
        for dataset in datasets:
            for metric in METRICS:
                old, new = pair_scores(df[df["dataset"] == dataset], old_model, new_model, metric)
                if len(old) == 0:  # no OpenAI BOLD responses
                    continue
                summary, boot_means = paired_stats(new - old, rng)
                rows.append({"family": family, "model_t": old_model, "model_t1": new_model, "dataset": dataset,
                             "metric": metric,
                             "strict_successor": (family, old_model, new_model) in STRICT_TRANSITIONS,
                             **summary})
                boot.append(boot_means)
    benchmark = pd.DataFrame(rows)
    benchmark["_boot"] = range(len(benchmark))
    benchmark = adjust_bh(benchmark)  # BH across 156 tests
    q = benchmark.pop("q_value")
    benchmark["bh_rank"] = np.arange(1, len(benchmark) + 1)
    benchmark["bh_critical"] = benchmark["bh_rank"] / len(benchmark) * ALPHA
    benchmark["q_value"] = q
    benchmark["significant_raw_p001"] = benchmark["p_value"] < 0.001
    benchmark["significant_bh_q05"] = benchmark["q_value"] < ALPHA

    aggregates = pd.concat([
        aggregate(benchmark, boot, ALL_8),
        aggregate(benchmark[benchmark["strict_successor"]], boot, STRICT_3),
    ], ignore_index=True)

    # Pooled across datasets
    rows = []
    for family, old_model, new_model in release_transitions():
        for metric in METRICS:
            summary, _ = paired_stats(pooled_diffs(df, old_model, new_model, datasets, metric), rng)
            del summary["t_statistic"]
            rows.append({"family": family, "model_t": old_model, "model_t1": new_model, "metric": metric, **summary})
    transition = adjust_bh(pd.DataFrame(rows))  # BH across 24 tests

    # Fully pooled
    rows = []
    for metric in METRICS:
        diffs = np.concatenate([pooled_diffs(df, old_model, new_model, datasets, metric)
                                for _, old_model, new_model in release_transitions()])
        summary, _ = paired_stats(diffs, rng)
        del summary["t_statistic"]
        rows.append({"metric": metric, **summary})

    return {
        "benchmark": benchmark.drop(columns="_boot"),
        "transition": transition,
        "aggregate": aggregates,
        "pooled": pd.DataFrame(rows),
    }


def context_drift(df):
    """Pooled drift for the cross-tier and scale comparisons (no BH)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for group, old_model, new_model in CONTEXT_TRANSITIONS:
        for metric in METRICS:
            summary, _ = paired_stats(pooled_diffs(df, old_model, new_model, DATASETS, metric), rng)
            del summary["t_statistic"]
            rows.append({"comparison_group": group, "model_t": old_model, "model_t1": new_model, "metric": metric,
                         **summary})
    return pd.DataFrame(rows)


def demographic_drift(df):
    """Paired drift per demographic axis; axes with too few pairs are skipped."""
    df = df.assign(demo_axis=df["axis"].map(AXIS_MAP))
    rng = np.random.default_rng(SEED)
    rows = []
    for family, old_model, new_model in release_transitions():
        for axis in sorted(df["demo_axis"].dropna().unique()):
            for metric in METRICS:
                old, new = pair_scores(df[df["demo_axis"] == axis], old_model, new_model, metric)
                if len(old) < MIN_GROUP_SIZE:
                    continue
                summary, _ = paired_stats(new - old, rng)
                del summary["t_statistic"]
                rows.append({"family": family, "model_t": old_model, "model_t1": new_model, "axis": axis,
                             "metric": metric, **summary})
    return adjust_bh(pd.DataFrame(rows))  # BH across 144 tests


def demographic_counts(df):
    """Prompts per model for each mapped axis label, and paired n per axis."""
    per_model = df.groupby(["axis", "model"]).size()
    paired = {}
    for axis in sorted(set(AXIS_MAP.values())):
        sub = df[df["axis"].map(AXIS_MAP) == axis]
        paired[axis] = min(len(pair_scores(sub, old_model, new_model, METRICS[0])[0])
                           for _, old_model, new_model in release_transitions())
    rows = []
    for label, axis in AXIS_MAP.items():
        counts = per_model.loc[label]
        rows.append({
            "axis": axis,
            "label": label,
            "datasets": ";".join(sorted(df[df["axis"] == label]["dataset"].unique())),
            "prompts_per_model": int(counts.mode().iloc[0]),
            "min_prompts_per_model": int(counts.min()),
            "paired_n": paired[axis],
            "eligible_for_testing": paired[axis] >= MIN_GROUP_SIZE,
        })
    return pd.DataFrame(rows)


METRIC_PAIRS = [
    ("sentiment_score", "toxicity_score"),
    ("sentiment_score", "stereotype_score"),
    ("toxicity_score", "stereotype_score"),
]


def correlations(df):
    """Pearson r per response and per (model, dataset) mean."""
    rows = []
    for dataset_set, data in [("all_7", df), ("ablation_5", df[~df["dataset"].isin(ABLATION_EXCLUDED)])]:
        cell_means = data.groupby(["model", "dataset"])[METRICS].mean().reset_index()
        for level, points in [("per_example", data), ("model_benchmark", cell_means)]:
            for m1, m2 in METRIC_PAIRS:
                r, p = stats.pearsonr(points[m1], points[m2])
                rows.append({"metric_1": m1, "metric_2": m2, "pearson_r": round(float(r), 4), "p_value": float(p),
                             "analysis_level": level, "benchmark_set": dataset_set,
                             "n_models": data["model"].nunique(), "n": len(points)})
    return pd.DataFrame(rows)


def provider_scores(df):
    """Mean and SD per provider (descriptive)."""
    rows = []
    for label, providers in PROVIDER_GROUPS.items():
        sub = df[df["provider"].isin(providers)]
        row = {
            "provider_group": label,
            "models": ";".join(sorted(sub["model"].unique())),
            "n_models": sub["model"].nunique(),
            "benchmarks": ";".join(sorted(sub["dataset"].unique())),
            "n_benchmarks": sub["dataset"].nunique(),
            "n": len(sub),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std())
        rows.append(row)
    return pd.DataFrame(rows)


def _filter_counts(group):
    total = len(group)
    dropped = int(group["issue"].isin(DROPPED).sum())
    api_error = group["issue"] == "api_error"
    refusals = int((group["issue"] == "refusal").sum())
    return {
        "total_outputs": total,
        "valid_outputs": total - dropped,
        "filtered_outputs": dropped,
        "filtered_percent": round(100 * dropped / total, 2) if total else 0.0,
        "empty_count": int((group["issue"] == "empty").sum()),
        "no_text_count": int((group["issue"] == "no_text").sum()),
        "blocked_count": int((group["issue"] == "policy_blocked").sum()),
        "api_quota_error_count": int((api_error & group["quota"]).sum()),
        "api_error_other_count": int((api_error & ~group["quota"]).sum()),
        "refusal_pattern_count": refusals,
        "refusal_pattern_retained_count": refusals,  # refusals are kept
    }


def filtering_summary(raw):
    """Counts of dropped and kept responses; no text."""
    labelled = raw[["provider", "dataset"]].copy()
    labelled["issue"] = raw["response"].map(response_issue)
    labelled["quota"] = raw["response"].astype(str).str.contains("insufficient_quota|429", regex=True)

    rows = [{"level": "overall", "provider": "", "benchmark": "", **_filter_counts(labelled)}]
    for level, cols in [("benchmark", ["dataset"]), ("provider", ["provider"]),
                        ("provider_benchmark", ["provider", "dataset"])]:
        for key, group in labelled.groupby(cols):
            ids = dict(zip(cols, key))
            rows.append({"level": level, "provider": ids.get("provider", ""), "benchmark": ids.get("dataset", ""),
                         **_filter_counts(group)})
    return pd.DataFrame(rows)


KEY_COLS = ["analysis_level", "transition_set", "family", "model_t", "model_t1", "metric"]
STAT_COLS = ["n_paired", "drift", "ci_lower", "ci_upper", "cohens_d", "p_value", "q_value", "bh_family_size"]


def _comparison_rows(results):
    frames = []
    for aggregate_set, label in [(ALL_8, "all_8"), (STRICT_3, "strict_3")]:
        agg = results["aggregate"][results["aggregate"]["aggregate_set"] == aggregate_set].reset_index(drop=True)
        agg = agg.rename(columns={"mean_drift": "drift", "total_paired_n": "n_paired"})
        frames.append(agg.assign(analysis_level="aggregate", transition_set=label))
    transition = results["transition"]
    frames.append(transition.assign(analysis_level="transition_pooled", transition_set="all_8",
                                    bh_family_size=len(transition)))
    frames.append(results["pooled"].assign(analysis_level="fully_pooled", transition_set="all_8"))
    rows = pd.concat(frames, ignore_index=True)
    for col in KEY_COLS + STAT_COLS:
        if col not in rows:
            rows[col] = pd.NA
    return rows[KEY_COLS + STAT_COLS]


def _aggregate_row(results, aggregate_set, metric):
    agg = results["aggregate"]
    return agg[(agg["aggregate_set"] == aggregate_set) & (agg["metric"] == metric)].iloc[0]


def _excludes_zero(row):
    return bool(row["ci_lower"] > 0 or row["ci_upper"] < 0)


def ablation_comparison(full, ablated):
    """Full and ablated estimates side by side."""
    f = _comparison_rows(full).add_prefix("full_").rename(columns={f"full_{k}": k for k in KEY_COLS})
    a = _comparison_rows(ablated).add_prefix("ablation_").rename(columns={f"ablation_{k}": k for k in KEY_COLS})
    rows = f.merge(a, on=KEY_COLS, how="outer", validate="one_to_one")
    rows["same_sign"] = (rows["full_drift"] * rows["ablation_drift"]) > 0
    rows["_order"] = rows["analysis_level"].map({"aggregate": 0, "transition_pooled": 1, "fully_pooled": 2})
    rows = rows.sort_values(["_order", "transition_set", "metric", "family", "model_t"], na_position="first")
    return rows.drop(columns="_order").reset_index(drop=True)


def ablation_summary(full, ablated, corr):
    """Values quoted in the paper's ablation table."""
    rows = []

    def add(quantity, metric, full_value, ablated_value):
        rows.append({"quantity": quantity, "metric": metric, "full_7": full_value, "ablation_5": ablated_value})

    add("benchmarks", "", ";".join(sorted(full["benchmark"]["dataset"].unique())),
        ";".join(sorted(ablated["benchmark"]["dataset"].unique())))
    add("benchmark_level_tests_bh_family", "", len(full["benchmark"]), len(ablated["benchmark"]))
    add("benchmark_level_significant_q05", "", int(full["benchmark"]["significant_bh_q05"].sum()),
        int(ablated["benchmark"]["significant_bh_q05"].sum()))
    add("transition_level_tests_bh_family", "", len(full["transition"]), len(ablated["transition"]))

    for metric in METRICS:
        f_tr = full["transition"][full["transition"]["metric"] == metric]
        a_tr = ablated["transition"][ablated["transition"]["metric"] == metric]
        add("total_paired_comparisons", metric, int(_aggregate_row(full, ALL_8, metric)["total_paired_n"]),
            int(_aggregate_row(ablated, ALL_8, metric)["total_paired_n"]))
        for direction, sign in [("negative", -1), ("positive", 1)]:
            f_moved = f_tr[sign * f_tr["drift"] > 0]
            a_moved = a_tr[sign * a_tr["drift"] > 0]
            add(f"transitions_drift_{direction}", metric,
                f"{len(f_moved)}/{len(f_tr)}", f"{len(a_moved)}/{len(a_tr)}")
            add(f"transitions_significant_{direction}_q05", metric,
                f"{(f_moved['q_value'] < ALPHA).sum()}/{len(f_tr)}",
                f"{(a_moved['q_value'] < ALPHA).sum()}/{len(a_tr)}")
        for aggregate_set, label in [(ALL_8, "all_8"), (STRICT_3, "strict_3")]:
            f_agg = _aggregate_row(full, aggregate_set, metric)
            a_agg = _aggregate_row(ablated, aggregate_set, metric)
            add(f"aggregate_{label}_mean_drift", metric, f_agg["mean_drift"], a_agg["mean_drift"])
            add(f"aggregate_{label}_ci_excludes_zero", metric, _excludes_zero(f_agg), _excludes_zero(a_agg))

    for level in ["per_example", "model_benchmark"]:
        c = corr[corr["analysis_level"] == level]
        full_r = c[c["benchmark_set"] == "all_7"]
        ablated_r = c[c["benchmark_set"] == "ablation_5"]
        for (_, f_row), (_, a_row) in zip(full_r.iterrows(), ablated_r.iterrows()):
            pair = f"{f_row['metric_1']}~{f_row['metric_2']}"
            add(f"pearson_r_{level}", pair, f_row["pearson_r"], a_row["pearson_r"])
        add(f"all_abs_r_below_0.2_{level}", "", bool((full_r["pearson_r"].abs() < 0.2).all()),
            bool((ablated_r["pearson_r"].abs() < 0.2).all()))
    return pd.DataFrame(rows)


def ablation(df):
    """paired_drift without StereoSet and CrowS-Pairs (own BH families)."""
    full = paired_drift(df, DATASETS)
    ablated = paired_drift(df, [d for d in DATASETS if d not in ABLATION_EXCLUDED])
    return ablated["benchmark"], ablation_comparison(full, ablated), ablation_summary(full, ablated, correlations(df))
