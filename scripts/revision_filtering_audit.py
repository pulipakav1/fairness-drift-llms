"""
Revision audit: recompute the true filtering rate and its breakdown.

The paper claims "filtering rate: 1.34%, concentrated in CrowS-Pairs and BBQ".
This script checks that claim against the actual raw -> scored data and writes
a corrected breakdown to outputs/tables/revision_filtering_breakdown.csv.
"""
import pandas as pd

RAW_PATH = "outputs/results/raw_results_20251115_035107.csv"
SCORED_PATH = "outputs/results/results_with_metrics_20251115_035107.csv"

KEY_COLS = ["provider", "model", "dataset", "prompt_id"]


def classify_reason(response):
    if not isinstance(response, str) or response.strip() == "":
        return "empty"
    r = response.strip().lower()
    if r.startswith("error:"):
        if "insufficient_quota" in r or "429" in r:
            return "api_quota_error"
        return "api_error_other"
    if "i cannot" in r or "i can't" in r or "i'm not able to" in r or "i am not able to" in r:
        return "refusal_pattern"
    if "policy" in r and ("block" in r or "violat" in r):
        return "policy_block"
    return "other_filtered"


def main():
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    scored = pd.read_csv(SCORED_PATH, low_memory=False)

    raw_keys = raw.set_index(KEY_COLS)
    scored_key_set = set(map(tuple, scored[KEY_COLS].values))

    filtered_mask = ~raw_keys.index.isin(scored_key_set)
    filtered = raw_keys[filtered_mask].reset_index()

    total_raw = len(raw)
    total_filtered = len(filtered)
    rate = total_filtered / total_raw * 100

    filtered["reason"] = filtered["response"].apply(classify_reason)

    print(f"Total raw rows: {total_raw}")
    print(f"Total filtered rows: {total_filtered}")
    print(f"True filtering rate: {rate:.2f}%")
    print()
    print("By reason:")
    print(filtered["reason"].value_counts())
    print()
    print("By dataset:")
    print(filtered.groupby("dataset").size().sort_values(ascending=False))
    print()
    print("By provider x dataset:")
    by_pd = filtered.groupby(["provider", "dataset"]).size().sort_values(ascending=False)
    print(by_pd)
    print()
    print("By reason x provider x dataset (top 15):")
    print(filtered.groupby(["reason", "provider", "dataset"]).size().sort_values(ascending=False).head(15))

    filtered.to_csv("outputs/tables/revision_filtering_breakdown.csv", index=False)

    # CrowS-Pairs + BBQ share, as claimed in the paper
    cb = filtered[filtered["dataset"].isin(["crows_pairs", "bbq"])]
    print()
    print(f"CrowS-Pairs + BBQ share of filtered rows: {len(cb)}/{total_filtered} = {len(cb)/total_filtered*100:.1f}%")
    bold_openai = filtered[(filtered["dataset"] == "bold") & (filtered["provider"] == "openai")]
    print(f"OpenAI+BOLD share of filtered rows: {len(bold_openai)}/{total_filtered} = {len(bold_openai)/total_filtered*100:.1f}%")


if __name__ == "__main__":
    main()
