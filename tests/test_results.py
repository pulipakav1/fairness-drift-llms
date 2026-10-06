import numpy as np
import pandas as pd
import pytest

from src.config import ALPHA, RAW_PATH, STRICT_TRANSITIONS, TABLES_DIR
from src.data import filter_responses, load_raw
from src.stats import adjust_bh


def table(name):
    return pd.read_csv(TABLES_DIR / name)


def test_bh_family_sizes():
    assert len(table("benchmark_drift.csv")) == 156
    assert len(table("transition_drift.csv")) == 24
    assert len(table("demographic_drift.csv")) == 144
    assert len(table("ablation_benchmark_drift.csv")) == 108


@pytest.mark.parametrize("name", ["benchmark_drift.csv", "transition_drift.csv", "demographic_drift.csv",
                                  "ablation_benchmark_drift.csv"])
def test_q_values_are_bh_within_table(name):
    t = table(name)
    assert np.allclose(adjust_bh(t[["p_value"]])["q_value"], t["q_value"], rtol=1e-12, atol=0)


def test_demographic_counts():
    counts = table("demographic_counts.csv").set_index("label")
    expected = {"ability": 100, "disability": 11, "Age": 100, "age": 2, "gender_identity": 100, "gender": 33,
                "race": 51, "race-color": 34, "nationality": 10, "socioeconomic": 10, "religion": 3}
    assert counts["prompts_per_model"].to_dict() == expected
    assert counts.groupby("axis")["paired_n"].first().to_dict() == {
        "Age": 102, "Disability": 111, "Gender": 133, "Nationality": 10, "Race/Ethnicity": 85,
        "Religion": 3, "Socioeconomic": 10}
    assert set(counts.loc[~counts["eligible_for_testing"], "axis"]) == {"Religion"}


def test_strict_flags_and_pair_counts():
    cells = table("benchmark_drift.csv")
    strict = set(cells.loc[cells["strict_successor"], ["family", "model_t", "model_t1"]].itertuples(index=False))
    assert strict == STRICT_TRANSITIONS
    assert cells.groupby("metric")["n_paired"].sum().unique().tolist() == [5193]
    assert cells["significant_bh_q05"].sum() == (cells["q_value"] < ALPHA).sum() == 15


@pytest.mark.skipif(not RAW_PATH.is_file(), reason="raw responses are not distributed")
def test_filtering_raw_responses():
    raw = load_raw()
    kept = filter_responses(raw)
    assert (len(raw), len(kept)) == (13300, 12789)
    assert not ((kept["provider"] == "openai") & (kept["dataset"] == "bold")).any()
