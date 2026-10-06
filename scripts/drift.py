import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import context_drift, paired_drift  # noqa: E402
from src.config import DATASETS  # noqa: E402
from src.data import load_scores, save_table  # noqa: E402


def main():
    df = load_scores()
    results = paired_drift(df, DATASETS)
    save_table(results["benchmark"], "benchmark_drift.csv")
    save_table(results["transition"], "transition_drift.csv")
    save_table(results["aggregate"], "aggregate_drift.csv")
    save_table(results["pooled"], "pooled_drift.csv")
    save_table(context_drift(df), "context_drift.csv")


if __name__ == "__main__":
    main()
