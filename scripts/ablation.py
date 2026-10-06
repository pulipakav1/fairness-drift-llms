import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import ablation  # noqa: E402
from src.data import load_scores, save_table  # noqa: E402


def main():
    benchmark, comparison, summary = ablation(load_scores())
    save_table(benchmark, "ablation_benchmark_drift.csv")
    save_table(comparison, "ablation.csv")
    save_table(summary, "ablation_summary.csv")


if __name__ == "__main__":
    main()
