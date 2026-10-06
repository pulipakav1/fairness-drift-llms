import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import demographic_counts, demographic_drift  # noqa: E402
from src.data import load_scores, save_table  # noqa: E402


def main():
    df = load_scores()
    save_table(demographic_drift(df), "demographic_drift.csv")
    save_table(demographic_counts(df), "demographic_counts.csv")


if __name__ == "__main__":
    main()
