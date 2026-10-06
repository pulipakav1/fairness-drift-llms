import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation  # noqa: E402
import correlations  # noqa: E402
import demographics  # noqa: E402
import drift  # noqa: E402
import filtering  # noqa: E402
import provider_scores  # noqa: E402
import score  # noqa: E402
from src.config import RAW_PATH, TABLES_DIR  # noqa: E402

TABLES = [
    "benchmark_drift.csv", "transition_drift.csv", "aggregate_drift.csv", "pooled_drift.csv", "context_drift.csv",
    "demographic_drift.csv", "demographic_counts.csv", "ablation_benchmark_drift.csv", "ablation.csv",
    "ablation_summary.csv", "correlations.csv", "provider_scores.csv", "filtering_summary.csv",
]


def main():
    if not RAW_PATH.is_file():
        sys.exit(f"Missing {RAW_PATH} (not distributed; see README).")
    for step in [score, drift, demographics, ablation, correlations, provider_scores, filtering]:
        step.main()

    print("\nSHA256")
    for name in TABLES:
        print(f"  {hashlib.sha256((TABLES_DIR / name).read_bytes()).hexdigest()}  {name}")
    print("Done")


if __name__ == "__main__":
    main()
