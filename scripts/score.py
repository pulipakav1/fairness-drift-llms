import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import SCORES_PATH  # noqa: E402
from src.data import filter_responses, load_raw  # noqa: E402
from src.metrics import add_scores  # noqa: E402


def main():
    raw = load_raw()
    scores = add_scores(filter_responses(raw))
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORES_PATH, index=False, lineterminator="\n")
    print(f"Scored {len(scores):,} of {len(raw):,} responses")


if __name__ == "__main__":
    main()
