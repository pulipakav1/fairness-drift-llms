import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import provider_scores  # noqa: E402
from src.data import load_scores, save_table  # noqa: E402


def main():
    save_table(provider_scores(load_scores()), "provider_scores.csv")


if __name__ == "__main__":
    main()
