import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis import filtering_summary  # noqa: E402
from src.data import load_raw, save_table  # noqa: E402


def main():
    save_table(filtering_summary(load_raw()), "filtering_summary.csv")


if __name__ == "__main__":
    main()
