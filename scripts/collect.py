import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATASETS, FAMILIES, PROVIDERS  # noqa: E402
from src.data import LOADERS  # noqa: E402
from src.models import query_model  # noqa: E402


def main():
    provider = {model: PROVIDERS[family] for family, models in FAMILIES.items() for model in models}

    parser = argparse.ArgumentParser(description="Collect raw model responses.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(provider), choices=list(provider))
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        prompts = LOADERS[dataset]()
        for model in args.models:
            # prompt_id = position in list
            for prompt_id, item in enumerate(tqdm(prompts, desc=f"{dataset} / {model}")):
                metadata = {k: v for k, v in item.items() if k != "prompt"}
                rows.append({
                    "timestamp": datetime.now().isoformat(),
                    "provider": provider[model],
                    "model": model,
                    "dataset": dataset,
                    "prompt_id": prompt_id,
                    "prompt": item["prompt"],
                    "response": query_model(provider[model], model, item["prompt"], args.temperature),
                    **metadata,
                })
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {len(rows):,} responses to {args.out}")


if __name__ == "__main__":
    main()
