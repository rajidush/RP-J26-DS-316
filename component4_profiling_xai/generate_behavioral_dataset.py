"""Function 2 - generate the synthetic behavioral training dataset (Step 2).

    python generate_behavioral_dataset.py            # writes data/
    python generate_behavioral_dataset.py --out path/to/dir --seed 42
"""

import argparse
from pathlib import Path

from src.behavioral_synthetic import SEED, build_dataset, write_outputs

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "data"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    result = build_dataset(args.seed)
    paths = write_outputs(result, args.out)
    meta = result.metadata

    print(f"{meta['rows']} child-days, {meta['children']} children, seed {meta['seed']} (SYNTHETIC)")
    print(f"Anomaly days: {meta['anomaly_rate']:.2%}  {meta['anomaly_days_by_type']}")
    for name, split in meta["splits"].items():
        print(f"  {name:<10} {split['dates'][0]} .. {split['dates'][1]}  "
              f"rows {split['rows']}  anomaly days {split['anomaly_days']}")
    for path in paths.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
