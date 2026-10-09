
"""Function 1 - Emotional Pattern Clustering Pipeline"""

import argparse
import json
from pathlib import Path

from src.emotional_pattern import run_dashboard_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to educator sessions CSV")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns, trends = run_dashboard_outputs(args.csv)

    patterns.to_csv(
        output_dir / "child_emotional_patterns.csv"
    )

    with open(
        output_dir / "emotion_trend.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(trends, f, indent=2)

    print("Emotional Pattern Results:")
    print(patterns["pattern"].value_counts())

    print("\nWorsening Children:")
    print([
        child for child, result in trends.items()
        if result["worsening"]
    ])

    print("\nOutputs saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
