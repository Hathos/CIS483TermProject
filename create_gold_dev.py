"""
CIS-483 Term Project
Noor Mahmoud
Sample a candidate gold dev set from labeled_samples.csv.
"""

import argparse
import os
from typing import List

import pandas as pd

DEFAULT_INPUT = os.path.join("data", "processed", "labeled_samples.csv")
DEFAULT_OUTPUT = os.path.join("data", "processed", "gold_dev_candidates.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default=DEFAULT_INPUT, help="Path to labeled_samples.csv"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Where to write the sampled dev candidates",
    )
    parser.add_argument(
        "--n_samples", type=int, default=400, help="Number of rows to sample"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"{args.input} not found.")

    df = pd.read_csv(args.input)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str)

    if len(df) < args.n_samples:
        sample_df = df.sample(n=len(df), random_state=42)
    else:
        sample_df = df.sample(n=args.n_samples, random_state=42)

    cols: List[str] = ["id", "source_type", "file_name", "chunk_index", "text", "label"]
    sample_df[cols].to_csv(args.output, index=False)
    print(f"Sampled {len(sample_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
