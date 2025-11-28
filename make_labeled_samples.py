"""
CIS-483 Term Project
Noor Mahmoud
Turn auto_labeled.csv into labeled_samples.csv for training.
"""

import argparse
import os
from typing import List, Optional, Tuple

import pandas as pd

DEFAULT_INPUT = os.path.join("data", "processed", "auto_labeled.csv")
DEFAULT_OUTPUT = os.path.join("data", "processed", "labeled_samples.csv")


def choose_label(
    row: pd.Series, min_score: float, keep_other: bool
) -> Tuple[Optional[str], Optional[float]]:
    """
    Pick the highest scoring label among top1/top2/top3.
    Returns (label, score) or (None, None) if no label passes thresholds.
    """
    candidates: List[Tuple[str, float]] = []
    for lbl_col, score_col in [
        ("predicted_label", "predicted_score"),
        ("predicted_label_2", "predicted_score_2"),
        ("predicted_label_3", "predicted_score_3"),
    ]:
        lbl = str(row.get(lbl_col, "") or "").strip()
        try:
            score = float(row.get(score_col, 0.0))
        except (TypeError, ValueError):
            score = 0.0
        if lbl:
            candidates.append((lbl, score))

    if not candidates:
        return None, None

    # Sort by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_label, best_score = candidates[0]

    if best_score < min_score:
        return None, None
    if not keep_other and best_label == "other":
        return None, None

    return best_label, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default=DEFAULT_INPUT, help="Path to auto_labeled.csv"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="Path to write labeled_samples.csv"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.6, help="Minimum score to keep a label"
    )
    parser.add_argument(
        "--keep-other",
        action="store_true",
        help="If set, keep rows even when best label is 'other'",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    chosen_labels = []
    chosen_scores = []
    for _, row in df.iterrows():
        lbl, score = choose_label(
            row, min_score=args.min_score, keep_other=args.keep_other
        )
        chosen_labels.append(lbl)
        chosen_scores.append(score)

    df["label"] = chosen_labels
    df["label_score"] = chosen_scores

    # Drop rows where no label survived
    before = len(df)
    df = df.dropna(subset=["label"])
    after = len(df)
    print(f"Dropped {before - after} rows that did not meet thresholds.")

    # Keep only the columns the trainers expect plus optional score
    keep_cols = [
        "id",
        "source_type",
        "file_name",
        "chunk_index",
        "text",
        "label",
        "label_score",
    ]
    df_out = df[keep_cols]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"Wrote {len(df_out)} rows to {args.output}")
    print(
        "You can now run train_baseline.py and train_transformer.py against this file."
    )


if __name__ == "__main__":
    main()
