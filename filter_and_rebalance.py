"""
CIS-483 Term Project
Noor Mahmoud
Filter noisy auto_labeled.csv and write a cleaner labeled_samples_filtered.csv.
"""

import argparse
import os
from collections import Counter
from typing import List

import pandas as pd

AUTO_PATH = os.path.join("data", "processed", "auto_labeled.csv")
OUTPUT_PATH = os.path.join("data", "processed", "labeled_samples_filtered.csv")


def choose_label(row: pd.Series, min_score: float) -> str:
    """
    Pick the best label across predicted_label/_2/_3 if score >= min_score.
    """
    candidates = []
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
        if lbl and score >= min_score:
            candidates.append((lbl, score))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def apply_cap(df: pd.DataFrame, max_per_label: int | None) -> pd.DataFrame:
    if max_per_label is None:
        return df
    frames: List[pd.DataFrame] = []
    for label, group in df.groupby("label"):
        if len(group) > max_per_label:
            frames.append(group.sample(n=max_per_label, random_state=42))
        else:
            frames.append(group)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=AUTO_PATH, help="Path to auto_labeled.csv")
    parser.add_argument(
        "--output", default=OUTPUT_PATH, help="Where to write filtered CSV"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="Minimum confidence to keep a label",
    )
    parser.add_argument(
        "--min-count", type=int, default=30, help="Minimum count to keep a label"
    )
    parser.add_argument(
        "--drop-rare",
        action="store_true",
        help="If set, drop labels with count < min-count. Otherwise merge them to 'other'",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=None,
        help="Cap examples per label (default: no cap)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"{args.input} not found.")

    df = pd.read_csv(args.input)
    df["text"] = df["text"].fillna("").astype(str)

    chosen_labels: List[str] = []
    for _, row in df.iterrows():
        lbl = choose_label(row, min_score=args.min_score)
        chosen_labels.append(lbl)
    df["label"] = chosen_labels

    before = len(df)
    df = df[df["label"] != ""]
    dropped_low_score = before - len(df)

    counts = Counter(df["label"])
    rare_labels = [lbl for lbl, cnt in counts.items() if cnt < args.min_count]

    if rare_labels:
        if args.drop_rare:
            df = df[~df["label"].isin(rare_labels)]
        else:
            df.loc[df["label"].isin(rare_labels), "label"] = "other"

    # Re-cap frequencies after rare handling
    df = apply_cap(df, args.max_per_label)

    # Summary
    print(f"Initial rows: {before}")
    print(f"Dropped for low score: {dropped_low_score}")
    print(f"Final rows: {len(df)}")
    print("Label counts:")
    for lbl, cnt in sorted(
        Counter(df["label"]).items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{lbl:30s} {cnt}")
    if rare_labels:
        print(f"Rare labels (<{args.min_count}): {rare_labels}")
        print(
            "Dropped rare labels"
            if args.drop_rare
            else "Merged rare labels into 'other'"
        )

    keep_cols = ["id", "source_type", "file_name", "chunk_index", "text", "label"]
    df[keep_cols].to_csv(args.output, index=False)
    print(f"Wrote filtered dataset to {args.output}")


if __name__ == "__main__":
    main()
