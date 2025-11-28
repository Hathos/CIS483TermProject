"""
CIS-483 Term Project
Noor Mahmoud
Auto-clean gold_dev_candidates.csv using the label simplifier.
"""

import argparse
import os
from collections import Counter
from typing import Dict

import pandas as pd

DEFAULT_INPUT = os.path.join("data", "processed", "gold_dev_candidates.csv")
DEFAULT_OUTPUT = os.path.join("data", "processed", "gold_dev.csv")

# Reuse the simplified mapping
# Note, some labels are merged into others
LABEL_MAP: Dict[str, str] = {
    "attachment_reframe": "attachment_reframe",
    "cognitive_reframe": "cognitive_reframe",
    "mindfulness": "mindfulness",
    "grounding": "mindfulness",
    "self_compassion": "self_compassion",
    "self_soothing": "self_compassion",
    "communication": "communication",
    "boundary_setting": "boundary_setting",
    "crisis_plan": "crisis_plan",
    "distress_tolerance": "distress_tolerance",
    "psychoeducation": "communication",
    "relationship_reflection": "communication",
    "other": "other",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default=DEFAULT_INPUT, help="Path to gold_dev_candidates.csv"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="Where to write gold_dev.csv"
    )
    parser.add_argument(
        "--drop-other",
        action="store_true",
        help="If set, drop rows mapped to 'other' instead of keeping them.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"{args.input} not found.")

    df = pd.read_csv(args.input)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str).str.strip()

    def map_label(lbl: str) -> str:
        return LABEL_MAP.get(lbl, "other")

    df["label"] = df["label"].apply(map_label)
    if args.drop_other:
        df = df[df["label"] != "other"]

    before = len(df)
    df = df[df["label"] != ""]
    after = len(df)

    counts = Counter(df["label"])
    print(f"Rows kept: {after} (dropped {before - after})")
    print("Label counts (gold dev, auto-cleaned):")
    for lbl, cnt in counts.most_common():
        print(f"{lbl:25s} {cnt}")

    keep_cols = ["id", "source_type", "file_name", "chunk_index", "text", "label"]
    df[keep_cols].to_csv(args.output, index=False)
    print(f"Wrote gold dev to {args.output}")


if __name__ == "__main__":
    main()
