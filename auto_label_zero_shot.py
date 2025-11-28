"""
CIS-483 Term Project
Noor MahmoudZero-shot labeler to tag raw_samples.csv with draft labels.
"""

import argparse
import os
from typing import Iterable, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

DATA_DIR = os.path.join("data", "processed")
INPUT_CSV = os.path.join(DATA_DIR, "raw_samples.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "auto_labeled.csv")

# Canonical labels
LABEL_KEYS: List[str] = [
    "self_soothing",
    "cognitive_reframe",
    "attachment_reframe",
    "mindfulness",
    "grounding",
    "distress_tolerance",
    "communication",
    "boundary_setting",
    "crisis_plan",
    "self_compassion",
    "psychoeducation",
    "relationship_reflection",
]
OTHER_LABEL = "other"

# Descriptive prompts help zero-shot models disambiguate
LABEL_PROMPTS = {
    "self_soothing": "self-soothing skills (comforting yourself, calming touch, safe place imagery, kind self-talk)",
    "cognitive_reframe": "cognitive reframing (challenging thoughts, alternative explanations, balanced thinking)",
    "attachment_reframe": "attachment reframing (secure-base mindset, revisiting attachment beliefs, repairing attachment wounds)",
    "mindfulness": "mindfulness practice (non-judgmental awareness, noticing sensations, observing thoughts)",
    "grounding": "grounding techniques (5-4-3-2-1 senses, describing the room, orienting to the present)",
    "distress_tolerance": "distress tolerance (urge surfing, ride the wave, radical acceptance, crisis survival)",
    "communication": "communication skills (expressing needs, active listening, conflict resolution, assertive requests)",
    "boundary_setting": "boundary setting (saying no, limits, consequences, protecting time/space)",
    "crisis_plan": "crisis planning (safety steps, reaching out for help, emergency coping plan)",
    "self_compassion": "self-compassion (kind inner voice, common humanity, self-forgiveness, supportive statements)",
    "psychoeducation": "psychoeducation or explanatory content (definitions, background theory, context, explanation, not a specific coping technique)",
    "relationship_reflection": "direct personal/supportive dialogue from chats (feelings about connection, describing interactions, processing dynamics)",
}

MODEL_NAME = "cross-encoder/nli-deberta-v3-large"
# Default threshold
MIN_SCORE = 0.55
# Optional per-label thresholds; if empty, all use MIN_SCORE
LABEL_THRESHOLDS = {
    "attachment_reframe": 0.60,
    "self_compassion": 0.60,
    "psychoeducation": 0.60,
    "relationship_reflection": 0.60,
}
# How many top labels to keep
TOP_K = 3


def batched(iterable: Iterable, batch_size: int) -> Iterable[List]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_candidates() -> Tuple[List[str], dict]:
    """
    Build candidate strings with descriptions and a map back to canonical keys.
    """
    candidate_strings: List[str] = []
    str_to_key = {}
    for key in LABEL_KEYS:
        desc = LABEL_PROMPTS.get(key, key)
        s = f"{key}: {desc}"
        candidate_strings.append(s)
        str_to_key[s] = key
    return candidate_strings, str_to_key


def label_batch(
    classifier,
    texts: List[str],
    source_types: List[str],
    file_names: List[str],
    candidate_strings: List[str],
    str_to_key: dict,
    pipe_batch_size: int,
):
    """
    Run zero-shot classification with multi-label, keep top-k above thresholds.
    Returns lists for top1 label/score and optional second best.
    """
    results = classifier(
        texts,
        candidate_strings,
        multi_label=True,
        batch_size=pipe_batch_size,
    )

    top1_labels: List[str] = []
    top1_scores: List[float] = []
    top2_labels: List[str] = []
    top2_scores: List[float] = []
    top3_labels: List[str] = []
    top3_scores: List[float] = []

    def threshold_for(key: str, source_type: str) -> float:
        base = LABEL_THRESHOLDS.get(key, MIN_SCORE)
        # Personal relationship talk is only valid for journals/logs,effectively disable for books.
        if key == "relationship_reflection" and source_type.lower() == "book":
            return 10.0  # impossible to hit, routes book content away from this label
        return base

    for res, src, fname in zip(results, source_types, file_names):
        pairs = [
            (str_to_key[lbl], float(scr))
            for lbl, scr in zip(res["labels"], res["scores"])
        ]
        filtered = []
        for key, scr in pairs:
            th = threshold_for(key, src or "")
            if scr >= th:
                filtered.append((key, scr))
        filtered.sort(key=lambda x: x[1], reverse=True)

        is_support_log = "discussions_with_" in (fname or "").lower()

        if is_support_log:
            # Always include relationship_reflection, keep best others as secondary.
            non_rr = [(k, s) for k, s in filtered if k != "relationship_reflection"]
            top1_labels.append("relationship_reflection")
            top1_scores.append(0.99)
            if non_rr:
                top2_labels.append(non_rr[0][0])
                top2_scores.append(non_rr[0][1])
            else:
                top2_labels.append("")
                top2_scores.append(0.0)

            if len(non_rr) > 1 and TOP_K > 2:
                top3_labels.append(non_rr[1][0])
                top3_scores.append(non_rr[1][1])
            else:
                top3_labels.append("")
                top3_scores.append(0.0)
        else:
            if not filtered:
                top1_labels.append(OTHER_LABEL)
                top1_scores.append(0.0)
                top2_labels.append("")
                top2_scores.append(0.0)
                top3_labels.append("")
                top3_scores.append(0.0)
                continue

            top1_labels.append(filtered[0][0])
            top1_scores.append(filtered[0][1])

            if TOP_K > 1 and len(filtered) > 1:
                top2_labels.append(filtered[1][0])
                top2_scores.append(filtered[1][1])
            else:
                top2_labels.append("")
                top2_scores.append(0.0)

            if TOP_K > 2 and len(filtered) > 2:
                top3_labels.append(filtered[2][0])
                top3_scores.append(filtered[2][1])
            else:
                top3_labels.append("")
                top3_scores.append(0.0)

    return top1_labels, top1_scores, top2_labels, top2_scores, top3_labels, top3_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=INPUT_CSV,
        help="Path to raw_samples.csv (default: data/processed/raw_samples.csv)",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_CSV,
        help="Where to write auto-labeled CSV (default: data/processed/auto_labeled.csv)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows to process (for quick tests).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Pandas chunk size when streaming the CSV.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Outer batch size (texts per call into the pipeline).",
    )
    parser.add_argument(
        "--pipe-batch-size",
        type=int,
        default=16,
        help="Inner pipeline batch size for model forward passes.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading zero-shot classifier: {MODEL_NAME}")

    # Use GPU if available, otherwise CPU.
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else None

    candidate_strings, str_to_key = build_candidates()

    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=device,
        dtype=dtype,
    )

    total_processed = 0
    first_write = True

    reader = pd.read_csv(args.input, chunksize=args.chunk_size)
    for chunk in reader:
        if args.max_rows is not None and total_processed >= args.max_rows:
            break

        texts = chunk["text"].astype(str).tolist()
        source_types = (
            chunk.get("source_type", pd.Series([""] * len(chunk)))
            .fillna("")
            .astype(str)
            .tolist()
        )
        file_names = (
            chunk.get("file_name", pd.Series([""] * len(chunk)))
            .fillna("")
            .astype(str)
            .tolist()
        )
        preds: List[str] = []
        scores: List[float] = []
        preds2: List[str] = []
        scores2: List[float] = []
        preds3: List[str] = []
        scores3: List[float] = []

        for batch in tqdm(
            list(batched(texts, args.batch_size)),
            desc="Labeling",
            leave=False,
        ):
            src_batch = source_types[len(preds) : len(preds) + len(batch)]
            fname_batch = file_names[len(preds) : len(preds) + len(batch)]
            b_preds, b_scores, b_preds2, b_scores2, b_preds3, b_scores3 = label_batch(
                classifier,
                batch,
                src_batch,
                fname_batch,
                candidate_strings,
                str_to_key,
                args.pipe_batch_size,
            )
            preds.extend(b_preds)
            scores.extend(b_scores)
            preds2.extend(b_preds2)
            scores2.extend(b_scores2)
            preds3.extend(b_preds3)
            scores3.extend(b_scores3)

        chunk = chunk.copy()
        chunk["predicted_label"] = preds
        chunk["predicted_score"] = scores
        chunk["predicted_label_2"] = preds2
        chunk["predicted_score_2"] = scores2
        chunk["predicted_label_3"] = preds3
        chunk["predicted_score_3"] = scores3

        if args.max_rows is not None:
            remaining = args.max_rows - total_processed
            if remaining < len(chunk):
                chunk = chunk.iloc[:remaining]

        chunk.to_csv(
            args.output,
            mode="w" if first_write else "a",
            index=False,
            header=first_write,
        )
        first_write = False
        total_processed += len(chunk)

        if args.max_rows is not None and total_processed >= args.max_rows:
            break

    print(f"Wrote {total_processed} rows to {args.output}")
    print("Note: labels are heuristic. Review/clean before training.")


if __name__ == "__main__":
    main()
