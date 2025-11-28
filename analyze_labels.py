"""
CIS-483 Term Project
Noor Mahmoud
Quick label exploration for labeled_samples.csv (counts + optional baseline eval).
"""

import os
import pickle
from collections import Counter
from typing import Optional

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

DEFAULT_CSV = os.path.join("data", "processed", "labeled_samples.csv")
BASELINE_DIR = os.path.join("models", "baseline")
VECTORIZER_PATH = os.path.join(BASELINE_DIR, "tfidf.pkl")
MODEL_PATH = os.path.join(BASELINE_DIR, "logreg.pkl")
ENCODER_PATH = os.path.join(BASELINE_DIR, "label_encoder.pkl")


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str).str.strip()
    df = df[df["label"] != ""]
    return df


def print_label_counts(df: pd.DataFrame) -> None:
    counts = Counter(df["label"])
    total = sum(counts.values())
    print(f"Total rows: {total}")
    print("\nLabel frequencies:")
    for label, count in counts.most_common():
        print(f"{label:30s} {count}")


def load_baseline() -> Optional[tuple]:
    if not (
        os.path.exists(VECTORIZER_PATH)
        and os.path.exists(MODEL_PATH)
        and os.path.exists(ENCODER_PATH)
    ):
        return None
    with open(VECTORIZER_PATH, "rb") as f:
        vec = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return vec, model, encoder


def run_baseline_eval(df: pd.DataFrame) -> None:
    loaded = load_baseline()
    if loaded is None:
        print(
            "\nBaseline artifacts not found under models/baseline/. Skipping baseline eval."
        )
        return
    vec, model, encoder = loaded
    X = vec.transform(df["text"])
    y_true = encoder.transform(df["label"])
    y_pred = model.predict(X)
    labels = list(encoder.classes_)

    print("\nBaseline classification report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nConfusion matrix (rows=true, cols=pred):")
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(cm_df)


def main() -> None:
    csv_path = DEFAULT_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    df = load_data(csv_path)
    print_label_counts(df)
    run_baseline_eval(df)


if __name__ == "__main__":
    main()
