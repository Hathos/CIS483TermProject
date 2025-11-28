"""
CIS-483 Term Project
Noor Mahmoud
TF-IDF + Logistic Regression baseline.
"""

import argparse
import os
import pickle
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = os.path.join("models", "baseline")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "logreg.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


def load_data(csv_path: str) -> Tuple[pd.Series, pd.Series]:
    """
    Load and clean the labeled CSV. Drops only truly empty labels.
    """
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str).str.strip()

    df = df[df["label"] != ""]
    if df.empty:
        raise ValueError("No labeled rows found. Fill 'label' column first.")

    if df["label"].nunique() < 2:
        raise ValueError("Need at least two distinct labels to train.")

    return df["text"], df["label"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path",
        default=os.path.join("data", "processed", "labeled_samples.csv"),
        help="Path to labeled CSV (default: data/processed/labeled_samples.csv)",
    )
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    X_text, y_str = load_data(args.csv_path)

    label_counts = y_str.value_counts()
    stratify_labels = y_str if label_counts.min() > 1 else None
    if stratify_labels is None:
        print("Warning: Some labels only appear once; skipping stratified split.")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    X_train, X_val, y_train, y_val = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_labels if stratify_labels is not None else None,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # Class weights help when labels are imbalanced
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,
    )
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_val_vec)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    # Save artifacts
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    print(f"Saved baseline model to {MODEL_DIR}")


if __name__ == "__main__":
    main()
