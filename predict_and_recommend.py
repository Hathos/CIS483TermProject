"""
CIS-483 Term Project
Noor Mahmoud
Interactive script to test the classifier and retrieve similar snippets.
"""

import argparse
import json
import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_transformer_model(model_dir: str):
    """Load fine-tuned Transformer and its label mapping."""
    print(f"Loading Transformer model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    labels_path = os.path.join(model_dir, "labels.json")
    with open(labels_path, "r", encoding="utf-8") as f:
        label_info = json.load(f)

    # id2label keys may be strings in JSON, convert to int
    id2label = {int(k): v for k, v in label_info["id2label"].items()}
    label2id = {k: int(v) for k, v in label_info["label2id"].items()}
    classes = label_info.get("classes", sorted(label2id, key=lambda x: label2id[x]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Loaded labels: {classes}")
    return model, tokenizer, id2label, label2id, classes, device


def load_baseline_model(base_dir: str):
    """Load TF-IDF + Logistic Regression baseline and its label encoder."""
    model_dir = os.path.join(base_dir, "models", "baseline")
    vectorizer_path = os.path.join(model_dir, "tfidf.pkl")
    model_path = os.path.join(model_dir, "logreg.pkl")
    le_path = os.path.join(model_dir, "label_encoder.pkl")

    if not (
        os.path.exists(vectorizer_path)
        and os.path.exists(model_path)
        and os.path.exists(le_path)
    ):
        print("Baseline artifacts not found; skipping baseline.")
        return None, None, None

    print(f"Loading baseline model from {model_dir} ...")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    with open(le_path, "rb") as f:
        le = pickle.load(f)

    print(f"Baseline labels: {list(le.classes_)}")
    return vectorizer, clf, le


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def predict_transformer(
    model,
    tokenizer,
    id2label,
    device,
    text: str,
    top_k: int = 2,
    max_length: int = 192,
) -> List[Tuple[str, float]]:
    """Return top-k (label, prob) from the Transformer model."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits[0]
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    indices = probs.argsort()[::-1][:top_k]
    return [(id2label[i], float(probs[i])) for i in indices]


def predict_baseline(
    vectorizer,
    clf,
    le,
    text: str,
    top_k: int = 2,
) -> List[Tuple[str, float]]:
    """Return top-k (label, prob) from the baseline model."""
    X = vectorizer.transform([text])
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
    else:
        # Fallback, use decision_function and softmax
        logits = clf.decision_function(X)[0]
        probs = softmax_np(logits)

    indices = np.argsort(probs)[::-1][:top_k]
    labels = le.inverse_transform(indices)
    return [(labels[i], float(probs[indices[i]])) for i in range(len(indices))]


def load_dataset_for_retrieval(csv_path: str, vectorizer=None):
    """
    Load a labeled dataset (e.g., labeled_samples_simplified.csv) and, if a
    TF-IDF vectorizer is provided, precompute its matrix for similarity search.
    """
    if not os.path.exists(csv_path):
        print(f"Dataset CSV not found at {csv_path}; retrieval disabled.")
        return None, None, None

    print(f"Loading dataset for retrieval from {csv_path} ...")
    df = pd.read_csv(csv_path)
    # Expecting columns id, source_type, file_name, chunk_index, text, label
    required = {"text", "label"}
    if not required.issubset(df.columns):
        print("Dataset is missing required columns; retrieval disabled.")
        return None, None, None

    texts = df["text"].fillna("").tolist()
    labels = df["label"].tolist()

    X = None
    if vectorizer is not None:
        print("Computing TF-IDF matrix for retrieval ...")
        X = vectorizer.transform(texts)

    return df, labels, X


def recommend_snippets(
    query_text: str,
    predicted_label: str,
    df: pd.DataFrame,
    labels: List[str],
    tfidf_matrix,
    vectorizer,
    top_n: int = 3,
) -> List[dict]:
    """
    Find top-n similar snippets from df with the same label using TF-IDF cosine similarity.
    """
    if df is None or tfidf_matrix is None or vectorizer is None:
        return []

    mask = df["label"] == predicted_label
    if mask.sum() == 0:
        return []

    q_vec = vectorizer.transform([query_text])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]

    # Rank by similarity, but only keep rows with the predicted label
    ranked_indices = np.argsort(sims)[::-1]
    recs = []
    for idx in ranked_indices:
        if not mask.iloc[idx]:
            continue
        row = df.iloc[idx]
        recs.append(
            {
                "similarity": float(sims[idx]),
                "source_type": row.get("source_type", ""),
                "file_name": row.get("file_name", ""),
                "chunk_index": (
                    int(row.get("chunk_index", -1))
                    if "chunk_index" in df.columns
                    else -1
                ),
                "text": row.get("text", ""),
            }
        )
        if len(recs) >= top_n:
            break
    return recs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path",
        type=str,
        default=os.path.join("data", "processed", "labeled_samples_simplified.csv"),
        help="CSV with text + label for retrieval examples.",
    )
    parser.add_argument(
        "--transformer-dir",
        type=str,
        default=os.path.join("models", "transformer"),
        help="Directory of the fine-tuned Transformer model.",
    )
    parser.add_argument(
        "--use-baseline",
        action="store_true",
        help="Also load and show predictions from TF-IDF + Logistic Regression.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="How many top labels to display.",
    )
    parser.add_argument(
        "--top-n-snippets",
        type=int,
        default=3,
        help="How many similar snippets to retrieve from the dataset.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load models
    model, tokenizer, id2label, label2id, classes, device = load_transformer_model(
        os.path.join(base_dir, args.transformer_dir)
    )

    vectorizer = None
    clf = None
    le = None
    if args.use_baseline:
        vectorizer, clf, le = load_baseline_model(base_dir)
        if vectorizer is None:
            print("Baseline not available; continuing with Transformer only.")
            args.use_baseline = False

    # Dataset for retrieval (we use the baseline's vectorizer for TF-IDF space)
    df, labels, tfidf_matrix = load_dataset_for_retrieval(
        os.path.join(base_dir, args.csv_path),
        vectorizer=vectorizer,
    )

    print("\nReady. Press Enter on an empty line to exit.\n")

    while True:
        try:
            user_text = input("Enter some text (or empty to quit):\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_text:
            print("Goodbye.")
            break

        # Transformer prediction
        tf_preds = predict_transformer(
            model,
            tokenizer,
            id2label,
            device,
            user_text,
            top_k=args.top_k,
        )
        print("\n[Transformer prediction]")
        for label, prob in tf_preds:
            print(f"  {label:25s}  {prob:.3f}")

        # Baseline prediction (optional)
        if (
            args.use_baseline
            and vectorizer is not None
            and clf is not None
            and le is not None
        ):
            bl_preds = predict_baseline(
                vectorizer, clf, le, user_text, top_k=args.top_k
            )
            print("\n[Baseline prediction]")
            for label, prob in bl_preds:
                print(f"  {label:25s}  {prob:.3f}")

        # Recommendations from dataset for top Transformer label
        if vectorizer is not None and df is not None and tfidf_matrix is not None:
            top_label = tf_preds[0][0]
            recs = recommend_snippets(
                user_text,
                top_label,
                df,
                labels,
                tfidf_matrix,
                vectorizer,
                top_n=args.top_n_snippets,
            )
            if recs:
                print(f"\n[Similar snippets for label '{top_label}']\n")
                for i, rec in enumerate(recs, start=1):
                    src = f"{rec['source_type']} | {rec['file_name']} | chunk {rec['chunk_index']}"
                    text_preview = rec["text"].replace("\n", " ")
                    if len(text_preview) > 220:
                        text_preview = text_preview[:217] + "..."
                    print(f"{i}. sim={rec['similarity']:.3f}  ({src})")
                    print(f"   {text_preview}\n")
            else:
                print(f"\n[No snippets found for label '{top_label}']\n")

        print("-" * 80)


if __name__ == "__main__":
    main()
