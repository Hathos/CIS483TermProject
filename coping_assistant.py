"""
CIS-483 Term Project
Noor Mahmoud
Interactive coping helper: predict label and show snippets from books.
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SKILL_TIPS: Dict[str, Dict[str, List[str]]] = {
    "mindfulness": {
        "summary": [
            "Mindfulness = noticing what is happening in the present moment without judging it.",
            "You are not trying to fix your feelings, just to notice them and let them pass like waves.",
        ],
    },
    "self_compassion": {
        "summary": [
            "Self-compassion = treating yourself like you would treat a close friend who is suffering.",
            "Instead of 'what’s wrong with me', it’s 'this is painful, and it makes sense I feel this way.'",
        ],
    },
    "cognitive_reframe": {
        "summary": [
            "Cognitive reframing = gently questioning the story your brain is telling and considering other possibilities.",
            "It is not pretending everything is great; it is making the story more balanced and less catastrophic.",
        ],
    },
    "attachment_reframe": {
        "summary": [
            "Attachment reframe = noticing when your attachment system is activated and separating present from past.",
            "The goal is to see that not every current trigger is proof that you are unsafe or unloveable.",
        ],
    },
    "communication": {
        "summary": [
            "Communication = expressing your needs and feelings in a clear, honest, and non-attacking way.",
            "Skillful communication gives the relationship a chance to respond to the real issue instead of panic.",
        ],
    },
    "boundary_setting": {
        "summary": [
            "Boundary setting = deciding what you will and will not do, rather than controlling the other person.",
            "Healthy boundaries protect your energy and safety while leaving others free to choose their actions.",
        ],
    },
    "crisis_plan": {
        "summary": [
            "Crisis planning = having a small, concrete toolkit for when your nervous system is overwhelmed.",
            "The goal is to get you out of the red zone safely.",
        ],
    },
    "distress_tolerance": {
        "summary": [
            "Distress tolerance = getting through intense feelings without making things worse.",
            "Focus on short-term coping rather than fixing the whole situation right now.",
        ],
    },
}


def build_retrieval_query(text: str, label: str) -> str:
    tips = SKILL_TIPS.get(label)
    if not tips:
        return text
    summary_snip = " ".join(tips.get("summary", [])[:2])
    return f"{text} {summary_snip}".strip()


def load_transformer_model(model_dir: str):
    print(f"Loading Transformer model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    labels_path = os.path.join(model_dir, "labels.json")
    with open(labels_path, "r", encoding="utf-8") as f:
        label_info = json.load(f)

    id2label = {int(k): v for k, v in label_info["id2label"].items()}
    label2id = {k: int(v) for k, v in label_info["label2id"].items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Available labels: {sorted(label2id.keys())}")
    return model, tokenizer, id2label, label2id, device


def load_baseline_vectorizer(base_dir: str):
    model_dir = os.path.join(base_dir, "models", "baseline")
    vectorizer_path = os.path.join(model_dir, "tfidf.pkl")
    if not os.path.exists(vectorizer_path):
        print("Baseline TF-IDF not found; will fit a new TF-IDF on the dataset.")
        return None

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    print("Loaded baseline TF-IDF vectorizer for retrieval.")
    return vectorizer


def load_dataset(csv_path: str, vectorizer):
    """
    Load the dataset for retrieval. Needs columns: text, label, file_name, source_type.

    Returns:
        df_all: full DataFrame
        tfidf_all: TF-IDF matrix
        vectorizer: fitted TF-IDF (loaded or newly trained)
    """
    if not os.path.exists(csv_path):
        print(f"Dataset CSV not found at {csv_path}; retrieval disabled.")
        return None, None, None

    print(f"Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path)
    required_cols = {"text", "label", "file_name", "source_type"}
    if not required_cols.issubset(df.columns):
        print("Dataset missing required columns; retrieval disabled.")
        return None, None, None

    df["text"] = df["text"].fillna("")

    if vectorizer is None:
        print("Fitting new TF-IDF vectorizer for retrieval ...")
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            min_df=2,
            stop_words="english",
        )
        tfidf_all = vectorizer.fit_transform(df["text"].tolist())
    else:
        print("Computing TF-IDF matrix for retrieval ...")
        tfidf_all = vectorizer.transform(df["text"].tolist())

    return df, tfidf_all, vectorizer


def predict_transformer(
    model,
    tokenizer,
    id2label,
    device,
    text: str,
    top_k: int = 2,
    max_length: int = 192,
) -> List[Tuple[str, float]]:
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


def recommend_label_snippets(
    query_text: str,
    predicted_label: str,
    df: pd.DataFrame,
    tfidf_all,
    vectorizer,
    top_n: int = 3,
    sources: Tuple[str, ...] = ("journal", "book"),
    min_sim: float = 0.2,
) -> List[dict]:
    if df is None or tfidf_all is None or vectorizer is None:
        return []

    mask_source = df["source_type"].isin(sources)
    if mask_source.sum() == 0:
        mask_source = np.ones(len(df), dtype=bool)

    mask_label = df["label"] == predicted_label
    mask = mask_source & mask_label
    if mask.sum() == 0:
        return []

    candidate_idx = np.where(mask)[0]
    candidate_matrix = tfidf_all[candidate_idx]

    rich_query = build_retrieval_query(query_text, predicted_label)
    q_vec = vectorizer.transform([rich_query])
    sims = cosine_similarity(q_vec, candidate_matrix)[0]

    ranked = np.argsort(sims)[::-1]
    recs = []
    for r in ranked:
        idx = candidate_idx[r]
        row = df.iloc[idx]
        if sims[r] < min_sim:
            continue
        recs.append(
            {
                "similarity": float(sims[r]),
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

    if not recs:
        for r in ranked:
            idx = candidate_idx[r]
            row = df.iloc[idx]
            recs.append(
                {
                    "similarity": float(sims[r]),
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


def print_skill_help(label: str, prob: float):
    tips = SKILL_TIPS.get(label)
    print(f"\n== Suggested skill: {label} (model confidence {prob:.2f}) ==")
    if not tips or not tips.get("summary"):
        print("No stored description yet.")
        return
    print(f"  {tips['summary'][0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path",
        type=str,
        default=os.path.join("data", "processed", "labeled_samples_simplified.csv"),
        help="CSV with text+label for retrieval.",
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
        help="If baseline TF-IDF exists, load it; otherwise fit TF-IDF on the dataset.",
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
        default=2,
        help="How many book snippets to show.",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model, tokenizer, id2label, label2id, device = load_transformer_model(
        os.path.join(base_dir, args.transformer_dir)
    )

    vectorizer = load_baseline_vectorizer(base_dir) if args.use_baseline else None
    df_all, tfidf_all, vectorizer = load_dataset(
        os.path.join(base_dir, args.csv_path),
        vectorizer=vectorizer,
    )

    print("\nCoping assistant ready.")
    print("Type what is going on (a few sentences).")
    print("Press Enter on an empty line to exit.\n")

    while True:
        try:
            text = input("What's going on right now?\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not text:
            print("Goodbye.")
            break

        preds = predict_transformer(
            model, tokenizer, id2label, device, text, top_k=args.top_k
        )

        print("\n[Model's best guesses]")
        for lbl, p in preds:
            print(f"  {lbl:20s}  {p:.3f}")

        top_label, top_prob = preds[0]
        print_skill_help(top_label, top_prob)

        if vectorizer is not None and df_all is not None and tfidf_all is not None:
            book_recs = recommend_label_snippets(
                text,
                top_label,
                df_all,
                tfidf_all,
                vectorizer,
                top_n=args.top_n_snippets,
                sources=("book",),
                min_sim=0.15,
            )

            if book_recs:
                print(f"\nFrom your books for '{top_label}':\n")
                for i, rec in enumerate(book_recs, start=1):
                    preview = rec["text"].replace("\n", " ")
                    if len(preview) > 260:
                        preview = preview[:257] + "..."
                    print(
                        f"{i}. (chunk {rec['chunk_index']} in {rec['file_name']}, sim={rec['similarity']:.3f})"
                    )
                    print(f"   {preview}\n")

        print("-" * 80)


if __name__ == "__main__":
    main()
