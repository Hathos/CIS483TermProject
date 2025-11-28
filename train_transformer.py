"""
CIS-483 Term Project
Noor Mahmoud
Fine-tune a transformer classifier on labeled data.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    DebertaV2Tokenizer,
    Trainer,
    TrainingArguments,
)

DATA_DIR = os.path.join("data", "processed")
DEFAULT_CSV = os.path.join(DATA_DIR, "labeled_samples.csv")
MODEL_OUT_DIR = os.path.join("models", "transformer")
GOLD_DEV_PATH = os.path.join(DATA_DIR, "gold_dev.csv")


def load_labeled_data(csv_path: str) -> pd.DataFrame:
    """
    Load labeled samples and drop blanks.
    """
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].fillna("").astype(str).str.strip()

    df = df[df["label"] != ""]
    if df.empty:
        raise ValueError("No labeled rows found.")
    if df["label"].nunique() < 2:
        raise ValueError("Need at least two distinct labels to fine-tune.")

    return df


@dataclass
class EncodedDataset:
    train: Dataset
    val: Dataset
    label_encoder: LabelEncoder


def split_train_val(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    label_counts = df["label"].value_counts()
    stratify_labels = df["label"] if label_counts.min() > 1 else None
    if stratify_labels is None:
        print("Warning: Some labels only appear once; skipping stratified split.")

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=stratify_labels if stratify_labels is not None else None,
    )
    return train_df, val_df


def build_datasets(
    csv_path: str, model_name: str = "distilbert-base-uncased"
) -> EncodedDataset:
    df = load_labeled_data(csv_path)
    train_df, val_df = split_train_val(df)

    le = LabelEncoder()
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["label_id"] = le.fit_transform(train_df["label"])
    val_df["label_id"] = le.transform(val_df["label"])

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=False, trust_remote_code=True
            )
        except Exception:
            if "deberta" in model_name.lower():
                tokenizer = DebertaV2Tokenizer.from_pretrained(
                    model_name, use_fast=False
                )
            else:
                raise

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=192,
        )

    train_dataset = Dataset.from_pandas(
        train_df[["text", "label_id"]], preserve_index=False
    ).rename_column("label_id", "labels")
    val_dataset = Dataset.from_pandas(
        val_df[["text", "label_id"]], preserve_index=False
    ).rename_column("label_id", "labels")

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=cols)
    val_dataset.set_format(type="torch", columns=cols)

    return EncodedDataset(train=train_dataset, val=val_dataset, label_encoder=le)


def compute_metrics(eval_pred):
    """
    Eval helper for HuggingFace Trainer.
    """
    predictions = getattr(eval_pred, "predictions", None)
    labels = getattr(eval_pred, "label_ids", None)
    if predictions is None or labels is None:
        predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = predictions.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_weighted": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path",
        default=DEFAULT_CSV,
        help="Path to labeled CSV (default: data/processed/labeled_samples.csv)",
    )
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="HF model name or path",
    )
    parser.add_argument(
        "--gold-dev-path",
        default=GOLD_DEV_PATH,
        help="Optional gold dev CSV for extra eval",
    )
    args = parser.parse_args()

    os.makedirs(MODEL_OUT_DIR, exist_ok=True)

    model_name = args.model_name
    enc_data = build_datasets(csv_path=args.csv_path, model_name=model_name)
    train_dataset = enc_data.train
    val_dataset = enc_data.val
    le = enc_data.label_encoder

    num_labels = len(le.classes_)
    id2label: Dict[int, str] = {i: label for i, label in enumerate(le.classes_)}
    label2id: Dict[str, int] = {label: i for i, label in enumerate(le.classes_)}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    sig_params = TrainingArguments.__init__.__code__.co_varnames
    eval_key = (
        "evaluation_strategy"
        if "evaluation_strategy" in sig_params
        else ("eval_strategy" if "eval_strategy" in sig_params else None)
    )
    save_key = "save_strategy" if "save_strategy" in sig_params else None
    load_best_supported = "load_best_model_at_end" in sig_params
    metric_supported = "metric_for_best_model" in sig_params
    greater_supported = "greater_is_better" in sig_params

    base_kwargs = dict(
        output_dir=os.path.join(MODEL_OUT_DIR, "checkpoints"),
        num_train_epochs=6,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        save_total_limit=2,
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=42,
        warmup_ratio=0.06,
    )
    if eval_key:
        base_kwargs[eval_key] = "epoch"
    if save_key:
        base_kwargs[save_key] = "epoch"
    if (
        eval_key
        and save_key
        and load_best_supported
        and metric_supported
        and greater_supported
    ):
        base_kwargs["load_best_model_at_end"] = True
        base_kwargs["metric_for_best_model"] = "f1_weighted"
        base_kwargs["greater_is_better"] = True
    elif "evaluate_during_training" in sig_params:
        base_kwargs["evaluate_during_training"] = True

    # Label smoothing helps with noisy labels if supported
    if "label_smoothing_factor" in sig_params:
        base_kwargs["label_smoothing_factor"] = 0.1

    training_args = TrainingArguments(**base_kwargs)

    # Class weights to address imbalance/noise
    label_counts = np.bincount(train_dataset["labels"])
    class_weights = 1.0 / np.sqrt(label_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    class WeightedTrainer(Trainer):
        def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print("Eval results (noisy val):", eval_results)

    # Optional gold dev evaluation
    if os.path.exists(args.gold_dev_path):
        try:
            gold_df = load_labeled_data(args.gold_dev_path)
            gold_df = gold_df[gold_df["label"].isin(le.classes_)]
            if not gold_df.empty:
                gold_df = gold_df.copy()
                gold_df["label_id"] = le.transform(gold_df["label"])
                gold_ds = Dataset.from_pandas(
                    gold_df[["text", "label_id"]], preserve_index=False
                ).rename_column("label_id", "labels")
                gold_ds = gold_ds.map(
                    lambda batch: tokenizer(
                        batch["text"],
                        truncation=True,
                        max_length=192,
                    ),
                    batched=True,
                )
                gold_ds.set_format(
                    type="torch", columns=["input_ids", "attention_mask", "labels"]
                )
                gold_metrics = trainer.evaluate(gold_ds, metric_key_prefix="gold_dev")
                print("GOLD DEV METRICS:", gold_metrics)
            else:
                print(
                    "Gold dev file exists but has no labels seen in training; skipping gold eval."
                )
        except Exception as e:
            print(f"Gold dev evaluation skipped due to error: {e}")
    else:
        print("Gold dev file not found; skipping gold eval.")

    trainer.save_model(MODEL_OUT_DIR)
    tokenizer.save_pretrained(MODEL_OUT_DIR)

    labels_path = os.path.join(MODEL_OUT_DIR, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(
            {"id2label": id2label, "label2id": label2id, "classes": list(le.classes_)},
            f,
            indent=2,
        )
    print(f"Saved transformer model and label mapping to {MODEL_OUT_DIR}")


if __name__ == "__main__":
    print("PyTorch CUDA available:", torch.cuda.is_available())
    main()
