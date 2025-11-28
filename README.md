# CIS-483 Term Project

Repo map:
Repo map:
- `prepare_dataset.py`:
  ```
  Build raw_samples.csv from Books/ and Journals/ text/markdown/jsons.
  ```
- `auto_label_zero_shot.py`:
  ```
  Zero-shot labeler to tag raw_samples.csv with draft labels.
  ```
- `make_labeled_samples.py`:
  ```
  Turn auto_labeled.csv into labeled_samples.csv for training.
  ```
- `filter_and_rebalance.py`:
  ```
  Filter noisy auto_labeled.csv and write a cleaner labeled_samples_filtered.csv.
  ```
- `simplify_labels.py`:
  ```
  Map labels to a smaller set of core labels and save a simplified CSV.
  ```
- `create_gold_dev.py`:
  ```
  Sample a candidate gold dev set from labeled_samples.csv.
  ```
- `auto_clean_gold_dev.py`:
  ```
  Auto-clean gold_dev_candidates.csv using the label simplifier.
  ```
- `train_baseline.py`:
  ```
  TF-IDF + Logistic Regression baseline.
  ```
- `train_transformer.py`:
  ```
  Fine-tune a transformer classifier on labeled data.
  ```
- `analyze_labels.py`:
  ```
  Quick label exploration for labeled_samples.csv (counts + optional baseline eval).
  ```
- `predict_and_recommend.py`:
  ```
  Interactive script to test the classifier and retrieve similar snippets.
  ```
- `coping_assistant.py`:
  ```
  Interactive coping helper: predict label and show snippets from books.
  ```
