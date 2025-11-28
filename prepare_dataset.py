"""
CIS-483 Term Project
Noor Mahmoud
Build raw_samples.csv from Books/ and Journals/ text/markdown/jsons.
"""

import csv
import json
import os
import re
from typing import Dict, Iterable, List

DATA_DIR = os.path.join("data")
BOOKS_DIR = os.path.join(DATA_DIR, "Books")
JOURNALS_DIR = os.path.join(DATA_DIR, "Journals")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "raw_samples.csv")

MAX_WORDS = 220
MIN_WORDS = 40


def read_text_file(path: str) -> str:
    """
    Read .txt or .md as plain text. Uses utf-8 with replacement to avoid
    crashing on stray bytes (Windows-safe).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def try_parse_json(path: str):
    """
    Try to parse chat logs. Supports two common shapes:
    - A regular JSON list/dict
    - JSON Lines (one JSON object per line)
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read().strip()

    if not raw:
        return []

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # JSON lines fallback
        messages = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return messages


def flatten_messages(parsed_json) -> List[Dict]:
    """
    Extract a flat list of message dicts from whatever structure we loaded.
    Tries to be forgiving so you can tweak fields later if needed.
    """
    if isinstance(parsed_json, list):
        return [m for m in parsed_json if isinstance(m, dict)]

    if isinstance(parsed_json, dict):
        for key in ("messages", "entries", "logs", "data"):
            if key in parsed_json and isinstance(parsed_json[key], list):
                return [m for m in parsed_json[key] if isinstance(m, dict)]

    return []


def message_to_line(msg: Dict) -> str:
    """
    Turn a chat message dict into a single readable line:
    "Speaker: message"
    Fields are intentionally loose to handle common export formats.
    """
    speaker = (
        msg.get("sender")
        or msg.get("author")
        or msg.get("speaker")
        or msg.get("from")
        or msg.get("role")
        or "Unknown"
    )
    content = (
        msg.get("body")
        or msg.get("content")
        or msg.get("text")
        or msg.get("message")
        or ""
    )

    # Collapse whitespace to avoid giant gaps
    speaker = str(speaker).strip() or "Unknown"
    content = re.sub(r"\s+", " ", str(content).strip())
    return f"{speaker}: {content}".strip()


def read_json_chat(path: str) -> str:
    parsed = try_parse_json(path)
    messages = flatten_messages(parsed)
    if not messages:
        return ""

    lines: List[str] = []
    for msg in messages:
        line = message_to_line(msg)
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


def split_long_paragraph(para: str, max_words: int) -> List[str]:
    """
    Breaks a single paragraph that is way over max_words into smaller pieces so
    we don't end up with one giant chunk.
    """
    words = para.split()
    if len(words) <= max_words:
        return [para]

    parts = []
    for i in range(0, len(words), max_words):
        part_words = words[i : i + max_words]
        parts.append(" ".join(part_words))
    return parts


def split_into_chunks(
    text: str, max_words: int = MAX_WORDS, min_words: int = MIN_WORDS
) -> List[str]:
    """
    Simple splitter:
    - Split by blank lines into paragraphs
    - Break oversized paragraphs
    - Merge until ~max_words
    - Drop chunks that are too short (< min_words)
    """
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs: List[str] = []
    for para in raw_paragraphs:
        cleaned = para.strip()
        if not cleaned:
            continue
        paragraphs.extend(split_long_paragraph(cleaned, max_words))

    chunks: List[str] = []
    current: List[str] = []
    current_count = 0

    for para in paragraphs:
        words = para.split()
        if not words:
            continue

        if current_count + len(words) <= max_words:
            current.append(para)
            current_count += len(words)
        else:
            if current_count >= min_words:
                chunks.append("\n\n".join(current))
            current = [para]
            current_count = len(words)

    if current and current_count >= min_words:
        chunks.append("\n\n".join(current))

    return chunks


def iter_files(root: str, exts: Iterable[str]) -> Iterable[str]:
    """Yield files under root that match extensions (case-insensitive)."""
    if not os.path.isdir(root):
        return
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in exts:
                yield os.path.join(dirpath, fname)


def gather_samples() -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []

    def add_samples_from_file(file_path: str, source_type: str):
        ext = os.path.splitext(file_path)[1].lower()
        rel_name = os.path.relpath(file_path, DATA_DIR)

        if ext in {".txt", ".md"}:
            text = read_text_file(file_path)
        elif ext == ".json":
            text = read_json_chat(file_path)
        else:
            return

        if not text.strip():
            return

        for i, chunk in enumerate(split_into_chunks(text)):
            samples.append(
                {
                    "id": f"{source_type}_{rel_name}_{i}",
                    "source_type": source_type,
                    "file_name": rel_name,
                    "chunk_index": str(i),
                    "text": chunk,
                    "label": "",
                }
            )

    for path in iter_files(BOOKS_DIR, {".txt", ".md"}):
        add_samples_from_file(path, "book")

    for path in iter_files(JOURNALS_DIR, {".txt", ".md", ".json"}):
        add_samples_from_file(path, "journal")

    return samples


def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    samples = gather_samples()
    if not samples:
        print("No samples found. Check your Books/ and Journals/ folders.")
        return

    fieldnames = ["id", "source_type", "file_name", "chunk_index", "text", "label"]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)

    print(f"Wrote {len(samples)} samples to {OUTPUT_CSV}")
    print(
        "Open this CSV, add labels, and save as labeled_samples.csv in the same folder."
    )


if __name__ == "__main__":
    main()
