import json
import re
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
import textstat
import spacy

# Load spaCy English model once
nlp = spacy.load("en_core_web_sm")


def load_records(path: Path):
    """Load records from a .json or .jsonl file."""
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v

        raise ValueError(f"Unsupported JSON structure in {path}")

    elif path.suffix.lower() == ".jsonl":
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    raise ValueError(f"Only .json and .jsonl are supported: {path}")


def safe_text(text) -> str:
    return "" if text is None else str(text)


def normalize_for_yule(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def yules_k(text: str) -> float:
    tokens = normalize_for_yule(text)
    if not tokens:
        return 0.0

    freq = Counter(tokens)
    freq_of_freq = Counter(freq.values())
    N = len(tokens)
    summation = sum((i ** 2) * v_i for i, v_i in freq_of_freq.items())
    k = 10000 * (summation - N) / (N ** 2)
    return round(k, 6)


def token_count_spacy(doc) -> int:
    return sum(1 for t in doc if not t.is_space)


def sentence_count_spacy(doc) -> int:
    return len(list(doc.sents))


def average_sentence_length(doc) -> float:
    sents = list(doc.sents)
    if not sents:
        return 0.0
    lengths = [sum(1 for t in sent if not t.is_space) for sent in sents]
    return round(sum(lengths) / len(lengths), 6)


def pos_counts(doc):
    noun_count = sum(1 for t in doc if t.pos_ in {"NOUN", "PROPN"})
    verb_count = sum(1 for t in doc if t.pos_ in {"VERB", "AUX"})
    adjective_count = sum(1 for t in doc if t.pos_ == "ADJ")
    return noun_count, verb_count, adjective_count


def extract_metrics(text: str) -> dict:
    text = safe_text(text)
    doc = nlp(text)

    token_count = token_count_spacy(doc)
    sentence_count = sentence_count_spacy(doc)
    avg_sentence_length = average_sentence_length(doc)

    try:
        flesch_reading_ease = round(textstat.flesch_reading_ease(text), 6)
    except Exception:
        flesch_reading_ease = 0.0

    try:
        gunning_fog_index = round(textstat.gunning_fog(text), 6)
    except Exception:
        gunning_fog_index = 0.0

    noun_count, verb_count, adjective_count = pos_counts(doc)

    return {
        "token_count": token_count,
        "sentence_count": sentence_count,
        "average_sentence_length": avg_sentence_length,
        "flesch_reading_ease": flesch_reading_ease,
        "gunning_fog_index": gunning_fog_index,
        "yules_k": yules_k(text),
        "noun_count": noun_count,
        "verb_count": verb_count,
        "adjective_count": adjective_count,
    }


def get_modified_prompt(rec: dict) -> str:
    if "rewritten_prompt" in rec:
        return safe_text(rec.get("rewritten_prompt", ""))

    if "prompt_text" in rec:
        return safe_text(rec.get("prompt_text", ""))

    raise ValueError(f"Cannot detect modified prompt field. Keys: {list(rec.keys())}")


def build_rows(records: list[dict]) -> list[dict]:
    rows = []

    for rec in records:
        modified_prompt = get_modified_prompt(rec)

        row = {
            "example_id": rec.get("example_id"),
            "category": rec.get("category"),
        }

        metrics = extract_metrics(modified_prompt)
        row.update({f"modified_{k}": v for k, v in metrics.items()})

        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compute prompt metrics on modified prompts for a single JSON/JSONL file."
    )
    parser.add_argument("--input-file", required=True, help="Input .json or .jsonl file.")
    parser.add_argument("--output-file", required=True, help="Output CSV file.")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = load_records(input_path)
    rows = build_rows(records)
    df = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[OK] {input_path} -> {output_path} | shape={df.shape}")


if __name__ == "__main__":
    main()