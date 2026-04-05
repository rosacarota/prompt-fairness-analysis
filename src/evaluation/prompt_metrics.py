import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import textstat
import spacy

# =========================
# CONFIG
# =========================
INPUT_FILE = "data/prompts/mutants/attribute_early_mutants.json"
OUTPUT_FILE = "experiments/evaluations/prompt_metrics/attribute_early_prompt_metrics.csv"

ORIGINAL_FIELD = "original_prompt"
REWRITTEN_FIELD = "rewritten_prompt"

META_FIELDS = [
    "example_id",
    "category",
    "source_prompt_type",
    "transformation_name",
    "transformation_target",
    "gold_label",
    "gold_answer",
    "rewriter_model"
]

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


def load_records(path: str):
    """Load records from a .json or .jsonl file."""
    p = Path(path)

    if p.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v

        raise ValueError("Unsupported JSON structure.")

    elif p.suffix.lower() == ".jsonl":
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    else:
        raise ValueError("Only .json and .jsonl files are supported.")


def safe_text(text) -> str:
    """Return an empty string for None, otherwise cast to string."""
    return "" if text is None else str(text)


def normalize_for_yule(text: str):
    """Normalize text for lexical diversity computation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def yules_k(text: str) -> float:
    """Compute Yule's K lexical diversity measure."""
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
    """Count non-space tokens."""
    return sum(1 for t in doc if not t.is_space)


def sentence_count_spacy(doc) -> int:
    """Count sentences using spaCy sentence boundaries."""
    return len(list(doc.sents))


def average_sentence_length(doc) -> float:
    """Compute average sentence length in tokens."""
    sents = list(doc.sents)

    if not sents:
        return 0.0

    lengths = [sum(1 for t in sent if not t.is_space) for sent in sents]
    return round(sum(lengths) / len(lengths), 6)


def pos_counts(doc):
    """Count nouns, verbs, and adjectives."""
    noun_count = sum(1 for t in doc if t.pos_ in {"NOUN", "PROPN"})
    verb_count = sum(1 for t in doc if t.pos_ in {"VERB", "AUX"})
    adjective_count = sum(1 for t in doc if t.pos_ == "ADJ")

    return noun_count, verb_count, adjective_count


def extract_metrics(text: str) -> dict:
    """Extract prompt metrics that do not require any manual lexicon."""
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


def prefix_metrics(metrics: dict, prefix: str) -> dict:
    """Prefix metric names to distinguish original vs rewritten prompts."""
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def main():
    records = load_records(INPUT_FILE)

    if not records:
        raise ValueError("No records found.")

    rows = []

    for rec in records:
        original_prompt = safe_text(rec.get(ORIGINAL_FIELD, ""))
        rewritten_prompt = safe_text(rec.get(REWRITTEN_FIELD, ""))

        row = {}

        for field in META_FIELDS:
            row[field] = rec.get(field, None)

        original_metrics = extract_metrics(original_prompt)
        rewritten_metrics = extract_metrics(rewritten_prompt)

        row.update(prefix_metrics(original_metrics, "original"))
        row.update(prefix_metrics(rewritten_metrics, "rewritten"))

        rows.append(row)

    df = pd.DataFrame(rows)

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved to: {output_path}")
    print(f"DataFrame shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()