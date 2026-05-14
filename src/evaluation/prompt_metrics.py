import argparse
import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import textstat
import spacy


nlp = spacy.load("en_core_web_sm")


PROMPT_FILE_NAMES = {
    "baseline_prompts.json",
    "role_based_prompts.json",
    "chain_of_thought_prompts.json",
    "attribute_early_prompts.json",
    "attribute_late_prompts.json",
}


def load_records(path: Path) -> list[dict]:
    """Load records from a .json or .jsonl file."""
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list):
                    return value

        raise ValueError(f"Unsupported JSON structure in {path}")

    if path.suffix.lower() == ".jsonl":
        records = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if line:
                    records.append(json.loads(line))

        return records

    raise ValueError(f"Only .json and .jsonl are supported: {path}")


def discover_input_files(input_file: Path | None, input_dir: Path | None) -> list[Path]:
    if input_file is not None:
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        return [input_file]

    if input_dir is None:
        raise ValueError("Either --input-file or --input-dir must be provided.")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files = []

    for path in input_dir.rglob("*.json"):
        if path.name in PROMPT_FILE_NAMES:
            files.append(path)

    for path in input_dir.rglob("*.jsonl"):
        files.append(path)

    files = sorted(files)

    if not files:
        raise FileNotFoundError(f"No prompt files found in: {input_dir}")

    return files


def safe_text(text) -> str:
    return "" if text is None else str(text)


def normalize_for_yule(text: str) -> list[str]:
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
    token_total = len(tokens)

    summation = sum((frequency ** 2) * count for frequency, count in freq_of_freq.items())
    k = 10000 * (summation - token_total) / (token_total ** 2)

    return round(k, 6)


def token_count_spacy(doc) -> int:
    return sum(1 for token in doc if not token.is_space)


def sentence_count_spacy(doc) -> int:
    return len(list(doc.sents))


def average_sentence_length(doc) -> float:
    sentences = list(doc.sents)

    if not sentences:
        return 0.0

    lengths = [
        sum(1 for token in sentence if not token.is_space)
        for sentence in sentences
    ]

    return round(sum(lengths) / len(lengths), 6)


def pos_counts(doc) -> tuple[int, int, int]:
    noun_count = sum(1 for token in doc if token.pos_ in {"NOUN", "PROPN"})
    verb_count = sum(1 for token in doc if token.pos_ in {"VERB", "AUX"})
    adjective_count = sum(1 for token in doc if token.pos_ == "ADJ")

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


def get_prompt_text(record: dict) -> str:
    prompt_text = record.get("prompt_text")

    if isinstance(prompt_text, str) and prompt_text.strip():
        return prompt_text.strip()

    raise ValueError(
        f"Missing prompt_text for "
        f"example_id={record.get('example_id')}, "
        f"category={record.get('category')}, "
        f"prompt_type={record.get('prompt_type')}"
    )


def build_rows_for_file(path: Path) -> list[dict]:
    records = load_records(path)
    rows = []

    for record in records:
        prompt_text = get_prompt_text(record)
        metrics = extract_metrics(prompt_text)

        row = {
            "source_file": path.name,
            "example_id": record.get("example_id"),
            "category": record.get("category"),
            "question_polarity": record.get("question_polarity"),
            "prompt_type": record.get("prompt_type"),
            "transformation_name": record.get("transformation_name"),
        }

        row.update({f"prompt_{key}": value for key, value in metrics.items()})

        rows.append(row)

    print(f"[OK] Loaded {len(rows)} records from {path}")

    return rows


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in df.columns
        if column.startswith("prompt_")
    ]

    group_columns = [
        "prompt_type",
        "transformation_name",
    ]

    summary = (
        df
        .groupby(group_columns, dropna=False)[metric_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute linguistic/readability metrics for prompt_text fields."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Single input .json or .jsonl prompt file.",
    )

    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing prompt files.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output CSV file with one row per prompt.",
    )

    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional output CSV file with aggregate metrics by prompt type.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_files = discover_input_files(
        input_file=args.input_file,
        input_dir=args.input_dir,
    )

    all_rows = []

    for input_file in input_files:
        all_rows.extend(build_rows_for_file(input_file))

    df = pd.DataFrame(all_rows)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_file, index=False, encoding="utf-8")

    print(f"[DONE] Metrics saved to {args.output_file} | shape={df.shape}")

    if args.summary_file is not None:
        summary_df = build_summary(df)

        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.summary_file, index=False, encoding="utf-8")

        print(f"[DONE] Summary saved to {args.summary_file} | shape={summary_df.shape}")


if __name__ == "__main__":
    main()