import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_JSON = Path("data/processed/bbq_disambiguated_annotated.json")


def save_json(data: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_text(value: Any) -> str:
    """
    Clean generic textual values from Excel.

    Keeps the original casing, unlike option letters.
    """
    if pd.isna(value):
        return ""

    return str(value).strip()


def clean_option(value: Any) -> str:
    """
    Clean option letters such as A, B, C.

    Used for unknown, target, and non_target.
    """
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def clean_int(value: Any) -> int:
    """
    Convert Excel numeric values to int.

    Excel sometimes reads integer columns as floats, e.g. 2.0.
    """
    if pd.isna(value):
        raise ValueError("Missing integer value.")

    return int(value)


def parse_stereotyped_groups(value: Any) -> list[str]:
    """
    Convert the Excel stereotyped_groups cell back to a list.

    The Excel export joined lists as:
        group1; group2

    This function restores:
        ["group1", "group2"]
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return [item.strip() for item in text.split(";") if item.strip()]


def excel_to_json(annotation_xlsx: Path, output_json: Path) -> None:
    """
    Transform the annotated Excel file back to the original BBQ JSON structure.

    The output keeps the original fields:

    - example_id
    - category
    - question_polarity
    - context
    - question
    - answers
    - label
    - gold_answer
    - answer_info
    - stereotyped_groups

    And adds the annotation fields:

    - unknown
    - target
    - non_target
    """
    df = pd.read_excel(annotation_xlsx)

    records = []

    for _, row in df.iterrows():
        record = {
            "example_id": clean_int(row["example_id"]),
            "category": clean_text(row["category"]),
            "question_polarity": clean_text(row["question_polarity"]),
            "context": clean_text(row["context"]),
            "question": clean_text(row["question"]),
            "answers": {
                "ans0": clean_text(row["A_answer"]),
                "ans1": clean_text(row["B_answer"]),
                "ans2": clean_text(row["C_answer"]),
            },
            "label": clean_int(row["label"]),
            "gold_answer": clean_text(row["gold_answer"]),
            "answer_info": {
                "ans0": [
                    clean_text(row["A_info_0"]),
                    clean_text(row["A_info_1"]),
                ],
                "ans1": [
                    clean_text(row["B_info_0"]),
                    clean_text(row["B_info_1"]),
                ],
                "ans2": [
                    clean_text(row["C_info_0"]),
                    clean_text(row["C_info_1"]),
                ],
            },
            "stereotyped_groups": parse_stereotyped_groups(row["stereotyped_groups"]),

            # Added annotation fields.
            # These are intentionally named without "_letter",
            # because the sampling script expects: target, non_target, unknown.
            "unknown": clean_option(row["unknown"]),
            "target": clean_option(row["target"]),
            "non_target": clean_option(row["non_target"]),
        }

        records.append(record)

    save_json(records, output_json)

    print(f"[OK] Converted Excel to JSON: {output_json}")
    print(f"Total records: {len(records)}")

    missing_target = sum(1 for r in records if not r["target"])
    missing_non_target = sum(1 for r in records if not r["non_target"])
    missing_unknown = sum(1 for r in records if not r["unknown"])

    print(f"Missing target values: {missing_target}")
    print(f"Missing non_target values: {missing_non_target}")
    print(f"Missing unknown values: {missing_unknown}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert manually annotated BBQ Excel file to JSON format."
    )

    parser.add_argument(
        "--annotation-xlsx",
        type=Path,
        required=True,
        help="Input Excel file with target/non_target annotations.",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON file.",
    )

    args = parser.parse_args()

    excel_to_json(args.annotation_xlsx, args.output_json)


if __name__ == "__main__":
    main()