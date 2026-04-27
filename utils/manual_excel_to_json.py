import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-json", type=Path, required=True)
    parser.add_argument("--annotation-xlsx", type=Path, required=True)
    parser.add_argument(
    "--output-json",
    type=Path,
    default=Path("data/processed/bbq_disambiguated_sample_380_annotated.json")
    )
    args = parser.parse_args()

    records = load_json(args.original_json)
    df = pd.read_excel(args.annotation_xlsx)

    annotations = {}

    for _, row in df.iterrows():
        key = (int(row["example_id"]), str(row["category"]))

        annotations[key] = {
            "unknown_letter": clean(row["unknown"]),
            "target_letter": clean(row["target"]),
            "non_target_letter": clean(row["non_target"]),
        }

    output = []

    for record in records:
        key = (int(record["example_id"]), str(record["category"]))

        new_record = dict(record)
        new_record.update(annotations[key])

        output.append(new_record)

    save_json(output, args.output_json)

    print(f"[OK] Saved: {args.output_json}")


if __name__ == "__main__":
    main()