import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


VALID_LETTERS = {"A", "B", "C"}

DEFAULT_INPUT_ROOT = Path("experiments/outputs")
DEFAULT_OUTPUT_CSV = Path("experiments/evaluations/invalid_outputs.csv")



def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in: {path}")

    return data


def normalize_letter(value: Any) -> str:
    if value is None:
        return ""

    value = str(value).strip().upper()

    if value in VALID_LETTERS:
        return value

    return ""


def is_valid_prediction(record: dict[str, Any]) -> bool:
    parsed_answer = normalize_letter(record.get("parsed_answer"))
    return parsed_answer in VALID_LETTERS


def normalize_check_status(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip().upper()


def discover_result_files(input_root: Path) -> list[Path]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    result_files = sorted(input_root.rglob("*_results.json"))

    if not result_files:
        raise FileNotFoundError(f"No *_results.json files found under: {input_root}")

    return result_files


def infer_model_prompt_run(path: Path, input_root: Path) -> tuple[str, str, int]:
    relative_parts = path.relative_to(input_root).parts

    if len(relative_parts) < 4:
        raise ValueError(
            f"Unexpected result path structure: {path}. "
            "Expected: <model>/<prompt_type>/run_XX/<file>_results.json"
        )

    model = relative_parts[0]
    prompt_type = relative_parts[1]
    run_folder = relative_parts[2]

    match = re.fullmatch(r"run_(\d+)", run_folder)

    if not match:
        raise ValueError(
            f"Unexpected run folder name: {run_folder}. "
            "Expected format: run_XX"
        )

    run = int(match.group(1))

    return model, prompt_type, run


def get_raw_output(record: dict[str, Any]) -> Any:
    """
    Tries common field names used to store the raw model answer.
    Keeps the script robust even if different result files use different names.
    """
    for key in [
        "raw_response",
    ]:
        if key in record:
            return record.get(key)

    return None


def get_invalid_reason(record: dict[str, Any]) -> str:
    parsed_answer = record.get("parsed_answer")

    if parsed_answer is None:
        return "missing_parsed_answer"

    parsed_answer_str = str(parsed_answer).strip()

    if parsed_answer_str == "":
        return "empty_parsed_answer"

    if parsed_answer_str.upper() not in VALID_LETTERS:
        return "parsed_answer_not_A_B_C"

    return "unknown_invalid_reason"


def extract_invalid_outputs(
    input_root: Path,
    runs: list[int] | None = None,
    models: list[str] | None = None,
    prompt_types: list[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result_files = discover_result_files(input_root)

    rows = []
    full_invalid_records = []

    for path in result_files:
        model, prompt_type, run = infer_model_prompt_run(path, input_root)

        if runs is not None and run not in runs:
            continue

        if models is not None and model not in models:
            continue

        if prompt_types is not None and prompt_type not in prompt_types:
            continue

        records = load_json(path)

        for index, record in enumerate(records):
            if is_valid_prediction(record):
                continue

            invalid_reason = get_invalid_reason(record)

            row = {
                "model": model,
                "prompt_type": prompt_type,
                "run": run,
                "source_file": str(path),
                "record_index": index,

                "id": record.get("id"),
                "example_id": record.get("example_id"),
                "category": record.get("category"),
                "question_polarity": record.get("question_polarity"),

                "check_status": normalize_check_status(record.get("check_status")),
                "parsed_answer": record.get("parsed_answer"),
                "normalized_parsed_answer": normalize_letter(record.get("parsed_answer")),
                "invalid_reason": invalid_reason,

                "basic_response_format_valid": record.get("basic_response_format_valid"),
                "strict_response_format_valid": record.get("strict_response_format_valid"),

                "gold_letter": record.get("gold_letter"),
                "biased_letter": record.get("biased_letter"),
                "anti_biased_letter": record.get("anti_biased_letter"),

                "raw_output": get_raw_output(record),
            }

            rows.append(row)

            full_record = {
                "metadata": {
                    "model": model,
                    "prompt_type": prompt_type,
                    "run": run,
                    "source_file": str(path),
                    "record_index": index,
                    "invalid_reason": invalid_reason,
                },
                "record": record,
            }

            full_invalid_records.append(full_record)

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            by=["model", "prompt_type", "run", "record_index"],
            ascending=[True, True, True, True],
        )

    return df, full_invalid_records


def save_outputs(
    df: pd.DataFrame,
    full_records: list[dict[str, Any]],
    output_csv: Path,
    output_json: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved CSV: {output_csv}")
    print(f"[OK] Invalid outputs: {len(df)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export invalid model outputs from *_results.json files."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root folder containing model output result files.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="CSV file where invalid outputs will be saved.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=None,
        help="Optional run numbers to include, e.g. --runs 1 2 3.",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Optional model folders to include.",
    )

    parser.add_argument(
        "--prompt-types",
        type=str,
        nargs="+",
        default=None,
        help="Optional prompt types to include.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df, full_records = extract_invalid_outputs(
        input_root=args.input_root,
        runs=args.runs,
        models=args.models,
        prompt_types=args.prompt_types,
    )


if __name__ == "__main__":
    main()