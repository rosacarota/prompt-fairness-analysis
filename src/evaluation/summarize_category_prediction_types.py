import argparse
import json
import re
from pathlib import Path

import pandas as pd


PREDICTION_TYPES = ["target", "non_target", "unknown", "invalid", "other_valid"]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_rate(num, den):
    return num / den if den > 0 else None


def infer_model_prompt_run(path: Path, input_root: Path):
    """
    Expected structure:

    experiments/evaluations/<model>/<prompt_type>/analysis_ready/<file>_analysis_ready.json
    """

    relative_parts = path.relative_to(input_root).parts

    prompt_type = relative_parts[0]
    model = input_root.name

    filename = path.stem

    if filename.endswith("_analysis_ready"):
        filename = filename[:-len("_analysis_ready")]

    match = re.search(r"_results_(\d+)$", filename)

    if match:
        run = int(match.group(1))
    else:
        run = 0

    return model, prompt_type, run


def get_prediction_type(record: dict) -> str:
    if record.get("is_valid_prediction", 0) == 0:
        return "invalid"

    if record.get("is_unknown_prediction", 0) == 1:
        return "unknown"

    if record.get("is_target_prediction", 0) == 1:
        return "target"

    if record.get("is_nontarget_prediction", 0) == 1:
        return "non_target"

    return "other_valid"


def compute_distribution(records):
    total = len(records)

    counts = {prediction_type: 0 for prediction_type in PREDICTION_TYPES}

    for record in records:
        prediction_type = get_prediction_type(record)
        counts[prediction_type] += 1

    row = {
        "n_records": total,
    }

    for prediction_type in PREDICTION_TYPES:
        row[f"n_{prediction_type}"] = counts[prediction_type]
        row[f"{prediction_type}_rate"] = safe_rate(counts[prediction_type], total)

    return row


def build_by_category_run_table(input_root: Path, runs: list[int] | None = None):
    analysis_files = sorted(input_root.rglob("*_analysis_ready.json"))

    if not analysis_files:
        raise FileNotFoundError(f"No *_analysis_ready.json files found under: {input_root}")

    rows = []

    for path in analysis_files:
        records = load_json(path)

        model, prompt_type, run = infer_model_prompt_run(path, input_root)

        if runs is not None and run not in runs:
            continue

        records_by_category = {}

        for record in records:
            category = record.get("category", "unknown_category")

            if category not in records_by_category:
                records_by_category[category] = []

            records_by_category[category].append(record)

        for category, category_records in records_by_category.items():
            distribution = compute_distribution(category_records)

            row = {
                "model": model,
                "prompt_type": prompt_type,
                "run": run,
                "category": category,
            }

            row.update(distribution)
            rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No analysis-ready files matched the selected runs {runs} under: {input_root}"
        )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["model", "prompt_type", "run", "category"],
        ascending=[True, True, True, True]
    )

    return df


def build_category_summary_table(by_category_run_df: pd.DataFrame):
    id_cols = ["model", "prompt_type", "category"]
    excluded_cols = set(id_cols + ["run"])

    metric_cols = [
        col for col in by_category_run_df.columns
        if col not in excluded_cols
    ]

    rows = []

    for keys, group in by_category_run_df.groupby(id_cols, dropna=False):
        model, prompt_type, category = keys

        row = {
            "model": model,
            "prompt_type": prompt_type,
            "category": category,
            "n_runs": len(group),
        }

        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()

            if len(values) == 0:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                row[f"{metric}_min"] = None
                row[f"{metric}_max"] = None
            else:
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0
                row[f"{metric}_min"] = values.min()
                row[f"{metric}_max"] = values.max()

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    summary_df = summary_df.sort_values(
        by=["model", "prompt_type", "category"],
        ascending=[True, True, True]
    )

    return summary_df


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize prediction type distribution by BBQ category."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Model evaluation folder, e.g. experiments/evaluations/llama3.1_8b"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output folder for category-level prediction type distribution CSV files."
    )

    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=None,
        help="Runs to include, e.g. --runs 0 1 2"
    )

    args = parser.parse_args()

    by_category_run_df = build_by_category_run_table(args.input_root, args.runs)
    category_summary_df = build_category_summary_table(by_category_run_df)

    save_csv(
        by_category_run_df,
        args.output_dir / "prediction_type_distribution_by_category_run.csv"
    )

    save_csv(
        category_summary_df,
        args.output_dir / "prediction_type_distribution_by_category_summary.csv"
    )

    print()
    print("Done.")
    print(f"Selected runs: {args.runs if args.runs is not None else 'all'}")
    print(f"Rows in by-category-run table: {len(by_category_run_df)}")
    print(f"Rows in category summary table: {len(category_summary_df)}")


if __name__ == "__main__":
    main()