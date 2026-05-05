import argparse
import json
import re
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "n_records_total",
    "n_valid_predictions",
    "n_invalid_predictions",
    "n_unknown_predictions",
    "n_target_predictions",
    "n_nontarget_predictions",

    "n_manual_review_records",
    "n_fairness_ready_records",
    "n_aligned_examples",
    "n_nonaligned_examples",
    "n_fairness_prediction_records",
    "n_biased_predictions",
    "n_anti_biased_predictions",

    # New nonaligned biased-error metrics
    "n_nonaligned_fairness_prediction_records",
    "n_biased_errors",
    "n_unbiased_success_nonaligned",

    "accuracy_dis",
    "accuracy_valid_only",
    "accuracy_fairness_ready",

    "valid_rate",
    "invalid_rate",
    "unknown_rate",
    "unknown_rate_total",
    "unknown_rate_valid_only",
    "target_prediction_rate_total",
    "nontarget_prediction_rate_total",
    "manual_review_rate",

    "accuracy_aligned",
    "accuracy_nonaligned",
    "accuracy_cost_bias_nonalignment",

    "biased_answer_rate",
    "anti_biased_answer_rate",

    # New nonaligned biased-error rates
    "biased_error_rate_nonaligned",
    "unbiased_success_rate_nonaligned",

    "sdis",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_model_prompt_run(path: Path, input_root: Path):
    relative_parts = path.relative_to(input_root).parts

    prompt_type = relative_parts[0]
    model = input_root.name

    filename = path.stem

    if filename.endswith("_metrics"):
        filename = filename[:-len("_metrics")]

    # Detect run from names ending in "_results_1", "_results_2", etc.
    # Files ending only in "_results" are treated as run 0.
    match = re.search(r"_results_(\d+)$", filename)

    if match:
        run = int(match.group(1))
    else:
        run = 0

    return model, prompt_type, run


def build_by_run_table(input_root: Path, runs: list[int] | None = None):
    metric_files = sorted(input_root.rglob("*_metrics.json"))

    if not metric_files:
        raise FileNotFoundError(f"No *_metrics.json files found under: {input_root}")

    rows = []

    for path in metric_files:
        metrics = load_json(path)

        model, prompt_type, run = infer_model_prompt_run(path, input_root)

        # Keep only selected runs if --runs is provided.
        if runs is not None and run not in runs:
            continue

        row = {
            "model": model,
            "prompt_type": prompt_type,
            "run": run,
        }

        for col in METRIC_COLUMNS:
            row[col] = metrics.get(col)

        # Backward compatibility with older metric files.
        if row.get("unknown_rate_total") is None and row.get("unknown_rate") is not None:
            row["unknown_rate_total"] = row["unknown_rate"]

        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No metric files matched the selected runs {runs} under: {input_root}"
        )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["model", "prompt_type", "run"],
        ascending=[True, True, True]
    )

    return df


def build_summary_table(by_run_df: pd.DataFrame):
    id_cols = ["model", "prompt_type"]
    excluded_cols = set(id_cols + ["run", "source_file"])

    metric_cols = [
        col for col in by_run_df.columns
        if col not in excluded_cols
    ]

    summary_rows = []

    for (model, prompt_type), group in by_run_df.groupby(id_cols, dropna=False):
        row = {
            "model": model,
            "prompt_type": prompt_type,
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

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        by=["model", "prompt_type"],
        ascending=[True, True]
    )

    return summary_df


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize global output metrics across prompt types and runs."
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
        help="Output folder for summary CSV files."
    )

    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=None,
        help="Runs to include, e.g. --runs 0 1 2"
    )

    args = parser.parse_args()

    by_run_df = build_by_run_table(args.input_root, args.runs)
    summary_df = build_summary_table(by_run_df)

    save_csv(
        by_run_df,
        args.output_dir / "global_metrics_by_run.csv"
    )

    save_csv(
        summary_df,
        args.output_dir / "global_metrics_summary.csv"
    )

    print()
    print("Done.")
    print(f"Selected runs: {args.runs if args.runs is not None else 'all'}")
    print(f"Rows in by-run table: {len(by_run_df)}")
    print(f"Rows in summary table: {len(summary_df)}")


if __name__ == "__main__":
    main()