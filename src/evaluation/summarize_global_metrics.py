# Summarize global output metrics across runs

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


INPUT_FILE = Path("experiments/evaluations/output_metrics_by_run.csv")
OUTPUT_FILE = Path("experiments/evaluations/global_metrics_summary.csv")

ID_COLUMNS = {
    "model",
    "prompt_type",
    "run",
}


def load_metrics_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        skipinitialspace=True,
    )

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    required_columns = {"model", "prompt_type", "run"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print("Columns found in input CSV:")
        print(list(df.columns))

        raise ValueError(
            f"Missing required columns in {path}: {sorted(missing_columns)}"
        )

    return df


def get_metric_columns(df: pd.DataFrame) -> list[str]:
    metric_columns = []

    for column in df.columns:
        if column in ID_COLUMNS:
            continue

        numeric_values = pd.to_numeric(df[column], errors="coerce")

        if numeric_values.notna().any():
            metric_columns.append(column)

    return metric_columns


def filter_dataframe(
    df: pd.DataFrame,
    runs: list[int] | None = None,
    models: list[str] | None = None,
    prompt_types: list[str] | None = None,
) -> pd.DataFrame:
    filtered_df = df.copy()

    if runs is not None:
        filtered_df = filtered_df[filtered_df["run"].isin(runs)]

    if models is not None:
        filtered_df = filtered_df[filtered_df["model"].isin(models)]

    if prompt_types is not None:
        filtered_df = filtered_df[filtered_df["prompt_type"].isin(prompt_types)]

    if filtered_df.empty:
        raise ValueError("No rows matched the selected filters.")

    return filtered_df


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["model", "prompt_type"]
    metric_columns = get_metric_columns(df)

    summary_rows: list[dict[str, Any]] = []

    for (model, prompt_type), group in df.groupby(group_columns, dropna=False):
        row: dict[str, Any] = {
            "model": model,
            "prompt_type": prompt_type,
            "runs_total": len(group),
        }

        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()

            if len(values) == 0:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
                row[f"{metric}_min"] = None
                row[f"{metric}_max"] = None
            else:
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
                row[f"{metric}_min"] = values.min()
                row[f"{metric}_max"] = values.max()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        by=["model", "prompt_type"],
        ascending=[True, True],
    )

    return summary_df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize global output metrics across runs."
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
        help="Optional model names to include, e.g. --models llama31_8b qwen3_8b.",
    )

    parser.add_argument(
        "--prompt-types",
        type=str,
        nargs="+",
        default=None,
        help="Optional prompt types to include, e.g. --prompt-types baseline role_based.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    by_run_df = load_metrics_csv(INPUT_FILE)

    filtered_df = filter_dataframe(
        df=by_run_df,
        runs=args.runs,
        models=args.models,
        prompt_types=args.prompt_types,
    )

    summary_df = build_summary_table(filtered_df)

    save_csv(summary_df, OUTPUT_FILE)

    print()
    print("=== Global Metrics Summary ===")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Rows in input table: {len(by_run_df)}")
    print(f"Rows after filters: {len(filtered_df)}")
    print(f"Rows in summary table: {len(summary_df)}")
    print(f"Models: {sorted(filtered_df['model'].unique())}")
    print(f"Prompt types: {sorted(filtered_df['prompt_type'].unique())}")
    print(f"Runs: {sorted(filtered_df['run'].unique())}")


if __name__ == "__main__":
    main()