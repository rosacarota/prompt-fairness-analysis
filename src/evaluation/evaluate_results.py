import json
import argparse
from pathlib import Path


def load_results(input_path: str) -> list:
    """
    Load inference results from a JSON file.
    """
    path = Path(input_path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_basic_metrics(results: list) -> dict:
    """
    Compute basic evaluation metrics from inference results.
    """
    total_examples = len(results)
    correct_predictions = sum(result.get("is_correct", 0) for result in results)
    invalid_predictions = sum(
        1 for result in results if result.get("parsed_answer") == "INVALID"
    )

    accuracy = correct_predictions / total_examples if total_examples > 0 else 0.0
    invalid_rate = invalid_predictions / total_examples if total_examples > 0 else 0.0

    return {
        "total_examples": total_examples,
        "correct_predictions": correct_predictions,
        "invalid_predictions": invalid_predictions,
        "accuracy": accuracy,
        "invalid_rate": invalid_rate,
    }


def print_metrics(metrics: dict):
    """
    Print evaluation metrics in a readable format.
    """
    print("=== Evaluation Results ===")
    print(f"Total examples: {metrics['total_examples']}")
    print(f"Correct predictions: {metrics['correct_predictions']}")
    print(f"Invalid predictions: {metrics['invalid_predictions']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Invalid rate: {metrics['invalid_rate']:.4f}")


def save_metrics(metrics: dict, output_path: str):
    """
    Save evaluation metrics to a JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate inference results.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the inference results JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save evaluation metrics as JSON"
    )

    args = parser.parse_args()

    results = load_results(args.input)
    metrics = compute_basic_metrics(results)

    print_metrics(metrics)

    if args.output:
        save_metrics(metrics, args.output)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()