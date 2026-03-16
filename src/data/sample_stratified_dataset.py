import json
import random
from pathlib import Path
from collections import defaultdict


INPUT_PATH = Path("data/processed/bbq_disambiguated.json")
OUTPUT_PATH = Path("data/processed/bbq_disambiguated_sample_380.json")

TOTAL_SAMPLES = 380
RANDOM_SEED = 42


def load_examples(input_path: Path) -> list:
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_examples(examples: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)


def group_by_category(examples: list) -> dict:
    grouped = defaultdict(list)

    for example in examples:
        grouped[example["category"]].append(example)

    return grouped


def compute_balanced_allocation(grouped_examples: dict, total_samples: int) -> dict:
    """
    Compute a balanced allocation across categories.

    If total_samples = 380 and categories = 11,
    then most categories will receive 34 or 35 samples.
    """
    categories = sorted(grouped_examples.keys())
    num_categories = len(categories)

    base_n = total_samples // num_categories
    remainder = total_samples % num_categories

    allocation = {}

    for i, category in enumerate(categories):
        allocation[category] = base_n + (1 if i < remainder else 0)

    return allocation


def sample_stratified(grouped_examples: dict, allocation: dict, seed: int) -> list:
    """
    Randomly sample examples within each category without replacement.
    """
    rng = random.Random(seed)
    sampled = []

    for category, examples in grouped_examples.items():
        n_to_sample = allocation[category]

        if n_to_sample > len(examples):
            raise ValueError(
                f"Category '{category}' has only {len(examples)} examples, "
                f"but {n_to_sample} were requested."
            )

        sampled_examples = rng.sample(examples, n_to_sample)
        sampled.extend(sampled_examples)

    rng.shuffle(sampled)
    return sampled


def print_summary(grouped_examples: dict, allocation: dict, sampled_examples: list) -> None:
    print("=== Stratified Sampling Summary ===")
    print(f"Total categories: {len(grouped_examples)}")
    print(f"Total sampled examples: {len(sampled_examples)}")
    print()

    for category in sorted(grouped_examples.keys()):
        original_count = len(grouped_examples[category])
        sampled_count = allocation[category]
        print(f"{category}: sampled {sampled_count} out of {original_count}")


def main():
    print("Loading processed dataset...")
    examples = load_examples(INPUT_PATH)

    print("Grouping examples by category...")
    grouped_examples = group_by_category(examples)

    print("Computing balanced allocation...")
    allocation = compute_balanced_allocation(grouped_examples, TOTAL_SAMPLES)

    print("Sampling examples within each category...")
    sampled_examples = sample_stratified(grouped_examples, allocation, RANDOM_SEED)

    print("Saving sampled dataset...")
    save_examples(sampled_examples, OUTPUT_PATH)

    print_summary(grouped_examples, allocation, sampled_examples)
    print()
    print(f"Sampled dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()