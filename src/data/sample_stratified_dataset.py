import argparse
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from itertools import product
from typing import Any

DEFAULT_INPUT_PATH = Path("data/processed/bbq_disambiguated_annotated.json")
DEFAULT_OUTPUT_PATH = Path("data/processed/bbq_disambiguated_594_sample.json")

SAMPLES_PER_STRATUM = 3
RANDOM_SEED = 42


# BBQ categories present in the complete dataset.
BBQ_CATEGORIES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_SES",
    "Race_x_gender",
    "Religion",
    "SES",
    "Sexual_orientation",
]

QUESTION_POLARITIES = ["neg", "nonneg"]
LABELS = ["0", "1", "2"]

# Target is encoded as the answer option associated with the target/stereotyped group.
TARGETS = ["A", "B", "C"]

# Main stratification.
STRATA_FIELDS = ["category", "question_polarity", "label", "target"]

# Extra fields to include in the summary if present in the JSON.
REPORT_FIELDS = ["category", "question_polarity", "label", "target", "non_target", "unknown"]


def load_examples(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of examples.")

    return data


def save_examples(examples: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)


def normalize_value(value: Any) -> str:
    """
    Generic normalization.
    """
    if value is None:
        return ""

    return str(value).strip()


def normalize_field_value(field: str, value: Any) -> str:
    """
    Field-specific normalization.

    - label is normalized as string: 0, 1, 2
    - target/non_target/unknown are normalized as uppercase letters: A, B, C
    - all other fields are stripped strings
    """
    value_norm = normalize_value(value)

    if field == "label":
        return value_norm

    if field in {"target", "non_target", "unknown"}:
        return value_norm.upper()

    return value_norm


def get_stratum(example: dict[str, Any], strata_fields: list[str]) -> tuple[str, ...]:
    return tuple(
        normalize_field_value(field, example.get(field, ""))
        for field in strata_fields
    )


def expected_strata() -> list[tuple[str, str, str, str]]:
    return list(product(BBQ_CATEGORIES, QUESTION_POLARITIES, LABELS, TARGETS))


def group_by_strata(
    examples: list[dict[str, Any]],
    strata_fields: list[str],
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped = defaultdict(list)

    for example in examples:
        stratum = get_stratum(example, strata_fields)
        grouped[stratum].append(example)

    return dict(grouped)


def validate_required_fields(
    examples: list[dict[str, Any]],
    strata_fields: list[str],
) -> None:
    missing_fields = sorted(
        field
        for field in strata_fields
        if any(field not in example for example in examples)
    )

    if missing_fields:
        raise ValueError(
            f"Missing required fields in at least one example: {missing_fields}"
        )


def validate_expected_strata(
    grouped_examples: dict[tuple[str, ...], list[dict[str, Any]]],
    strata_fields: list[str],
) -> None:
    if strata_fields != ["category", "question_polarity", "label", "target"]:
        return

    expected = set(expected_strata())
    observed = set(grouped_examples.keys())

    missing_strata = sorted(expected - observed)
    unexpected_strata = sorted(observed - expected)

    if missing_strata:
        formatted = "\n".join(str(s) for s in missing_strata[:30])
        raise ValueError(
            f"Missing {len(missing_strata)} expected strata. "
            f"First missing strata:\n{formatted}"
        )

    if unexpected_strata:
        formatted = "\n".join(str(s) for s in unexpected_strata[:30])
        raise ValueError(
            f"Found {len(unexpected_strata)} unexpected strata. "
            f"This usually means invalid values in category/question_polarity/label/target. "
            f"First unexpected strata:\n{formatted}"
        )


def validate_dataset(
    examples: list[dict[str, Any]],
    grouped_examples: dict[tuple[str, ...], list[dict[str, Any]]],
    strata_fields: list[str],
) -> None:
    validate_required_fields(examples, strata_fields)
    validate_expected_strata(grouped_examples, strata_fields)


def sample_balanced_by_strata(
    grouped_examples: dict[tuple[str, ...], list[dict[str, Any]]],
    samples_per_stratum: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sampled = []

    for stratum in sorted(grouped_examples.keys()):
        examples = grouped_examples[stratum]

        if samples_per_stratum > len(examples):
            raise ValueError(
                f"Stratum {stratum} has only {len(examples)} examples, "
                f"but {samples_per_stratum} were requested."
            )

        sampled.extend(rng.sample(examples, samples_per_stratum))

    rng.shuffle(sampled)
    return sampled


def count_field(examples: list[dict[str, Any]], field: str) -> Counter:
    return Counter(
        normalize_field_value(field, example.get(field, ""))
        for example in examples
    )


def count_strata(
    examples: list[dict[str, Any]],
    strata_fields: list[str],
) -> Counter:
    return Counter(get_stratum(example, strata_fields) for example in examples)


def check_duplicate_example_keys(examples: list[dict[str, Any]]) -> None:
    keys = []

    for example in examples:
        example_id = example.get("example_id")
        category = example.get("category")

        if example_id is not None and category is not None:
            keys.append((example_id, category))

    duplicated = [
        key
        for key, count in Counter(keys).items()
        if count > 1
    ]

    if duplicated:
        preview = ", ".join(
            f"(example_id={example_id}, category={category})"
            for example_id, category in duplicated[:10]
        )

        raise ValueError(
            f"Duplicated example_id + category keys in sample: {preview}"
        )


def print_counter(title: str, counter: Counter) -> None:
    print(title)

    for key, value in sorted(counter.items(), key=lambda item: str(item[0])):
        print(f"  {key}: {value}")

    print()


def print_original_strata_availability(
    grouped_examples: dict[tuple[str, ...], list[dict[str, Any]]],
) -> None:
    sizes = Counter(len(examples) for examples in grouped_examples.values())

    print("Original strata availability:")
    for stratum_size, how_many_strata in sorted(sizes.items()):
        print(f"  {how_many_strata} strata have {stratum_size} examples")

    print()


def print_summary(
    original_examples: list[dict[str, Any]],
    grouped_examples: dict[tuple[str, ...], list[dict[str, Any]]],
    sampled_examples: list[dict[str, Any]],
    strata_fields: list[str],
    samples_per_stratum: int,
) -> None:
    print("=== Balanced Sampling Summary ===")
    print(f"Strata fields: {strata_fields}")
    print(f"Total original examples: {len(original_examples)}")
    print(f"Total strata: {len(grouped_examples)}")
    print(f"Samples per stratum: {samples_per_stratum}")
    print(f"Total sampled examples: {len(sampled_examples)}")
    print()

    print_original_strata_availability(grouped_examples)

    for field in REPORT_FIELDS:
        if any(field in example for example in sampled_examples):
            print_counter(
                f"Sampled distribution by {field}:",
                count_field(sampled_examples, field),
            )

    print("Sampled distribution by full stratum:")
    stratum_counts = count_strata(sampled_examples, strata_fields)

    for stratum, count in sorted(stratum_counts.items()):
        print(f"  {stratum}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Balanced stratified sampling for BBQ disambiguated annotated data. "
            "Hard-coded stratification: category x question_polarity x label x target. "
            "Hard-coded sample size: 594 examples, 3 per stratum."
        )
    )

    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input full BBQ disambiguated annotated JSON file.",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output sampled JSON file.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    print("Loading processed annotated dataset...")
    examples = load_examples(args.input_json)

    print("Grouping examples by selected strata...")
    grouped_examples = group_by_strata(examples, STRATA_FIELDS)

    print("Validating dataset and expected strata...")
    validate_dataset(examples, grouped_examples, STRATA_FIELDS)

    samples_per_stratum = SAMPLES_PER_STRATUM
    expected_total = len(grouped_examples) * samples_per_stratum

    print("Sampling configuration:")
    print(f"  Number of strata: {len(grouped_examples)}")
    print(f"  Samples per stratum: {samples_per_stratum}")
    print(f"  Expected total sample size: {expected_total}")
    print()

    print("Sampling examples within each stratum...")
    sampled_examples = sample_balanced_by_strata(
        grouped_examples=grouped_examples,
        samples_per_stratum=samples_per_stratum,
        seed=RANDOM_SEED,
    )

    check_duplicate_example_keys(sampled_examples)

    print("Saving sampled dataset...")
    save_examples(sampled_examples, args.output_json)

    print_summary(
        original_examples=examples,
        grouped_examples=grouped_examples,
        sampled_examples=sampled_examples,
        strata_fields=STRATA_FIELDS,
        samples_per_stratum=samples_per_stratum,
    )

    print(f"Sampled dataset saved to: {args.output_json}")


if __name__ == "__main__":
    main()