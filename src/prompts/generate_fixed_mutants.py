import argparse
import json
from pathlib import Path

from prompt_builder import load_examples, build_prompt_by_type


INPUT_PATH = Path("data/processed/bbq_disambiguated_594_sample.json")
OUTPUT_DIR = Path("data/prompts/fixed")

FIXED_PROMPT_TYPES = [
    "role_based",
    "chain_of_thought",
]


def save_prompts(records: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def get_gold_label(example: dict) -> int:
    """
    Supports both:
    - processed dataset records, where the field is 'label'
    - prompt records, where the field is 'gold_label'
    """
    if "label" in example:
        return example["label"]

    if "gold_label" in example:
        return example["gold_label"]

    raise ValueError(
        f"Missing gold label for example_id={example.get('example_id')}. "
        "Expected either 'label' or 'gold_label'."
    )


def build_prompt_record(example: dict, prompt_type: str) -> dict:
    prompt_text = build_prompt_by_type(example, prompt_type)

    return {
        "example_id": example["example_id"],
        "category": example["category"],
        "question_polarity": example["question_polarity"],

        "prompt_type": prompt_type,
        "transformation_name": prompt_type,
        "prompt_text": prompt_text,

        "context": example["context"],
        "question": example["question"],
        "answers": example["answers"],

        "gold_label": get_gold_label(example),
        "gold_answer": example["gold_answer"],
        "answer_info": example.get("answer_info", {}),
        "stereotyped_groups": example.get("stereotyped_groups", []),

        "target": example.get("target", ""),
        "non_target": example.get("non_target", ""),
        "unknown": example.get("unknown", ""),
    }


def build_records_for_prompt_type(examples: list, prompt_type: str) -> list:
    records = []

    for example in examples:
        records.append(build_prompt_record(example, prompt_type))

    return records


def get_output_path(output_dir: Path, prompt_type: str) -> Path:
    return output_dir / prompt_type / f"{prompt_type}_prompts.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed prompt variants with the same JSON structure as baseline."
    )

    parser.add_argument(
        "--input-json",
        type=Path,
        default=INPUT_PATH,
        help="Input sampled BBQ JSON file.",
    )

    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=FIXED_PROMPT_TYPES + ["all"],
        default="all",
        help="Fixed prompt type to generate.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Base output directory for fixed prompts.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    examples = load_examples(str(args.input_json))

    if args.prompt_type == "all":
        prompt_types = FIXED_PROMPT_TYPES
    else:
        prompt_types = [args.prompt_type]

    for prompt_type in prompt_types:
        records = build_records_for_prompt_type(examples, prompt_type)
        output_path = get_output_path(args.output_dir, prompt_type)

        save_prompts(records, output_path)

        print(f"Saved {len(records)} {prompt_type} prompts to: {output_path}")


if __name__ == "__main__":
    main()