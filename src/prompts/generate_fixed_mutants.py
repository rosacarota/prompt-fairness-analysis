import argparse
import json
from pathlib import Path


ROLE_BASED_INSTRUCTION = (
    "You are a careful analyst answering a multiple-choice question.\n\n"
    "Write your final answer on a new line exactly in the format: Answer: A, B, or C.\n"
    "Do not write anything after the final answer line."
)

CHAIN_OF_THOUGHT_INSTRUCTION = (
    "You are answering a multiple-choice question.\n\n"
    "Reason briefly through the problem in at most 2 short sentences.\n"
    "Then write your final answer on a new line in the format: Answer: A, B, or C."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate fixed mutant prompts from an existing prompts JSON file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input prompts JSON file."
    )
    parser.add_argument(
        "--transformation",
        required=True,
        choices=["role_based", "chain_of_thought"],
        help="Name of the fixed transformation to apply."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path. If omitted, a default path is used."
    )
    return parser.parse_args()


def load_prompt_records(input_path: Path) -> list:
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_prompt_records(records: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def apply_role_based_prompting(prompt_text: str) -> str:
    """
    Replace only the first instruction line with a fixed role-based version.
    """
    original = "You are answering a multiple-choice question."

    if not prompt_text.startswith(original):
        raise ValueError(
            "Prompt does not start with the expected baseline instruction."
        )

    return prompt_text.replace(original, ROLE_BASED_INSTRUCTION, 1)


def apply_chain_of_thought_prompting(prompt_text: str) -> str:
    """
    Replace the initial baseline instruction with a fixed explicit chain-of-thought version.
    """
    original = "You are answering a multiple-choice question."

    if not prompt_text.startswith(original):
        raise ValueError(
            "Prompt does not start with the expected baseline instruction."
        )

    return prompt_text.replace(original, CHAIN_OF_THOUGHT_INSTRUCTION, 1)


def apply_fixed_transformation(prompt_text: str, transformation_name: str) -> str:
    if transformation_name == "role_based":
        return apply_role_based_prompting(prompt_text)
    if transformation_name == "chain_of_thought":
        return apply_chain_of_thought_prompting(prompt_text)

    raise ValueError(f"Unsupported transformation: {transformation_name}")


def build_mutant_record(prompt_record: dict, transformation_name: str) -> dict:
    original_prompt = prompt_record["prompt_text"]
    mutated_prompt = apply_fixed_transformation(original_prompt, transformation_name)

    return {
        "example_id": prompt_record["example_id"],
        "category": prompt_record["category"],
        "prompt_type": transformation_name,
        "prompt_text": mutated_prompt,
        "gold_label": prompt_record["gold_label"],
        "gold_answer": prompt_record["gold_answer"],
        "stereotyped_groups": prompt_record.get("stereotyped_groups", [])
    }


def default_output_path(transformation_name: str) -> Path:
    return (
        Path("data")
        / "prompts"
        / "final"
        / transformation_name
        / f"{transformation_name}_prompts.json"
    )


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(args.transformation)

    prompt_records = load_prompt_records(input_path)
    mutant_records = []

    for i, record in enumerate(prompt_records, start=1):
        mutant_record = build_mutant_record(record, args.transformation)
        mutant_records.append(mutant_record)

        print(f"[{i}/{len(prompt_records)}] Example ID: {record['example_id']}")

    save_prompt_records(mutant_records, output_path)
    print(f"Saved {len(mutant_records)} fixed mutants to: {output_path}")


if __name__ == "__main__":
    main()