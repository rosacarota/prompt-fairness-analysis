import json
import argparse
from pathlib import Path

from prompt_rewriter import load_prompts, build_rewriter_record


INPUT_PATH = Path("data/prompts/baseline/baseline_prompts.json")
OUTPUT_BASE_DIR = Path("data/meta-prompting")


def build_output_path(transformation_name: str, base_dir: Path) -> Path:
    """
    Build the output file path for the selected transformation.
    All meta-prompts are stored in the same folder, while the file name
    indicates the transformation that was applied.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{transformation_name}_metaprompts.json"


def save_metaprompts(records: list, output_path: Path) -> None:
    """
    Save generated meta-prompt records to JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate meta-prompts for a selected transformation.")
    parser.add_argument(
        "--transformation",
        type=str,
        required=True,
        help="Transformation name, e.g. attribute_early"
    )

    args = parser.parse_args()

    prompt_records = load_prompts(str(INPUT_PATH))
    output_path = build_output_path(args.transformation, OUTPUT_BASE_DIR)

    metaprompt_records = []

    for prompt_record in prompt_records:
        rewriter_record = build_rewriter_record(
            prompt_record=prompt_record,
            transformation_name=args.transformation
        )
        metaprompt_records.append(rewriter_record)

    save_metaprompts(metaprompt_records, output_path)

    print(f"Transformation: {args.transformation}")
    print(f"Saved {len(metaprompt_records)} meta-prompts to: {output_path}")


if __name__ == "__main__":
    main()