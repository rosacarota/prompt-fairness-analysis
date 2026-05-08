import argparse
import json
from pathlib import Path

from prompt_rewriter import load_prompts, build_rewriter_record


INPUT_PATH = Path("data/prompts/baseline/baseline_prompts.json")
OUTPUT_BASE_DIR = Path("data/meta-prompting")

TRANSFORMATIONS = [
    "attribute_early",
    "attribute_late",
]


def build_output_path(transformation_name: str, base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{transformation_name}_metaprompts.json"


def save_metaprompts(records: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate context-rewriting meta-prompts for a selected transformation."
    )

    parser.add_argument(
        "--transformation",
        type=str,
        required=True,
        choices=TRANSFORMATIONS,
        help="Transformation name.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="Input baseline prompts JSON file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_BASE_DIR,
        help="Output directory for meta-prompts.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prompt_records = load_prompts(str(args.input))
    output_path = build_output_path(args.transformation, args.output_dir)

    metaprompt_records = []

    for prompt_record in prompt_records:
        rewriter_record = build_rewriter_record(
            prompt_record=prompt_record,
            transformation_name=args.transformation,
        )
        metaprompt_records.append(rewriter_record)

    save_metaprompts(metaprompt_records, output_path)

    print(f"Transformation: {args.transformation}")
    print(f"Saved {len(metaprompt_records)} meta-prompts to: {output_path}")


if __name__ == "__main__":
    main()