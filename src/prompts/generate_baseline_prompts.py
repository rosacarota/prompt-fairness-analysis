import json
from pathlib import Path

from prompt_builder import load_examples, build_baseline_prompt


INPUT_PATH = Path("data/processed/bbq_disambiguated_sample_380.json")
OUTPUT_PATH = Path("data/prompts/baseline/baseline_prompts.json")


def save_prompts(records: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main():
    examples = load_examples(str(INPUT_PATH))
    records = []

    for example in examples:
        prompt_text = build_baseline_prompt(example)

        record = {
            "example_id": example["example_id"],
            "category": example["category"],
            "prompt_type": "baseline",
            "prompt_text": prompt_text,
            "gold_label": example["label"],
            "gold_answer": example["gold_answer"],
            "stereotyped_groups": example["stereotyped_groups"]
            
        }

        records.append(record)

    save_prompts(records, OUTPUT_PATH)
    print(f"Saved {len(records)} baseline prompts to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()