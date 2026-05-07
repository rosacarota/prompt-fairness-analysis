import json
from pathlib import Path

from prompt_builder import load_examples, build_baseline_prompt


INPUT_PATH = Path("data/processed/bbq_disambiguated_594_sample.json")
OUTPUT_PATH = Path("data/prompts/baseline/baseline_prompts.json")


def save_prompts(records: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def build_prompt_record(example: dict) -> dict:
    prompt_text = build_baseline_prompt(example)

    return {
        "example_id": example["example_id"],
        "category": example["category"],
        "question_polarity": example["question_polarity"],

        "prompt_type": "baseline",
        "transformation_name": "baseline",
        "prompt_text": prompt_text,

        "context": example["context"],
        "question": example["question"],
        "answers": example["answers"],

        "gold_label": example["label"],
        "gold_answer": example["gold_answer"],
        "answer_info": example["answer_info"],
        "stereotyped_groups": example.get("stereotyped_groups", []),

        "target": example.get("target", ""),
        "non_target": example.get("non_target", ""),
        "unknown": example.get("unknown", ""),
    }


def main():
    examples = load_examples(str(INPUT_PATH))
    records = []

    for example in examples:
        records.append(build_prompt_record(example))

    save_prompts(records, OUTPUT_PATH)
    print(f"Saved {len(records)} baseline prompts to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()