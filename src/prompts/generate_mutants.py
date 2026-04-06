import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_MODEL_NAME = "openai/gpt-4o-mini-2024-07-18"
DEFAULT_TEMPERATURE = 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate mutant prompts from meta-prompts."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input meta-prompts JSON file."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output mutants JSON file."
    )

    return parser.parse_args()


load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def load_metaprompts(input_path: Path) -> list:
    """
    Load meta-prompt records from JSON.
    """
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def query_model(prompt: str) -> str:
    """
    Send a meta-prompt to OpenRouter through the OpenAI SDK
    and return the rewritten prompt text.
    """
    response = client.chat.completions.create(
        model=DEFAULT_MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=DEFAULT_TEMPERATURE
    )

    content = response.choices[0].message.content
    return content.strip() if content else ""


def save_mutants(records: list, output_path: Path) -> None:
    """
    Save generated mutant prompts to JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    metaprompt_records = load_metaprompts(input_path)
    mutant_records = []

    for i, record in enumerate(metaprompt_records, start=1):
        rewritten_prompt = query_model(
            prompt=record["meta_prompt"]
        )

        mutant_record = {
            "example_id": record["example_id"],
            "category": record["category"],
            "source_prompt_type": record["source_prompt_type"],
            "transformation_name": record["transformation_name"],
            "transformation_target": record["transformation_target"],
            "original_prompt": record["original_prompt"],
            "meta_prompt": record["meta_prompt"],
            "rewritten_prompt": rewritten_prompt,
            "gold_label": record["gold_label"],
            "gold_answer": record["gold_answer"],
            "stereotyped_groups": record.get("stereotyped_groups", []),
            "rewriter_model": DEFAULT_MODEL_NAME
        }

        mutant_records.append(mutant_record)

        print(f"[{i}/{len(metaprompt_records)}] Example ID: {record['example_id']}")
        print("Transformation:", record["transformation_name"])
        print("Rewritten prompt preview:")
        print(rewritten_prompt[:200] + ("..." if len(rewritten_prompt) > 200 else ""))
        print("-" * 60)

    save_mutants(mutant_records, output_path)
    print(f"Saved {len(mutant_records)} mutant prompts to: {output_path}")


if __name__ == "__main__":
    main()