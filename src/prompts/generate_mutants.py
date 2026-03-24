import json
from pathlib import Path

from openai import OpenAI


INPUT_PATH = Path("data/meta-prompting/attribute_early_metaprompts.json")
OUTPUT_PATH = Path("data/prompts/mutants/attribute_early_mutants.json")

MODEL_NAME = "llama3.1:8b"
TEMPERATURE = 0.0

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


def load_metaprompts(input_path: Path) -> list:
    """
    Load meta-prompt records from JSON.
    """
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def query_model(prompt: str, model_name: str) -> str:
    """
    Send a meta-prompt to the local Ollama model through the OpenAI SDK
    and return the rewritten prompt text.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE
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
    metaprompt_records = load_metaprompts(INPUT_PATH)

    mutant_records = []

    for i, record in enumerate(metaprompt_records, start=1):
        rewritten_prompt = query_model(record["meta_prompt"], MODEL_NAME)

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
            "rewriter_model": MODEL_NAME
        }

        mutant_records.append(mutant_record)

        print(f"[{i}/{len(metaprompt_records)}] Example ID: {record['example_id']}")
        print("Transformation:", record["transformation_name"])
        print("Rewritten prompt preview:")
        print(rewritten_prompt[:200] + ("..." if len(rewritten_prompt) > 200 else ""))
        print("-" * 60)

    save_mutants(mutant_records, OUTPUT_PATH)
    print(f"Saved {len(mutant_records)} mutant prompts to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()