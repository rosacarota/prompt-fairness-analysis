import json
import argparse
import re
import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

from inference.prompt_builder import load_examples, build_baseline_prompt


load_dotenv()  # Load environment variables from .env file

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found. Please set it in your .env file.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def query_model(prompt: str, model_name: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

def save_results(results: list, output_path: str):
    """
    Save inference results to a JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def build_prompt(example: dict, strategy: str) -> str:
    """
    Build the prompt according to the selected strategy.
    """
    if strategy == "baseline":
        return build_baseline_prompt(example)

    raise ValueError(f"Unknown prompt strategy: {strategy}")


def label_to_letter(label: int) -> str:
    """
    Convert BBQ label indices to answer letters.
    0 -> A, 1 -> B, 2 -> C
    """
    mapping = {0: "A", 1: "B", 2: "C"}
    return mapping.get(label, "INVALID")


def parse_model_answer(raw_response: str) -> str:
    """
    Normalize the model response to one of: A, B, C, INVALID.
    """
    if not raw_response:
        return "INVALID"

    cleaned = raw_response.strip().upper()

    if cleaned in {"A", "B", "C"}:
        return cleaned

    match = re.search(r"\b([ABC])\b", cleaned)
    if match:
        return match.group(1)

    return "INVALID"


def compute_correctness(parsed_answer: str, gold_letter: str) -> int:
    """
    Return 1 if the parsed answer matches the gold answer, else 0.
    """
    return int(parsed_answer == gold_letter)


def sanitize_model_name(model_name: str) -> str:
    """
    Convert the model name into a filesystem-safe string.
    """
    return model_name.replace(":", "_").replace("/", "_")


def build_output_path(model_name: str, base_output_dir: str = "experiments/outputs") -> str:
    """
    Create a model-specific output folder and generate an incremental file name.
    """
    safe_model_name = sanitize_model_name(model_name)
    model_dir = Path(base_output_dir) / safe_model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(model_dir.glob(f"{safe_model_name}_test_*.json"))

    max_index = 0
    pattern = re.compile(rf"{re.escape(safe_model_name)}_test_(\d+)\.json$")

    for file_path in existing_files:
        match = pattern.search(file_path.name)
        if match:
            file_index = int(match.group(1))
            max_index = max(max_index, file_index)

    next_index = max_index + 1
    output_path = model_dir / f"{safe_model_name}_test_{next_index}.json"

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Run BBQ inference with the OpenAI Python SDK.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name, e.g. openai/gpt-oss-120b:free"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/bbq_disambiguated.json",
        help="Path to the processed input dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/outputs",
        help="Base directory where model-specific result folders will be created"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="baseline",
        help="Prompt strategy to use"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1,
        help="Number of examples to run"
    )

    args = parser.parse_args()

    output_path = build_output_path(args.model, args.output)

    examples = load_examples(args.input)
    selected_examples = examples[:args.num_examples]

    results = []

    for i, example in enumerate(selected_examples, start=1):
        prompt = build_prompt(example, args.strategy)
        response_text = query_model(prompt, args.model)

        parsed_answer = parse_model_answer(response_text)
        gold_letter = label_to_letter(example["label"])
        is_correct = compute_correctness(parsed_answer, gold_letter)

        result = {
            "example_id": example["example_id"],
            "model_name": args.model,
            "prompt_strategy": args.strategy,
            "prompt_text": prompt,
            "raw_response": response_text,
            "parsed_answer": parsed_answer,
            "gold_label": example["label"],
            "gold_letter": gold_letter,
            "gold_answer": example["gold_answer"],
            "is_correct": is_correct
        }

        results.append(result)

        print(f"[{i}/{len(selected_examples)}] Example ID: {example['example_id']}")
        print("Raw response:", response_text)
        print("Parsed answer:", parsed_answer)
        print("Gold letter:", gold_letter)
        print("Correct:", is_correct)
        print("-" * 50)

    save_results(results, output_path)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()