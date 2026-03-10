import json
import argparse
from pathlib import Path

import requests

from prompt_builder import load_examples, build_baseline_prompt


OLLAMA_URL = "http://localhost:11434/api/generate"


def query_ollama(prompt: str, model_name: str) -> str:
    """
    Send a prompt to Ollama and return the generated response.
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    return data["response"].strip()


def save_results(results: list, output_path: str):
    """
    Save inference results to a JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def build_prompt(example: dict, strategy: str) -> str:
    if strategy == "baseline":
        return build_baseline_prompt(example)

    raise ValueError(f"Unknown prompt strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser(description="Run BBQ inference with Ollama.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ollama model name, e.g. qwen2.5:7b or gemma3:4b"
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
        required=True,
        help="Path to save inference results"
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

    examples = load_examples(args.input)
    selected_examples = examples[:args.num_examples]

    results = []

    for i, example in enumerate(selected_examples, start=1):
        prompt = build_prompt(example, args.strategy)
        response_text = query_ollama(prompt, args.model)

        result = {
            "example_id": example["example_id"],
            "model_name": args.model,
            "prompt_strategy": args.strategy,
            "prompt_text": prompt,
            "raw_response": response_text,
            "gold_label": example["label"],
            "gold_answer": example["gold_answer"]
        }

        results.append(result)

        print(f"[{i}/{len(selected_examples)}] Example ID: {example['example_id']}")
        print("Model response:", response_text)
        print("-" * 50)

    save_results(results, args.output)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()