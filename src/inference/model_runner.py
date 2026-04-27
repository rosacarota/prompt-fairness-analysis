import json
import argparse
import re
from pathlib import Path

from openai import OpenAI

TEMPERATURE = 0.0

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


def load_prompt_records(json_path: str) -> list:
    """
    Load prompt records from a JSON file.
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def query_model(prompt: str, model_name: str, temperature: float = 0.0) -> str:
    """
    Query a local Ollama model through the OpenAI-compatible API.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )

    content = response.choices[0].message.content
    return content.strip() if content else ""


def save_results(results: list, output_path: str) -> None:
    """
    Save inference results to JSON.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def extract_prompt_text(record: dict) -> str:
    """
    Prefer rewritten_prompt for normalized/fixed files,
    fallback to prompt_text for legacy files.
    """
    if "rewritten_prompt" in record and isinstance(record["rewritten_prompt"], str):
        if record["rewritten_prompt"].strip():
            return record["rewritten_prompt"].strip()

    if "prompt_text" in record and isinstance(record["prompt_text"], str):
        if record["prompt_text"].strip():
            return record["prompt_text"].strip()

    raise ValueError(
        f"Missing prompt text for example_id={record.get('example_id')}. "
        f"Available keys: {list(record.keys())}"
    )


def label_to_letter(label: int) -> str:
    """
    Convert BBQ label indices to answer letters.
    0 -> A, 1 -> B, 2 -> C
    """
    mapping = {0: "A", 1: "B", 2: "C"}
    return mapping.get(label, "INVALID")


def parse_model_answer(raw_response: str, record: dict) -> str:
    """
    Normalize the model response to one of: A, B, C, INVALID.

    For fixed prompts (role_based / chain_of_thought), prefer:
    Answer: A

    For legacy prompts, keep the simpler parsing.
    """
    if not raw_response:
        return "INVALID"

    text = raw_response.strip()
    cleaned = text.upper()

    transformation_name = record.get("transformation_name")
    prompt_type = record.get("prompt_type")

    is_fixed_prompt = (
        transformation_name in {"role_based", "chain_of_thought"}
        or prompt_type in {"role_based", "chain_of_thought"}
    )

    if is_fixed_prompt:
        match = re.search(r"(?im)^Answer:\s*([ABC])\s*$", text)
        if match:
            return match.group(1).upper()

        match = re.search(r"(?i)answer:\s*([ABC])\b", text)
        if match:
            return match.group(1).upper()

        match = re.fullmatch(r"\s*([ABC])\s*", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

        return "INVALID"

    if cleaned in {"A", "B", "C"}:
        return cleaned

    match = re.search(r"\b([ABC])\b", cleaned)
    if match:
        return match.group(1)

    return "INVALID"


def compute_correctness(parsed_answer: str, gold_letter: str) -> int:
    return int(parsed_answer == gold_letter)


def sanitize_model_name(model_name: str) -> str:
    """
    Convert the model name into a filesystem-safe string.
    """
    return model_name.replace(":", "_").replace("/", "_")


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def build_output_path(
    model_name: str,
    input_file: str,
    base_output_dir: str = "experiments/outputs",
    unique: bool = True
) -> str:
    safe_model_name = sanitize_model_name(model_name)
    model_dir = Path(base_output_dir) / safe_model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    input_stem = Path(input_file).stem
    output_path = model_dir / f"{input_stem}_results.json"

    if unique:
        output_path = ensure_unique_path(output_path)

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on prompt files with local Ollama models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Local Ollama model name, e.g. llama3.1:8b"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the prompt JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/outputs",
        help="Base directory where result files will be saved"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Optional number of prompt records to run"
    )
    parser.add_argument(
        "--no-unique-names",
        action="store_true",
        help="If set, do not append _1, _2, ... when output file already exists"
    )

    args = parser.parse_args()

    prompt_records = load_prompt_records(args.input)

    if args.num_examples is not None:
        prompt_records = prompt_records[:args.num_examples]

    output_path = build_output_path(
        model_name=args.model,
        input_file=args.input,
        base_output_dir=args.output,
        unique=not args.no_unique_names
    )

    results = []

    for i, record in enumerate(prompt_records, start=1):
        prompt_text = extract_prompt_text(record)

        print(f"[{i}/{len(prompt_records)}] Example ID: {record['example_id']} - sending request...")

        try:
            response_text = query_model(prompt_text, args.model, TEMPERATURE)
        except Exception as e:
            print(f"Request failed: {e}")
            response_text = ""

        parsed_answer = parse_model_answer(response_text, record)
        gold_letter = label_to_letter(record["gold_label"])
        is_correct = compute_correctness(parsed_answer, gold_letter)

        result = {
            "example_id": record["example_id"],
            "category": record.get("category"),
            "model_name": args.model,
            "input_file": args.input,
            "prompt_source_type": record.get("source_prompt_type", record.get("prompt_type")),
            "transformation_name": record.get("transformation_name"),
            "rewriter_model": record.get("rewriter_model"),
            "prompt_text": prompt_text,
            "raw_response": response_text,
            "parsed_answer": parsed_answer,
            "gold_label": record["gold_label"],
            "gold_letter": gold_letter,
            "gold_answer": record["gold_answer"],
            "stereotyped_groups": record.get("stereotyped_groups", []),
            "is_correct": is_correct
        }

        results.append(result)

        print("Raw response:", response_text)
        print("Parsed answer:", parsed_answer)
        print("Gold letter:", gold_letter)
        print("Correct:", is_correct)
        print("-" * 50)

    save_results(results, output_path)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()