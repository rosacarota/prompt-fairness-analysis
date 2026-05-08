import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from prompt_builder import build_baseline_prompt
from prompt_rewriter import clean_rewritten_context


DEFAULT_MODEL_NAME = "openai/gpt-5-mini"
DEFAULT_TEMPERATURE = 0.0
MAX_RETRIES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate attribute-position prompt variants from context-rewriting "
            "meta-prompts."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input meta-prompts JSON file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output final prompt JSON file.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Rewriter model name.",
    )

    return parser.parse_args()


load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def load_metaprompts(input_path: Path) -> list:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of meta-prompt records.")

    return data


def query_model(system_prompt: str, user_prompt: str, model_name: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=DEFAULT_TEMPERATURE,
    )

    content = response.choices[0].message.content
    return content.strip() if content else ""


def save_json(records: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def build_failures_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_failures{output_path.suffix}")


def build_final_mutant_record(record: dict, rewritten_context: str) -> dict:
    """
    Build an inference-ready prompt record with the same structure used by:
    - baseline
    - role_based
    - chain_of_thought

    Intermediate fields such as system_prompt, user_prompt, and original_context
    are intentionally not included in the final output.
    """
    transformation_name = record["transformation_name"]

    transformed_example = {
        "example_id": record["example_id"],
        "category": record["category"],
        "question_polarity": record["question_polarity"],

        "context": rewritten_context,
        "question": record["question"],
        "answers": record["answers"],

        "gold_label": record["gold_label"],
        "gold_answer": record["gold_answer"],
        "answer_info": record.get("answer_info", {}),
        "stereotyped_groups": record.get("stereotyped_groups", []),

        "target": record.get("target", ""),
        "non_target": record.get("non_target", ""),
        "unknown": record.get("unknown", ""),
    }

    prompt_text = build_baseline_prompt(transformed_example)

    return {
        "example_id": record["example_id"],
        "category": record["category"],
        "question_polarity": record["question_polarity"],

        "prompt_type": transformation_name,
        "transformation_name": transformation_name,
        "prompt_text": prompt_text,

        "context": rewritten_context,
        "question": record["question"],
        "answers": record["answers"],

        "gold_label": record["gold_label"],
        "gold_answer": record["gold_answer"],
        "answer_info": record.get("answer_info", {}),
        "stereotyped_groups": record.get("stereotyped_groups", []),

        "target": record.get("target", ""),
        "non_target": record.get("non_target", ""),
        "unknown": record.get("unknown", ""),
    }


def rewrite_context_with_retries(record: dict, model_name: str) -> tuple[str, str, int]:
    """
    Return:
    - cleaned rewritten context
    - last raw model output
    - number of attempts used

    Raises no exception. If all attempts fail, rewritten_context is "".
    """
    last_raw_output = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_output = query_model(
                system_prompt=record["system_prompt"],
                user_prompt=record["user_prompt"],
                model_name=model_name,
            )
        except Exception as e:
            raw_output = ""
            print(
                f"[WARNING] Request failed for example_id={record.get('example_id')} "
                f"on attempt {attempt}/{MAX_RETRIES}: {e}"
            )

        last_raw_output = raw_output
        rewritten_context = clean_rewritten_context(raw_output)

        if rewritten_context:
            return rewritten_context, last_raw_output, attempt

        print(
            f"[WARNING] Empty rewritten context for example_id={record.get('example_id')} "
            f"on attempt {attempt}/{MAX_RETRIES}"
        )
        print("Raw model output repr:")
        print(repr(raw_output))
        print("-" * 60)

    return "", last_raw_output, MAX_RETRIES


def build_failure_record(record: dict, raw_output: str, attempts: int) -> dict:
    """
    Store failed rewrites for later inspection/regeneration.
    """
    return {
        "example_id": record.get("example_id"),
        "category": record.get("category"),
        "question_polarity": record.get("question_polarity"),
        "source_prompt_type": record.get("source_prompt_type"),
        "transformation_name": record.get("transformation_name"),
        "transformation_target": record.get("transformation_target"),

        "attempts": attempts,
        "failure_reason": "empty_rewritten_context",
        "last_raw_output": raw_output,

        "system_prompt": record.get("system_prompt", ""),
        "user_prompt": record.get("user_prompt", ""),

        "original_context": record.get("original_context", ""),
        "question": record.get("question", ""),
        "answers": record.get("answers", {}),

        "gold_label": record.get("gold_label"),
        "gold_answer": record.get("gold_answer"),
        "answer_info": record.get("answer_info", {}),
        "stereotyped_groups": record.get("stereotyped_groups", []),

        "target": record.get("target", ""),
        "non_target": record.get("non_target", ""),
        "unknown": record.get("unknown", ""),
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    failures_output_path = build_failures_output_path(output_path)

    metaprompt_records = load_metaprompts(input_path)
    mutant_records = []
    failure_records = []

    for i, record in enumerate(metaprompt_records, start=1):
        rewritten_context, raw_output, attempts = rewrite_context_with_retries(
            record=record,
            model_name=args.model,
        )

        if not rewritten_context:
            failure_record = build_failure_record(
                record=record,
                raw_output=raw_output,
                attempts=attempts,
            )
            failure_records.append(failure_record)

            print(
                f"[FAILED] Example ID: {record.get('example_id')} "
                f"after {attempts} attempts. Skipping this record."
            )
            print("=" * 60)

            # Save progress after a failure.
            save_json(mutant_records, output_path)
            save_json(failure_records, failures_output_path)
            continue

        mutant_record = build_final_mutant_record(
            record=record,
            rewritten_context=rewritten_context,
        )

        mutant_records.append(mutant_record)

        print(f"[{i}/{len(metaprompt_records)}] Example ID: {record['example_id']}")
        print("Transformation:", record["transformation_name"])
        print(f"Attempts used: {attempts}")
        print("Original context preview:")
        print(
            record["original_context"][:160]
            + ("..." if len(record["original_context"]) > 160 else "")
        )
        print("Rewritten context preview:")
        print(
            rewritten_context[:160]
            + ("..." if len(rewritten_context) > 160 else "")
        )
        print("-" * 60)

        # Save progress after every successful record.
        save_json(mutant_records, output_path)

    save_json(mutant_records, output_path)

    if failure_records:
        save_json(failure_records, failures_output_path)

    print(f"Saved {len(mutant_records)} final mutant prompts to: {output_path}")

    if failure_records:
        print(f"Saved {len(failure_records)} failed rewrites to: {failures_output_path}")
    else:
        print("No failed rewrites.")


if __name__ == "__main__":
    main()