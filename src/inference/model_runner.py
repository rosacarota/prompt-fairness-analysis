import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_PROMPTS_ROOT = Path("data/prompts")
DEFAULT_OUTPUT_ROOT = Path("experiments/outputs")
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SAVE_EVERY = 25
RAW_RESPONSE_PREVIEW_CHARS = 300

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_TOP_P = 1
DEFAULT_MAX_TOKENS = 512
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_SLEEP_SECONDS = 5

QWEN_DISABLE_THINKING = True
QWEN_REASONING_EFFORT = "none"

MODEL_ALIASES = {
    "llama31_8b": "meta-llama/llama-3.1-8b-instruct",
    "qwen3_8b": "qwen/qwen3-8b",
    "ministral3_8b": "mistralai/ministral-8b-2512",
}

MODEL_CHOICES = list(MODEL_ALIASES.keys()) + ["all"]

PROMPT_FILES = {
    "baseline": Path("baseline/baseline_prompts.json"),
    "role_based": Path("fixed/role_based/role_based_prompts.json"),
    "chain_of_thought": Path("fixed/chain_of_thought/chain_of_thought_prompts.json"),
    "attribute_early": Path("generated/attribute_early/attribute_early_prompts.json"),
    "attribute_late": Path("generated/attribute_late/attribute_late_prompts.json"),
}

PROMPT_TYPES = list(PROMPT_FILES.keys())


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "Missing OPENROUTER_API_KEY. "
        "Add it to your .env file as: OPENROUTER_API_KEY=your_key"
    )


client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    default_headers={
        "HTTP-Referer": "https://localhost",
        "X-Title": "BBQ Fairness Prompt Experiments",
    },
)


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of prompt records in: {path}")

    return data


def save_json(data: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sanitize_model_name(model_name: str) -> str:
    return (
        model_name
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def resolve_prompt_file(prompt_type: str, prompts_root: Path) -> Path:
    if prompt_type not in PROMPT_FILES:
        raise ValueError(
            f"Unknown prompt type: {prompt_type}. "
            f"Available: {PROMPT_TYPES + ['all']}"
        )

    path = prompts_root / PROMPT_FILES[prompt_type]

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    return path


def selected_prompt_types(prompt_type: str) -> list[str]:
    if prompt_type == "all":
        return PROMPT_TYPES

    return [prompt_type]


def resolve_model_name(model_arg: str) -> str:
    if model_arg in MODEL_ALIASES:
        return MODEL_ALIASES[model_arg]

    if "/" in model_arg:
        return model_arg

    raise ValueError(
        f"Unknown model: {model_arg}. "
        f"Use one of {MODEL_CHOICES}, or pass a full OpenRouter model id."
    )


def selected_models(model_arg: str) -> list[tuple[str, str]]:
    if model_arg == "all":
        return list(MODEL_ALIASES.items())

    model_name = resolve_model_name(model_arg)

    if model_arg in MODEL_ALIASES:
        model_alias = model_arg
    else:
        model_alias = sanitize_model_name(model_arg)

    return [(model_alias, model_name)]


def extract_prompt_text(record: dict[str, Any]) -> str:
    prompt_text = record.get("prompt_text")

    if isinstance(prompt_text, str) and prompt_text.strip():
        return prompt_text.strip()

    raise ValueError(
        f"Missing prompt_text for "
        f"example_id={record.get('example_id')}, "
        f"category={record.get('category')}"
    )


def label_to_letter(label: Any) -> str:
    try:
        label_int = int(label)
    except (TypeError, ValueError):
        return "INVALID"

    mapping = {
        0: "A",
        1: "B",
        2: "C",
    }

    return mapping.get(label_int, "INVALID")


def normalize_letter(value: Any) -> str:
    if value is None:
        return ""

    value = str(value).strip().upper()

    if value in {"A", "B", "C"}:
        return value

    return ""


def get_answer_text_by_letter(record: dict[str, Any], letter: str) -> str:
    answers = record.get("answers", {})

    mapping = {
        "A": "ans0",
        "B": "ans1",
        "C": "ans2",
    }

    ans_key = mapping.get(letter)

    if not ans_key:
        return ""

    return str(answers.get(ans_key, ""))


def extract_line_value(raw_response: str, field_name: str) -> str:
    if not raw_response:
        return ""

    pattern = rf"(?im)^\s*{re.escape(field_name)}\s*:\s*(.*?)\s*$"
    match = re.search(pattern, raw_response)

    if not match:
        return ""

    return match.group(1).strip()


def extract_reasoning(raw_response: str) -> str:
    if not raw_response:
        return ""

    match = re.search(
        r"(?ims)^\s*Reasoning\s*:\s*(.*?)(?=^\s*Answer\s*:)",
        raw_response,
    )

    if match:
        return " ".join(match.group(1).strip().split())

    return extract_line_value(raw_response, "Reasoning")


def parse_answer(raw_response: str) -> str:
    if not raw_response:
        return "INVALID"

    answer_value = extract_line_value(raw_response, "Answer").upper()

    if answer_value in {"A", "B", "C"}:
        return answer_value

    if answer_value == "NA":
        return "NA"

    match = re.search(r"(?i)\bAnswer\s*:\s*([ABC])\b", raw_response)
    if match:
        return match.group(1).upper()

    match = re.fullmatch(r"\s*([ABC])\s*", raw_response, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return "INVALID"


def parse_model_response(raw_response: str, prompt_type: str) -> dict[str, Any]:
    check_status = extract_line_value(raw_response, "Check").upper()
    postcheck_status = extract_line_value(raw_response, "Postcheck").upper()
    reasoning = extract_reasoning(raw_response)
    parsed_answer = parse_answer(raw_response)

    has_check = check_status in {"UNDERSTOOD", "NOT_UNDERSTOOD"}
    has_answer = parsed_answer in {"A", "B", "C", "NA"}
    has_postcheck = postcheck_status in {"OK", "FAILED"}

    understood_format = (
        check_status == "UNDERSTOOD"
        and parsed_answer in {"A", "B", "C"}
        and postcheck_status == "OK"
    )

    not_understood_format = (
        check_status == "NOT_UNDERSTOOD"
        and parsed_answer == "NA"
        and postcheck_status == "FAILED"
    )

    basic_response_format_valid = int(
        has_check
        and has_answer
        and has_postcheck
        and (understood_format or not_understood_format)
    )

    if prompt_type == "chain_of_thought":
        strict_response_format_valid = int(
            basic_response_format_valid == 1
            and (
                bool(reasoning.strip())
                or not_understood_format
            )
        )
    else:
        strict_response_format_valid = basic_response_format_valid

    return {
        "check_status": check_status,
        "reasoning": reasoning,
        "parsed_answer": parsed_answer,
        "postcheck_status": postcheck_status,
        "has_check": int(has_check),
        "has_answer": int(has_answer),
        "has_postcheck": int(has_postcheck),
        "basic_response_format_valid": basic_response_format_valid,
        "strict_response_format_valid": strict_response_format_valid,
    }


def get_prediction_group(record: dict[str, Any], parsed_answer: str) -> str:
    target = normalize_letter(record.get("target"))
    non_target = normalize_letter(record.get("non_target"))
    unknown = normalize_letter(record.get("unknown"))

    if parsed_answer == target:
        return "target"

    if parsed_answer == non_target:
        return "non_target"

    if parsed_answer == unknown:
        return "unknown"

    if parsed_answer == "NA":
        return "not_understood"

    return "invalid"


def get_biased_and_antibiased_letters(record: dict[str, Any]) -> tuple[str, str]:
    question_polarity = str(record.get("question_polarity", "")).strip().lower()

    target = normalize_letter(record.get("target"))
    non_target = normalize_letter(record.get("non_target"))

    if question_polarity == "neg":
        biased_letter = target
        anti_biased_letter = non_target
    elif question_polarity == "nonneg":
        biased_letter = non_target
        anti_biased_letter = target
    else:
        biased_letter = ""
        anti_biased_letter = ""

    return biased_letter, anti_biased_letter


def compute_fairness_fields(
    record: dict[str, Any],
    parsed_answer: str,
    gold_letter: str,
) -> dict[str, Any]:
    target = normalize_letter(record.get("target"))
    non_target = normalize_letter(record.get("non_target"))
    unknown = normalize_letter(record.get("unknown"))

    biased_letter, anti_biased_letter = get_biased_and_antibiased_letters(record)

    prediction_group = get_prediction_group(record, parsed_answer)
    gold_group = get_prediction_group(record, gold_letter)

    return {
        "target": target,
        "non_target": non_target,
        "unknown": unknown,
        "biased_letter": biased_letter,
        "anti_biased_letter": anti_biased_letter,
        "prediction_group": prediction_group,
        "gold_group": gold_group,
        "is_target_prediction": int(parsed_answer == target),
        "is_non_target_prediction": int(parsed_answer == non_target),
        "is_unknown_prediction": int(parsed_answer == unknown),
        "is_biased_prediction": int(parsed_answer == biased_letter),
        "is_anti_biased_prediction": int(parsed_answer == anti_biased_letter),
        "is_gold_target": int(gold_letter == target),
        "is_gold_non_target": int(gold_letter == non_target),
        "is_gold_unknown": int(gold_letter == unknown),
        "predicted_answer_text": get_answer_text_by_letter(record, parsed_answer),
        "gold_answer_text_by_letter": get_answer_text_by_letter(record, gold_letter),
    }


def build_example_key(record: dict[str, Any]) -> str:
    return (
        f"{record.get('example_id')}::"
        f"{record.get('category')}::"
        f"{record.get('question_polarity')}"
    )

def is_qwen3_model(model_name: str) -> bool:
    model_name_lower = model_name.lower()
    return "qwen/qwen3" in model_name_lower or "qwen3" in model_name_lower


def build_extra_body_for_model(model_name: str) -> dict[str, Any] | None:
    if QWEN_DISABLE_THINKING and is_qwen3_model(model_name):
        return {
            "reasoning": {
                "effort": QWEN_REASONING_EFFORT,
                "exclude": True,
            }
        }

    return None

def query_model(prompt_text: str, model_name: str) -> str:
    last_error = ""

    for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
        try:
            request_kwargs: dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt_text}
                ],
                "temperature": DEFAULT_TEMPERATURE,
                "top_p": DEFAULT_TOP_P,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "timeout": DEFAULT_TIMEOUT_SECONDS,
            }

            extra_body = build_extra_body_for_model(model_name)

            if extra_body is not None:
                request_kwargs["extra_body"] = extra_body

            response = client.chat.completions.create(**request_kwargs)

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            last_error = str(e)

            if attempt < DEFAULT_MAX_RETRIES:
                sleep_seconds = DEFAULT_RETRY_SLEEP_SECONDS * attempt
                print(
                    f"[RETRY] attempt={attempt}/{DEFAULT_MAX_RETRIES} | "
                    f"sleep={sleep_seconds}s | error={last_error}"
                )
                time.sleep(sleep_seconds)

    raise RuntimeError(last_error)


def build_result_record(
    record: dict[str, Any],
    prompt_text: str,
    raw_response: str,
    parsed_response: dict[str, Any],
    model_alias: str,
    model_name: str,
    prompt_file: Path,
    run_id: int,
    request_failed: bool,
    error_message: str,
) -> dict[str, Any]:
    prompt_type = record.get("prompt_type", "")
    transformation_name = record.get("transformation_name", "")

    gold_letter = label_to_letter(record.get("gold_label"))
    parsed_answer = parsed_response["parsed_answer"]
    is_correct = int(parsed_answer == gold_letter)

    fairness_fields = compute_fairness_fields(
        record=record,
        parsed_answer=parsed_answer,
        gold_letter=gold_letter,
    )

    return {
        "run_id": run_id,
        "example_key": build_example_key(record),
        "example_id": record.get("example_id"),
        "category": record.get("category"),
        "question_polarity": record.get("question_polarity"),
        "api_provider": "openrouter",
        "model_alias": model_alias,
        "model_name": model_name,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "prompt_file": str(prompt_file),
        "prompt_type": prompt_type,
        "transformation_name": transformation_name,
        "context": record.get("context"),
        "question": record.get("question"),
        "answers": record.get("answers", {}),
        "answer_info": record.get("answer_info", {}),
        "stereotyped_groups": record.get("stereotyped_groups", []),
        "prompt_text": prompt_text,
        "raw_response": raw_response,
        "request_failed": int(request_failed),
        "error_message": error_message,
        "check_status": parsed_response["check_status"],
        "reasoning": parsed_response["reasoning"],
        "parsed_answer": parsed_answer,
        "postcheck_status": parsed_response["postcheck_status"],
        "has_check": parsed_response["has_check"],
        "has_answer": parsed_response["has_answer"],
        "has_postcheck": parsed_response["has_postcheck"],
        "basic_response_format_valid": parsed_response["basic_response_format_valid"],
        "strict_response_format_valid": parsed_response["strict_response_format_valid"],
        "gold_label": record.get("gold_label"),
        "gold_letter": gold_letter,
        "gold_answer": record.get("gold_answer"),
        **fairness_fields,
        "is_correct": is_correct,
    }


def build_run_output_path(
    output_root: Path,
    model_alias: str,
    prompt_type: str,
    run_id: int,
) -> Path:
    safe_model_name = sanitize_model_name(model_alias)

    run_dir = (
        output_root
        / safe_model_name
        / prompt_type
        / f"run_{run_id:02d}"
    )

    return run_dir / f"{prompt_type}_results.json"


def get_existing_run_ids(
    output_root: Path,
    model_alias: str,
    prompt_type: str,
) -> list[int]:
    safe_model_name = sanitize_model_name(model_alias)
    prompt_output_dir = output_root / safe_model_name / prompt_type

    if not prompt_output_dir.exists():
        return []

    run_ids = []

    for child in prompt_output_dir.iterdir():
        if not child.is_dir():
            continue

        match = re.fullmatch(r"run_(\d+)", child.name)

        if match:
            run_ids.append(int(match.group(1)))

    return sorted(run_ids)


def get_next_run_id(
    output_root: Path,
    model_alias: str,
    prompt_type: str,
) -> int:
    existing_run_ids = get_existing_run_ids(
        output_root=output_root,
        model_alias=model_alias,
        prompt_type=prompt_type,
    )

    if not existing_run_ids:
        return 1

    return max(existing_run_ids) + 1


def print_result_preview(result: dict[str, Any], raw_response: str) -> None:
    raw_preview = raw_response[:RAW_RESPONSE_PREVIEW_CHARS].replace("\n", "\\n")

    print(f"  raw_response_preview={raw_preview}")
    print(
        f"  parsed={result['parsed_answer']} | "
        f"gold={result['gold_letter']} | "
        f"correct={result['is_correct']} | "
        f"group={result['prediction_group']} | "
        f"biased={result['is_biased_prediction']} | "
        f"anti_biased={result['is_anti_biased_prediction']} | "
        f"format={result['strict_response_format_valid']}"
    )


def run_single_prompt_type(
    prompt_type: str,
    prompt_file: Path,
    model_alias: str,
    model_name: str,
    output_root: Path,
    num_runs: int,
    num_examples: int | None,
    save_every: int,
) -> None:
    records = load_json(prompt_file)

    if num_examples is not None:
        records = records[:num_examples]

    existing_run_ids = get_existing_run_ids(
        output_root=output_root,
        model_alias=model_alias,
        prompt_type=prompt_type,
    )

    start_run_id = get_next_run_id(
        output_root=output_root,
        model_alias=model_alias,
        prompt_type=prompt_type,
    )

    end_run_id = start_run_id + num_runs - 1

    print()
    print("=" * 70)
    print(f"Model alias: {model_alias}")
    print(f"OpenRouter model: {model_name}")
    print(f"Prompt type: {prompt_type}")
    print(f"Prompt file: {prompt_file}")
    print(f"Records: {len(records)}")
    print(f"Existing runs: {existing_run_ids}")
    print(f"New runs to create: run_{start_run_id:02d} -> run_{end_run_id:02d}")
    print("=" * 70)

    for run_id in range(start_run_id, end_run_id + 1):
        output_path = build_run_output_path(
            output_root=output_root,
            model_alias=model_alias,
            prompt_type=prompt_type,
            run_id=run_id,
        )

        results = []

        print()
        print(f"--- {model_alias} | {prompt_type} | run_{run_id:02d} ---")
        print(f"Output: {output_path}")

        for index, record in enumerate(records, start=1):
            prompt_text = extract_prompt_text(record)

            print(
                f"[{index}/{len(records)}] "
                f"model={model_alias} | "
                f"run_{run_id:02d} | "
                f"example_id={record.get('example_id')} | "
                f"category={record.get('category')}"
            )

            request_failed = False
            error_message = ""
            raw_response = ""

            try:
                raw_response = query_model(
                    prompt_text=prompt_text,
                    model_name=model_name,
                )
            except Exception as e:
                request_failed = True
                error_message = str(e)
                print(f"[ERROR] {error_message}")

            parsed_response = parse_model_response(
                raw_response=raw_response,
                prompt_type=str(record.get("prompt_type", "")),
            )

            result = build_result_record(
                record=record,
                prompt_text=prompt_text,
                raw_response=raw_response,
                parsed_response=parsed_response,
                model_alias=model_alias,
                model_name=model_name,
                prompt_file=prompt_file,
                run_id=run_id,
                request_failed=request_failed,
                error_message=error_message,
            )

            results.append(result)

            print_result_preview(
                result=result,
                raw_response=raw_response,
            )

            if save_every > 0 and index % save_every == 0:
                save_json(results, output_path)
                print(f"  [partial save] {output_path}")

        save_json(results, output_path)

        print(
            f"[DONE] {model_alias} | {prompt_type} run_{run_id:02d}: "
            f"saved {len(results)} records to {output_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference over standardized prompt files using OpenRouter."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODEL_CHOICES,
        help=(
            "Model alias to run. "
            "Available: llama31_8b, qwen3_8b, ministral3_8b, all."
        ),
    )

    parser.add_argument(
        "--prompt-type",
        type=str,
        required=True,
        choices=PROMPT_TYPES + ["all"],
        help="Prompt type to run, or 'all'.",
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help=(
            "Number of new runs to create. "
            "If previous runs exist, the script starts from the next run id."
        ),
    )

    parser.add_argument(
        "--prompts-root",
        type=Path,
        default=DEFAULT_PROMPTS_ROOT,
        help="Root directory containing prompt JSON files.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory.",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Optional number of examples to run from each prompt file.",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=DEFAULT_SAVE_EVERY,
        help="Partial save every N examples.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prompt_types = selected_prompt_types(args.prompt_type)
    models_to_run = selected_models(args.model)

    print("=== Inference configuration ===")
    print("API provider: OpenRouter")
    print(f"Requested model: {args.model}")
    print("Resolved models:")
    for model_alias, model_name in models_to_run:
        print(f"  - {model_alias}: {model_name}")
    print(f"Temperature: {DEFAULT_TEMPERATURE}")
    print(f"Top-p: {DEFAULT_TOP_P}")
    print(f"Max tokens: {DEFAULT_MAX_TOKENS}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Prompt root: {args.prompts_root}")
    print(f"Output root: {args.output_root}")
    print(f"New runs requested: {args.num_runs}")
    print(f"Num examples: {args.num_examples}")
    print()

    for model_alias, model_name in models_to_run:
        print()
        print("#" * 70)
        print(f"MODEL: {model_alias}")
        print(f"OPENROUTER ID: {model_name}")
        print("#" * 70)

        for prompt_type in prompt_types:
            prompt_file = resolve_prompt_file(
                prompt_type=prompt_type,
                prompts_root=args.prompts_root,
            )

            run_single_prompt_type(
                prompt_type=prompt_type,
                prompt_file=prompt_file,
                model_alias=model_alias,
                model_name=model_name,
                output_root=args.output_root,
                num_runs=args.num_runs,
                num_examples=args.num_examples,
                save_every=args.save_every,
            )


if __name__ == "__main__":
    main()