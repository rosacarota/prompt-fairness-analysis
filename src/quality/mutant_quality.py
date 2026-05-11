import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer, util

    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


REQUIRED_FIELDS = [
    "example_id",
    "category",
    "question_polarity",
    "prompt_type",
    "transformation_name",
    "prompt_text",
    "context",
    "question",
    "answers",
    "gold_label",
    "gold_answer",
    "answer_info",
    "stereotyped_groups",
    "target",
    "non_target",
    "unknown",
]

INVARIANT_FIELDS = [
    "question_polarity",
    "question",
    "answers",
    "gold_label",
    "gold_answer",
    "answer_info",
    "stereotyped_groups",
    "target",
    "non_target",
    "unknown",
]


def load_json(json_path: str | Path) -> list:
    path = Path(json_path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of records.")

    return data


def save_json(data: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def infer_transformation_name(
    records: list[dict],
    fallback: str = "unknown_transformation",
) -> str:
    names = {
        r.get("transformation_name")
        for r in records
        if r.get("transformation_name")
    }

    if not names:
        return fallback

    if len(names) > 1:
        print(f"Warning: multiple transformation names found: {sorted(names)}")
        print(f"Using: {sorted(names)[0]}")

    return sorted(names)[0]


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
    input_file: str,
    output_dir: str = "experiments/quality_checks",
    transformation_name: str | None = None,
    unique: bool = True,
) -> Path:
    input_path = Path(input_file)
    input_stem = input_path.stem

    if transformation_name is None:
        transformation_name = "unknown_transformation"

    output_path = Path(output_dir) / transformation_name / f"{input_stem}_quality.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if unique:
        output_path = ensure_unique_path(output_path)

    return output_path


def deserialize_prompt_text(text: str) -> str:
    """
    Normalize prompt text so checks work both with:
    - real newlines
    - serialized escaped newlines like \\n
    - malformed backslash-newline patterns like \\\n
    """
    if not text:
        return ""

    text = str(text).replace("\r\n", "\n")

    if "\\n" in text or "\\t" in text or '\\"' in text or "\\'" in text:
        try:
            text = bytes(text, "utf-8").decode("unicode_escape")
        except Exception:
            pass

    text = re.sub(r"\\+\n", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)

    return text


def normalize_text(text: str) -> str:
    text = deserialize_prompt_text(text)
    return " ".join(text.strip().split()) if text else ""


def make_key(record: dict) -> tuple:
    """
    example_id alone is not globally unique in BBQ.
    We use example_id + category.
    """
    return (
        record.get("example_id"),
        record.get("category"),
    )


def build_index(records: list[dict], name: str) -> dict:
    keys = [make_key(record) for record in records]
    duplicates = [
        key
        for key, count in Counter(keys).items()
        if count > 1
    ]

    if duplicates:
        raise ValueError(
            f"Duplicate keys found in {name}. "
            f"First duplicates: {duplicates[:10]}"
        )

    return {
        make_key(record): record
        for record in records
    }


def extract_answer_options_from_prompt(text: str) -> dict:
    """
    Extract A/B/C options from the prompt_text.

    This reads only the block after 'Options:' and stops after A, B, C
    have been found and the next blank line is reached.
    """
    if not text:
        return {}

    text = deserialize_prompt_text(text)
    lines = text.splitlines()

    options = {}
    in_options_block = False

    for line in lines:
        stripped = line.strip()

        if stripped == "Options:":
            in_options_block = True
            continue

        if not in_options_block:
            continue

        if stripped == "":
            if set(options.keys()) == {"A", "B", "C"}:
                break
            continue

        match = re.match(r"^\s*([ABC])[\.\)]\s*(.*)$", line)

        if match:
            label = match.group(1)
            content = match.group(2)
            options[label] = normalize_text(content)

    return options


def options_dict_to_letters(answers: dict) -> dict:
    return {
        "A": normalize_text(answers.get("ans0", "")),
        "B": normalize_text(answers.get("ans1", "")),
        "C": normalize_text(answers.get("ans2", "")),
    }


def exact_preservation_of_answer_options(
    baseline_record: dict,
    generated_record: dict,
) -> int:
    return int(
        options_dict_to_letters(baseline_record.get("answers", {}))
        == options_dict_to_letters(generated_record.get("answers", {}))
    )


def prompt_contains_answer_options(generated_record: dict) -> int:
    prompt_text = generated_record.get("prompt_text", "")
    extracted_options = extract_answer_options_from_prompt(prompt_text)
    expected_options = options_dict_to_letters(generated_record.get("answers", {}))

    return int(extracted_options == expected_options)


def has_labels_a_b_c(text: str) -> int:
    if not text:
        return 0

    text = deserialize_prompt_text(text)

    has_a = bool(re.search(r"^\s*A[\.\)]", text, flags=re.MULTILINE))
    has_b = bool(re.search(r"^\s*B[\.\)]", text, flags=re.MULTILINE))
    has_c = bool(re.search(r"^\s*C[\.\)]", text, flags=re.MULTILINE))

    return int(has_a and has_b and has_c)


def has_three_options(text: str) -> int:
    options = extract_answer_options_from_prompt(text)
    return int(set(options.keys()) == {"A", "B", "C"})


def no_unresolved_placeholders(text: str) -> int:
    if not text:
        return 0

    text = deserialize_prompt_text(text)

    patterns = [
        r"\{\s*[^{}]+\s*\}",
        r"\[\s*PLACEHOLDER.*?\]",
        r"<\s*placeholder.*?>",
        r"\bTODO\b",
        r"\bTBD\b",
    ]

    found = any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )

    return int(not found)


def preserve_required_sections(text: str) -> int:
    if not text:
        return 0

    text = deserialize_prompt_text(text)

    required = [
        "Context:",
        "Question:",
        "Options:",
        "Check:",
        "Answer:",
        "Postcheck:",
    ]

    return int(all(section in text for section in required))


def has_checked_output_format(text: str) -> int:
    if not text:
        return 0

    text = deserialize_prompt_text(text)

    has_not_understood_block = all(
        section in text
        for section in [
            "Check: NOT_UNDERSTOOD",
            "Answer: NA",
            "Postcheck: FAILED",
        ]
    )

    has_understood_block = all(
        section in text
        for section in [
            "Check: UNDERSTOOD",
            "Answer: <A or B or C>",
            "Postcheck: OK",
        ]
    )

    return int(has_not_understood_block and has_understood_block)


def no_invalid_formatting(generated_record: dict) -> int:
    prompt_text = generated_record.get("prompt_text", "")

    checks = [
        bool(prompt_text and prompt_text.strip()),
        has_labels_a_b_c(prompt_text) == 1,
        has_three_options(prompt_text) == 1,
        no_unresolved_placeholders(prompt_text) == 1,
        preserve_required_sections(prompt_text) == 1,
        has_checked_output_format(prompt_text) == 1,
    ]

    return int(all(checks))


def prompt_text_contains_context(generated_record: dict) -> int:
    context = normalize_text(generated_record.get("context", ""))
    prompt_text = normalize_text(generated_record.get("prompt_text", ""))

    if not context or not prompt_text:
        return 0

    return int(context in prompt_text)


def context_has_artifacts(context: str) -> int:
    if not context:
        return 1

    stripped = context.strip()

    artifact_patterns = [
        r"^---",
        r"---$",
        r"^```",
        r"```$",
        r"(?i)^\s*rewritten context\s*:",
        r"(?i)^\s*context\s*:",
        r"(?i)^\s*here is",
    ]

    has_artifact = any(
        re.search(pattern, stripped)
        for pattern in artifact_patterns
    )

    return int(has_artifact)


def semantic_similarity_fallback(original_text: str, mutated_text: str) -> float:
    original_text = deserialize_prompt_text(original_text)
    mutated_text = deserialize_prompt_text(mutated_text)

    return round(SequenceMatcher(None, original_text, mutated_text).ratio(), 4)


class SemanticSimilarityScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

        if SBERT_AVAILABLE:
            self.model = SentenceTransformer(model_name)

    def score(self, original_text: str, mutated_text: str) -> float:
        original_text = deserialize_prompt_text(original_text)
        mutated_text = deserialize_prompt_text(mutated_text)

        if self.model is not None:
            emb1 = self.model.encode(original_text, convert_to_tensor=True)
            emb2 = self.model.encode(mutated_text, convert_to_tensor=True)
            score = float(util.cos_sim(emb1, emb2).item())
            return round(score, 4)

        return semantic_similarity_fallback(original_text, mutated_text)


def evaluate_generated_record(
    baseline_record: dict,
    generated_record: dict,
    sim_scorer: SemanticSimilarityScorer,
) -> dict:
    transformation_name = generated_record.get("transformation_name")

    baseline_context = deserialize_prompt_text(baseline_record.get("context", ""))
    generated_context = deserialize_prompt_text(generated_record.get("context", ""))

    baseline_prompt = deserialize_prompt_text(baseline_record.get("prompt_text", ""))
    generated_prompt = deserialize_prompt_text(generated_record.get("prompt_text", ""))

    errors = []
    warnings = []

    for field in REQUIRED_FIELDS:
        if field not in generated_record:
            errors.append(f"missing_field:{field}")

    for field in INVARIANT_FIELDS:
        if generated_record.get(field) != baseline_record.get(field):
            errors.append(f"changed_invariant_field:{field}")

    if generated_record.get("prompt_type") != transformation_name:
        errors.append("prompt_type_differs_from_transformation_name")

    if not generated_context:
        errors.append("empty_context")

    if not generated_prompt:
        errors.append("empty_prompt_text")

    if prompt_text_contains_context(generated_record) == 0:
        errors.append("prompt_text_does_not_contain_context")

    if context_has_artifacts(generated_context):
        warnings.append("context_has_formatting_artifacts")

    exact_options = exact_preservation_of_answer_options(
        baseline_record=baseline_record,
        generated_record=generated_record,
    )

    prompt_options_ok = prompt_contains_answer_options(generated_record)
    valid_format = no_invalid_formatting(generated_record)

    context_similarity = sim_scorer.score(
        baseline_context,
        generated_context,
    )

    prompt_similarity = sim_scorer.score(
        baseline_prompt,
        generated_prompt,
    )

    result = {
        "example_id": generated_record.get("example_id"),
        "category": generated_record.get("category"),
        "question_polarity": generated_record.get("question_polarity"),
        "prompt_type": generated_record.get("prompt_type"),
        "transformation_name": transformation_name,

        "baseline_context": baseline_context,
        "generated_context": generated_context,
        "baseline_prompt": baseline_prompt,
        "generated_prompt": generated_prompt,

        "context_semantic_similarity_to_baseline": context_similarity,
        "prompt_semantic_similarity_to_baseline": prompt_similarity,

        "exact_preservation_of_answer_options": exact_options,
        "prompt_contains_answer_options": prompt_options_ok,
        "no_invalid_formatting": valid_format,
        "has_labels_A_B_C": has_labels_a_b_c(generated_prompt),
        "has_three_options": has_three_options(generated_prompt),
        "no_unresolved_placeholders": no_unresolved_placeholders(generated_prompt),
        "preserve_required_sections": preserve_required_sections(generated_prompt),
        "has_checked_output_format": has_checked_output_format(generated_prompt),
        "prompt_text_contains_context": prompt_text_contains_context(generated_record),
        "context_has_artifacts": context_has_artifacts(generated_context),

        "errors": errors,
        "warnings": warnings,
    }

    result["is_valid_generated_prompt"] = int(
        len(errors) == 0
        and result["exact_preservation_of_answer_options"] == 1
        and result["prompt_contains_answer_options"] == 1
        and result["no_invalid_formatting"] == 1
        and result["prompt_text_contains_context"] == 1
        and result["context_has_artifacts"] == 0
    )

    return result


def summarize_quality_results(results: list[dict]) -> None:
    total = len(results)

    valid_count = sum(
        r.get("is_valid_generated_prompt", 0)
        for r in results
    )

    error_count = sum(
        1
        for r in results
        if r.get("errors")
    )

    warning_count = sum(
        1
        for r in results
        if r.get("warnings")
    )

    error_counter = Counter()
    warning_counter = Counter()

    for result in results:
        error_counter.update(result.get("errors", []))
        warning_counter.update(result.get("warnings", []))

    print("=== Generated Prompt Quality Summary ===")
    print(f"Total evaluated records: {total}")
    print(f"Valid generated prompts: {valid_count}/{total}")
    print(f"Records with errors: {error_count}")
    print(f"Records with warnings: {warning_count}")
    print()

    if error_counter:
        print("Errors:")
        for key, value in error_counter.most_common():
            print(f"  {key}: {value}")
        print()

    if warning_counter:
        print("Warnings:")
        for key, value in warning_counter.most_common():
            print(f"  {key}: {value}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated prompt quality by comparing a generated prompt file "
            "against the baseline prompt file."
        )
    )

    parser.add_argument(
        "--baseline",
        type=str,
        default="data/prompts/baseline/baseline_prompts.json",
        help="Path to the baseline prompts JSON file.",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the generated prompts JSON file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output JSON file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/quality_checks",
        help=(
            "Base output directory; a transformation-specific subfolder "
            "will be created automatically."
        ),
    )

    parser.add_argument(
        "--similarity_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name for semantic similarity.",
    )

    parser.add_argument(
        "--no-unique-names",
        action="store_true",
        help="If set, do not append _1, _2, ... when output file already exists.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline_records = load_json(args.baseline)
    generated_records = load_json(args.input)

    baseline_index = build_index(baseline_records, "baseline")
    generated_index = build_index(generated_records, "generated")

    missing_generated = sorted(
        set(baseline_index.keys()) - set(generated_index.keys())
    )

    unexpected_generated = sorted(
        set(generated_index.keys()) - set(baseline_index.keys())
    )

    if missing_generated:
        print(f"[WARNING] Missing generated records: {len(missing_generated)}")
        print(f"First missing keys: {missing_generated[:10]}")

    if unexpected_generated:
        print(f"[WARNING] Unexpected generated records: {len(unexpected_generated)}")
        print(f"First unexpected keys: {unexpected_generated[:10]}")

    sim_scorer = SemanticSimilarityScorer(args.similarity_model)

    quality_results = []

    for key, generated_record in generated_index.items():
        if key not in baseline_index:
            continue

        baseline_record = baseline_index[key]

        result = evaluate_generated_record(
            baseline_record=baseline_record,
            generated_record=generated_record,
            sim_scorer=sim_scorer,
        )

        quality_results.append(result)

    transformation_name = infer_transformation_name(quality_results)
    unique_names = not args.no_unique_names

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = build_output_path(
            input_file=args.input,
            output_dir=args.output_dir,
            transformation_name=transformation_name,
            unique=unique_names,
        )

    save_json(quality_results, output_path)

    summarize_quality_results(quality_results)

    print(f"Quality results saved to: {output_path}")

    if not SBERT_AVAILABLE:
        print(
            "sentence-transformers not installed: "
            "using SequenceMatcher fallback for semantic similarity."
        )


if __name__ == "__main__":
    main()