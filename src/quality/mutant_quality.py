import json
import argparse
import re
from pathlib import Path
from difflib import SequenceMatcher

try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False


def load_json(json_path: str) -> list:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def infer_transformation_name(records: list[dict], fallback: str = "unknown_transformation") -> str:
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
    unique: bool = True
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

    text = text.replace("\r\n", "\n")

    if "\\n" in text or "\\t" in text or '\\"' in text or "\\'" in text:
        try:
            text = bytes(text, "utf-8").decode("unicode_escape")
        except Exception:
            pass

    text = re.sub(r'\\+\n', '\n', text)
    text = re.sub(r"[ \t]+\n", "\n", text)

    return text


def normalize_text(text: str) -> str:
    text = deserialize_prompt_text(text)
    return " ".join(text.strip().split()) if text else ""


def extract_answer_options(text: str) -> dict:
    if not text:
        return {}

    text = deserialize_prompt_text(text)

    pattern = r'^\s*([ABC])[\.\)]\s*(.+?)(?=^\s*[ABC][\.\)]\s*|\Z)'
    matches = re.findall(pattern, text, flags=re.DOTALL | re.MULTILINE)

    options = {}
    for label, content in matches:
        options[label] = normalize_text(content)

    return options


def exact_preservation_of_answer_options(original_prompt: str, mutated_prompt: str) -> int:
    original_options = extract_answer_options(original_prompt)
    mutated_options = extract_answer_options(mutated_prompt)
    return int(original_options == mutated_options)


def has_question(text: str) -> int:
    text = deserialize_prompt_text(text)
    return int("?" in text) if text else 0


def has_labels_a_b_c(text: str) -> int:
    if not text:
        return 0

    text = deserialize_prompt_text(text)

    has_a = bool(re.search(r'^\s*A[\.\)]', text, flags=re.MULTILINE))
    has_b = bool(re.search(r'^\s*B[\.\)]', text, flags=re.MULTILINE))
    has_c = bool(re.search(r'^\s*C[\.\)]', text, flags=re.MULTILINE))
    return int(has_a and has_b and has_c)


def has_three_options(text: str) -> int:
    options = extract_answer_options(text)
    return int(set(options.keys()) == {"A", "B", "C"})


def no_unresolved_placeholders(text: str) -> int:
    if not text:
        return 0

    text = deserialize_prompt_text(text)

    patterns = [
        r"\{[^{}]+\}",
        r"\[\s*PLACEHOLDER.*?\]",
        r"<\s*placeholder.*?>",
        r"\bTODO\b",
        r"\bTBD\b"
    ]

    found = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
    return int(not found)


def preserve_required_sections(text: str) -> int:
    if not text:
        return 0

    text = deserialize_prompt_text(text)

    required = [
        "Context:",
        "Question:",
        "Options:",
        "Answer with only one letter:"
    ]
    return int(all(section in text for section in required))


def paragraph_count(text: str) -> int:
    if not text:
        return 0

    text = deserialize_prompt_text(text)
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    return len(paragraphs)


def preserve_paragraph_structure(original_prompt: str, mutated_prompt: str) -> int:
    return int(paragraph_count(original_prompt) == paragraph_count(mutated_prompt))


def preserve_line_count(original_prompt: str, mutated_prompt: str) -> int:
    original_prompt = deserialize_prompt_text(original_prompt)
    mutated_prompt = deserialize_prompt_text(mutated_prompt)

    return int(len(original_prompt.splitlines()) == len(mutated_prompt.splitlines()))


def no_invalid_formatting(mutated_prompt: str) -> int:
    checks = [
        bool(mutated_prompt and mutated_prompt.strip()),
        has_question(mutated_prompt) == 1,
        has_labels_a_b_c(mutated_prompt) == 1,
        has_three_options(mutated_prompt) == 1,
        no_unresolved_placeholders(mutated_prompt) == 1,
    ]
    return int(all(checks))


def semantic_similarity_fallback(original_prompt: str, mutated_prompt: str) -> float:
    original_prompt = deserialize_prompt_text(original_prompt)
    mutated_prompt = deserialize_prompt_text(mutated_prompt)
    return round(SequenceMatcher(None, original_prompt, mutated_prompt).ratio(), 4)


class SemanticSimilarityScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

        if SBERT_AVAILABLE:
            self.model = SentenceTransformer(model_name)

    def score(self, original_prompt: str, mutated_prompt: str) -> float:
        original_prompt = deserialize_prompt_text(original_prompt)
        mutated_prompt = deserialize_prompt_text(mutated_prompt)

        if self.model is not None:
            emb1 = self.model.encode(original_prompt, convert_to_tensor=True)
            emb2 = self.model.encode(mutated_prompt, convert_to_tensor=True)
            score = float(util.cos_sim(emb1, emb2).item())
            return round(score, 4)

        return semantic_similarity_fallback(original_prompt, mutated_prompt)


def evaluate_mutant_record(mutant_record: dict, sim_scorer: SemanticSimilarityScorer) -> dict:
    original_prompt = deserialize_prompt_text(mutant_record.get("original_prompt", ""))
    mutated_prompt = deserialize_prompt_text(mutant_record.get("rewritten_prompt", ""))

    if not original_prompt or not mutated_prompt:
        return {
            "example_id": mutant_record.get("example_id"),
            "category": mutant_record.get("category"),
            "source_prompt_type": mutant_record.get("source_prompt_type"),
            "transformation_name": mutant_record.get("transformation_name"),
            "transformation_target": mutant_record.get("transformation_target"),
            "rewriter_model": mutant_record.get("rewriter_model"),
            "original_prompt": original_prompt,
            "rewritten_prompt": mutated_prompt,
            "exact_preservation_of_answer_options": 0,
            "semantic_similarity_to_original": None,
            "no_invalid_formatting": 0,
            "has_question": 0,
            "has_labels_A_B_C": 0,
            "has_three_options": 0,
            "no_unresolved_placeholders": 0,
            "preserve_required_sections": 0,
            "preserve_paragraph_structure": 0,
            "preserve_line_count": 0,
            "is_valid_mutant": 0,
            "error": "Missing original_prompt or rewritten_prompt"
        }

    exact_options = exact_preservation_of_answer_options(original_prompt, mutated_prompt)
    valid_format = no_invalid_formatting(mutated_prompt)
    sim_score = sim_scorer.score(original_prompt, mutated_prompt)
    required_sections = preserve_required_sections(mutated_prompt)
    paragraph_structure = preserve_paragraph_structure(original_prompt, mutated_prompt)
    line_count = preserve_line_count(original_prompt, mutated_prompt)

    result = {
        "example_id": mutant_record.get("example_id"),
        "category": mutant_record.get("category"),
        "source_prompt_type": mutant_record.get("source_prompt_type"),
        "transformation_name": mutant_record.get("transformation_name"),
        "transformation_target": mutant_record.get("transformation_target"),
        "rewriter_model": mutant_record.get("rewriter_model"),
        "original_prompt": original_prompt,
        "rewritten_prompt": mutated_prompt,
        "exact_preservation_of_answer_options": exact_options,
        "semantic_similarity_to_original": sim_score,
        "no_invalid_formatting": valid_format,
        "has_question": has_question(mutated_prompt),
        "has_labels_A_B_C": has_labels_a_b_c(mutated_prompt),
        "has_three_options": has_three_options(mutated_prompt),
        "no_unresolved_placeholders": no_unresolved_placeholders(mutated_prompt),
        "preserve_required_sections": required_sections,
        "preserve_paragraph_structure": paragraph_structure,
        "preserve_line_count": line_count,
    }

    result["is_valid_mutant"] = int(
        result["exact_preservation_of_answer_options"] == 1
        and result["no_invalid_formatting"] == 1
        and result["preserve_required_sections"] == 1
        and result["preserve_paragraph_structure"] == 1
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate mutant prompt quality using original_prompt and rewritten_prompt from the same JSON record"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the mutant prompts JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Explicit output JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/quality_checks",
        help="Base output directory; a transformation-specific subfolder will be created automatically"
    )
    parser.add_argument(
        "--similarity_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name for semantic similarity"
    )
    parser.add_argument(
        "--no-unique-names",
        action="store_true",
        help="If set, do not append _1, _2, ... when output file already exists"
    )

    args = parser.parse_args()

    mutant_records = load_json(args.input)
    sim_scorer = SemanticSimilarityScorer(args.similarity_model)

    quality_results = []

    for mutant_record in mutant_records:
        result = evaluate_mutant_record(mutant_record, sim_scorer)
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
            unique=unique_names
        )

    save_json(quality_results, output_path)

    print(f"Quality results saved to: {output_path}")
    if not SBERT_AVAILABLE:
        print("sentence-transformers not installed: using SequenceMatcher fallback for semantic similarity.")


if __name__ == "__main__":
    main()