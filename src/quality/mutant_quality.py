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


def save_json(data: list, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_output_path(mutants_file: str, output_dir: str = "experiments/quality_checks") -> str:
    input_stem = Path(mutants_file).stem
    return str(Path(output_dir) / f"{input_stem}_quality.json")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split()) if text else ""


def extract_answer_options(text: str) -> dict:
    """
    Extract answer options from patterns like:
    A. ...
    B. ...
    C. ...

    or:
    A) ...
    B) ...
    C) ...
    """
    if not text:
        return {}

    text = text.replace("\r\n", "\n")

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
    return int("?" in text) if text else 0


def has_labels_a_b_c(text: str) -> int:
    if not text:
        return 0

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

    patterns = [
        r"\{[^{}]+\}",
        r"\[\s*PLACEHOLDER.*?\]",
        r"<\s*placeholder.*?>",
        r"\bTODO\b",
        r"\bTBD\b"
    ]

    found = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
    return int(not found)


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
    """
    Fallback similarity if sentence-transformers is not installed.
    Less semantically robust, but usable as backup.
    """
    return round(SequenceMatcher(None, original_prompt, mutated_prompt).ratio(), 4)


class SemanticSimilarityScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

        if SBERT_AVAILABLE:
            self.model = SentenceTransformer(model_name)

    def score(self, original_prompt: str, mutated_prompt: str) -> float:
        if self.model is not None:
            emb1 = self.model.encode(original_prompt, convert_to_tensor=True)
            emb2 = self.model.encode(mutated_prompt, convert_to_tensor=True)
            score = float(util.cos_sim(emb1, emb2).item())
            return round(score, 4)

        return semantic_similarity_fallback(original_prompt, mutated_prompt)


def evaluate_mutant_record(mutant_record: dict, sim_scorer: SemanticSimilarityScorer) -> dict:
    original_prompt = mutant_record.get("original_prompt", "")
    mutated_prompt = mutant_record.get("rewritten_prompt", "")

    if not original_prompt or not mutated_prompt:
        return {
            "example_id": mutant_record.get("example_id"),
            "category": mutant_record.get("category"),
            "transformation_name": mutant_record.get("transformation_name"),
            "rewriter_model": mutant_record.get("rewriter_model"),
            "exact_preservation_of_answer_options": 0,
            "semantic_similarity_to_original": None,
            "no_invalid_formatting": 0,
            "has_question": 0,
            "has_labels_A_B_C": 0,
            "has_three_options": 0,
            "no_unresolved_placeholders": 0,
            "is_valid_mutant": 0,
            "error": "Missing original_prompt or rewritten_prompt"
        }

    exact_options = exact_preservation_of_answer_options(original_prompt, mutated_prompt)
    valid_format = no_invalid_formatting(mutated_prompt)
    sim_score = sim_scorer.score(original_prompt, mutated_prompt)

    result = {
        "example_id": mutant_record.get("example_id"),
        "category": mutant_record.get("category"),
        "source_prompt_type": mutant_record.get("source_prompt_type"),
        "transformation_name": mutant_record.get("transformation_name"),
        "transformation_target": mutant_record.get("transformation_target"),
        "rewriter_model": mutant_record.get("rewriter_model"),
        "exact_preservation_of_answer_options": exact_options,
        "semantic_similarity_to_original": sim_score,
        "no_invalid_formatting": valid_format,
        "has_question": has_question(mutated_prompt),
        "has_labels_A_B_C": has_labels_a_b_c(mutated_prompt),
        "has_three_options": has_three_options(mutated_prompt),
        "no_unresolved_placeholders": no_unresolved_placeholders(mutated_prompt),
    }

    result["is_valid_mutant"] = int(
        result["exact_preservation_of_answer_options"] == 1
        and result["no_invalid_formatting"] == 1
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
        default="experiments/quality_checks",
        help="Directory where quality results will be saved"
    )
    parser.add_argument(
        "--similarity_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name for semantic similarity"
    )

    args = parser.parse_args()

    mutant_records = load_json(args.input)
    sim_scorer = SemanticSimilarityScorer(args.similarity_model)

    quality_results = []

    for mutant_record in mutant_records:
        result = evaluate_mutant_record(mutant_record, sim_scorer)
        quality_results.append(result)

    output_path = build_output_path(args.input, args.output)
    save_json(quality_results, output_path)

    print(f"Quality results saved to: {output_path}")
    if not SBERT_AVAILABLE:
        print("sentence-transformers not installed: using SequenceMatcher fallback for semantic similarity.")


if __name__ == "__main__":
    main()