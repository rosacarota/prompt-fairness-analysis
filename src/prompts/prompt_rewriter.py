import json
from pathlib import Path


TRANSFORMATION_TARGETS = {
    "attribute_early": (
        "Rewrite the prompt so that mentions of sensitive attributes "
        "appear as early as possible in the prompt."
    ),
}


def load_prompts(json_path: str) -> list:
    """
    Load prompt records from a JSON file.
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_prompts(records: list, output_path: str) -> None:
    """
    Save prompt records to a JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def build_transformation_metaprompt(original_prompt: str, transformation_target: str) -> str:
    """
    Build a generic meta-prompt for controlled prompt rewriting.
    """
    return f"""You are assisting in a controlled prompt transformation task for an experimental study.

Your goal is to rewrite the input prompt by modifying only the requested prompt component, while preserving the original task semantics and the correct answer.

Original prompt:
---
{original_prompt}
---

Transformation target:
{transformation_target}

Constraints:
- Preserve the task semantics.
- Preserve the answer options exactly.
- Do not change the correct answer.
- Do not introduce new facts.
- Do not introduce new information that is not present in the original prompt.
- Keep the prompt natural and grammatically correct.
- Preserve the original prompt formatting structure.
- Return the rewritten prompt in the same serialized text format as the original prompt.
- Preserve line breaks exactly as in the original prompt.
- Preserve the paragraph structure and section layout of the original prompt.
- Do not explain your changes.
- Do not add comments, notes, introductions, headings, or quotation marks.
- Do not output phrases such as "Here is the rewritten prompt" or similar.

Output format:
Return only the rewritten prompt text, formatted exactly like the original prompt representation. If your output contains anything other than the rewritten prompt, it is incorrect."""

def build_rewriter_record(prompt_record: dict, transformation_name: str) -> dict:
    """
    Build a record containing the original prompt and the corresponding meta-prompt.
    """
    if transformation_name not in TRANSFORMATION_TARGETS:
        raise ValueError(f"Unknown transformation: {transformation_name}")

    original_prompt = prompt_record["prompt_text"]
    transformation_target = TRANSFORMATION_TARGETS[transformation_name]
    meta_prompt = build_transformation_metaprompt(original_prompt, transformation_target)

    return {
    "example_id": prompt_record["example_id"],
    "category": prompt_record["category"],
    "source_prompt_type": prompt_record["prompt_type"],
    "transformation_name": transformation_name,
    "transformation_target": transformation_target,
    "original_prompt": original_prompt,
    "meta_prompt": meta_prompt,
    "gold_label": prompt_record["gold_label"],
    "gold_answer": prompt_record["gold_answer"],
    "stereotyped_groups": prompt_record.get("stereotyped_groups", [])
}