import json
from pathlib import Path


def load_examples(json_path: str):
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_baseline_prompt(example: dict) -> str:
    """
    Build a simple baseline multiple-choice prompt for a BBQ example.
    The model is instructed to answer with only A, B, or C.
    """
    context = example["context"]
    question = example["question"]
    answers = example["answers"]

    prompt = f"""You are answering a multiple-choice question.

Context: {context}

Question: {question}

Options:
A. {answers["ans0"]}
B. {answers["ans1"]}
C. {answers["ans2"]}

Answer with only one letter: A, B, or C.
"""
    return prompt