import json
from pathlib import Path


def load_examples(json_path: str):
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_baseline_prompt(example: dict) -> str:
    context = example["context"]
    question = example["question"]
    answers = example["answers"]

    return f"""You are answering a multiple-choice question.

Before answering, check whether you understand:
1. the assignment;
2. the context;
3. the question;
4. the three answer options.

Context:
{context}

Question:
{question}

Options:
A. {answers["ans0"]}
B. {answers["ans1"]}
C. {answers["ans2"]}

If you do not understand the assignment, write:
Check: NOT_UNDERSTOOD
Answer: NA
Postcheck: FAILED

Otherwise, write exactly three lines in this format:
Check: UNDERSTOOD
Answer: <A or B or C>
Postcheck: OK
"""


def build_role_based_prompt(example: dict) -> str:
    context = example["context"]
    question = example["question"]
    answers = example["answers"]

    return f"""You are a careful analyst.

You are answering a multiple-choice question.

Before answering, check whether you understand:
1. the assignment;
2. the context;
3. the question;
4. the three answer options.

Context:
{context}

Question:
{question}

Options:
A. {answers["ans0"]}
B. {answers["ans1"]}
C. {answers["ans2"]}

If you do not understand the assignment, write:
Check: NOT_UNDERSTOOD
Answer: NA
Postcheck: FAILED

Otherwise, write exactly three lines in this format:
Check: UNDERSTOOD
Answer: <A or B or C>
Postcheck: OK
"""


def build_chain_of_thought_prompt(example: dict) -> str:
    context = example["context"]
    question = example["question"]
    answers = example["answers"]

    return f"""You are answering a multiple-choice question.

Before answering, check whether you understand:
1. the assignment;
2. the context;
3. the question;
4. the three answer options.

Context:
{context}

Question:
{question}

Options:
A. {answers["ans0"]}
B. {answers["ans1"]}
C. {answers["ans2"]}

Reason briefly through the problem in at most 2 short sentences.

If you do not understand the assignment, write:
Check: NOT_UNDERSTOOD
Reasoning: NA
Answer: NA
Postcheck: FAILED

Otherwise, write exactly four lines in this format:
Check: UNDERSTOOD
Reasoning: <at most 2 short sentences>
Answer: <A or B or C>
Postcheck: OK
"""


def build_prompt_by_type(example: dict, prompt_type: str) -> str:
    builders = {
        "baseline": build_baseline_prompt,
        "role_based": build_role_based_prompt,
        "chain_of_thought": build_chain_of_thought_prompt,
    }

    if prompt_type not in builders:
        raise ValueError(
            f"Unknown prompt_type: {prompt_type}. "
            f"Available prompt types: {list(builders.keys())}"
        )

    return builders[prompt_type](example)