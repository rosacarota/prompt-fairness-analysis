import argparse
import json
import shutil
import re
from pathlib import Path
from typing import Any


DEFAULT_ROOT_DIR = Path("data/prompts")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_question_mark(question: str) -> tuple[str, bool]:
    """
    Add a final question mark if missing.

    Returns:
    - updated question
    - whether it changed
    """
    if question is None:
        return "", False

    original = str(question)
    stripped = original.strip()

    if not stripped:
        return original, False

    if stripped.endswith("?"):
        return original, False

    # If the question ends with a period, replace it with '?'.
    if stripped.endswith("."):
        updated_stripped = stripped[:-1].rstrip() + "?"
    else:
        updated_stripped = stripped + "?"

    # Preserve leading indentation if any, but normalize trailing whitespace.
    leading_spaces = original[: len(original) - len(original.lstrip())]
    updated = leading_spaces + updated_stripped

    return updated, updated != original


def fix_question_field(record: dict) -> bool:
    """
    Fix the top-level 'question' field if present.
    """
    if "question" not in record:
        return False

    updated_question, changed = ensure_question_mark(record["question"])

    if changed:
        record["question"] = updated_question

    return changed


def fix_prompt_question_section(record: dict) -> bool:
    """
    Fix the Question section inside prompt_text.

    Expected prompt structure:

    Question:
    <question text>

    Options:
    ...
    """
    prompt_text = record.get("prompt_text")

    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return False

    pattern = r"(Question:\n)(.*?)(\n\nOptions:)"

    match = re.search(pattern, prompt_text, flags=re.DOTALL)

    if not match:
        return False

    prefix = match.group(1)
    question_text = match.group(2)
    suffix = match.group(3)

    updated_question_text, changed = ensure_question_mark(question_text)

    if not changed:
        return False

    updated_prompt_text = (
        prompt_text[: match.start()]
        + prefix
        + updated_question_text
        + suffix
        + prompt_text[match.end() :]
    )

    record["prompt_text"] = updated_prompt_text

    return True


def process_record(record: dict) -> tuple[bool, bool]:
    """
    Returns:
    - question_field_changed
    - prompt_text_changed
    """
    question_changed = fix_question_field(record)
    prompt_changed = fix_prompt_question_section(record)

    return question_changed, prompt_changed


def process_json_file(path: Path, dry_run: bool, backup: bool) -> dict:
    data = load_json(path)

    if not isinstance(data, list):
        return {
            "file": str(path),
            "skipped": True,
            "reason": "JSON root is not a list",
            "records": 0,
            "question_field_fixes": 0,
            "prompt_text_fixes": 0,
            "modified": False,
        }

    question_field_fixes = 0
    prompt_text_fixes = 0

    for record in data:
        if not isinstance(record, dict):
            continue

        question_changed, prompt_changed = process_record(record)

        if question_changed:
            question_field_fixes += 1

        if prompt_changed:
            prompt_text_fixes += 1

    modified = question_field_fixes > 0 or prompt_text_fixes > 0

    if modified and not dry_run:
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)

        save_json(data, path)

    return {
        "file": str(path),
        "skipped": False,
        "reason": "",
        "records": len(data),
        "question_field_fixes": question_field_fixes,
        "prompt_text_fixes": prompt_text_fixes,
        "modified": modified,
    }


def find_json_files(root_dir: Path) -> list[Path]:
    return sorted(root_dir.rglob("*.json"))


def print_file_result(result: dict) -> None:
    if result["skipped"]:
        print(f"[SKIPPED] {result['file']} - {result['reason']}")
        return

    if result["modified"]:
        print(
            f"[FIXED] {result['file']} | "
            f"records={result['records']} | "
            f"question={result['question_field_fixes']} | "
            f"prompt_text={result['prompt_text_fixes']}"
        )
    else:
        print(
            f"[OK] {result['file']} | "
            f"records={result['records']} | no missing question marks"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively scan prompt JSON files and add missing question marks "
            "to both the 'question' field and the Question section inside 'prompt_text'."
        )
    )

    parser.add_argument(
        "--root-dir",
        type=Path,
        default=DEFAULT_ROOT_DIR,
        help="Root directory containing prompt JSON files.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report changes without writing files.",
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a .bak backup before modifying each changed file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root_dir = args.root_dir

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    json_files = find_json_files(root_dir)

    print("=== Question Mark Fixer ===")
    print(f"Root directory: {root_dir}")
    print(f"JSON files found: {len(json_files)}")
    print(f"Dry run: {args.dry_run}")
    print(f"Backup enabled: {args.backup}")
    print()

    total_records = 0
    total_question_field_fixes = 0
    total_prompt_text_fixes = 0
    modified_files = 0
    skipped_files = 0

    for path in json_files:
        try:
            result = process_json_file(
                path=path,
                dry_run=args.dry_run,
                backup=args.backup,
            )
        except Exception as e:
            print(f"[ERROR] {path} - {e}")
            continue

        print_file_result(result)

        if result["skipped"]:
            skipped_files += 1
            continue

        total_records += result["records"]
        total_question_field_fixes += result["question_field_fixes"]
        total_prompt_text_fixes += result["prompt_text_fixes"]

        if result["modified"]:
            modified_files += 1

    print()
    print("=== Summary ===")
    print(f"Files scanned: {len(json_files)}")
    print(f"Files modified: {modified_files}")
    print(f"Files skipped: {skipped_files}")
    print(f"Records scanned: {total_records}")
    print(f"Question field fixes: {total_question_field_fixes}")
    print(f"Prompt text fixes: {total_prompt_text_fixes}")

    if args.dry_run:
        print()
        print("Dry run only: no files were modified.")


if __name__ == "__main__":
    main()