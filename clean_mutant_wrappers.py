import json
import argparse
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_rewritten_prompt(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    if text.startswith("---"):
        text = text[3:].lstrip()

    if text.endswith("---"):
        text = text[:-3].rstrip()

    return text


def main():
    parser = argparse.ArgumentParser(
        description="Remove leading/trailing --- wrappers from rewritten_prompt fields."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input mutants JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output cleaned JSON")

    args = parser.parse_args()

    records = load_json(args.input)
    cleaned = []
    changed = 0

    for record in records:
        new_record = dict(record)

        original = record.get("rewritten_prompt", "")
        updated = clean_rewritten_prompt(original)

        if updated != original:
            changed += 1

        new_record["rewritten_prompt"] = updated
        cleaned.append(new_record)

    save_json(cleaned, args.output)

    print(f"Processed records: {len(cleaned)}")
    print(f"Modified rewritten_prompt fields: {changed}")
    print(f"Saved cleaned file to: {args.output}")


if __name__ == "__main__":
    main()