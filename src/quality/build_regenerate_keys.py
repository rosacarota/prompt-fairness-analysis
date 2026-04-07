
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
    base_dir: Path,
    transformation_name: str,
    unique: bool = True
) -> Path:
    output_path = base_dir / transformation_name / f"{transformation_name}_regenerate_keys.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if unique:
        output_path = ensure_unique_path(output_path)

    return output_path


def classify_reason(record: dict) -> str | None:
    invalid = record.get("is_valid_mutant", 1) == 0
    leakage = record.get("suspicious_semantic_leakage", 0) == 1

    if invalid and leakage:
        return "invalid_and_leakage"
    if invalid:
        return "invalid"
    if leakage:
        return "leakage"
    return None


def build_keys(records: list[dict]) -> list[dict]:
    selected = []
    seen = set()

    for r in records:
        reason = classify_reason(r)
        if reason is None:
            continue

        key = (r.get("example_id"), r.get("category"))
        if key in seen:
            continue
        seen.add(key)

        selected.append({
            "example_id": r.get("example_id"),
            "category": r.get("category"),
            "regeneration_reason": reason,
        })

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Build regenerate keys JSON from a mutant quality file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the quality JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/prompts/to_regenerate"),
        help="Base output directory; a transformation-specific subfolder will be created automatically"
    )
    parser.add_argument(
        "--no-unique-names",
        action="store_true",
        help="If set, do not append _1, _2, ... when output file already exists"
    )

    args = parser.parse_args()

    records = load_json(args.input)
    transformation_name = infer_transformation_name(records)
    selected_keys = build_keys(records)

    if args.output is not None:
        output_path = args.output
    else:
        output_path = build_output_path(
            base_dir=args.output_dir,
            transformation_name=transformation_name,
            unique=not args.no_unique_names
        )

    save_json(selected_keys, output_path)

    print(f"Transformation: {transformation_name}")
    print(f"Selected {len(selected_keys)} records for regeneration")
    print(f"Saved keys to: {output_path}")


if __name__ == "__main__":
    main()