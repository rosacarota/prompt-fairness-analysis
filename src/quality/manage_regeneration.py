import json
import argparse
from pathlib import Path
from collections import Counter


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_keys(path: Path):
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Keys file must contain a JSON list.")

    return raw


def record_key(record: dict, key_fields: list[str]):
    return tuple(record.get(field) for field in key_fields)


def normalize_keys(keys_data: list[dict], key_fields: list[str]):
    return {tuple(item.get(field) for field in key_fields) for item in keys_data}


def check_keys(data: list[dict], key_fields: list[str]):
    keys = [record_key(x, key_fields) for x in data]
    counts = Counter(keys)
    duplicates = {k: v for k, v in counts.items() if v > 1}

    print(f"Total records: {len(data)}")
    print(f"Unique keys: {len(counts)}")
    print(f"Duplicated keys: {len(duplicates)}")

    if duplicates:
        print("\nFirst duplicated keys:")
        for k, v in list(duplicates.items())[:20]:
            print(f"{k} -> {v}")


def infer_transformation_name(records: list[dict], fallback: str = "unknown_transformation") -> str:
    """
    Infer transformation_name from records.
    If multiple names are found, return the first one in sorted order.
    """
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
    """
    If path already exists, create path_1, path_2, ...
    Example:
      file.json
      file_1.json
      file_2.json
    """
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


def build_output_in_transformation_folder(
    base_dir: Path,
    transformation_name: str,
    filename: str,
    unique: bool = True
) -> Path:
    """
    Build output path like:
      base_dir / transformation_name / filename

    If unique=True and the file already exists, append _1, _2, ...
    """
    output_path = base_dir / transformation_name / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if unique:
        output_path = ensure_unique_path(output_path)

    return output_path


def extract_records(
    input_file: Path,
    keys_file: Path,
    output_file: Path | None,
    output_dir: Path | None,
    key_fields: list[str],
    unique_names: bool = True
):
    data = load_json(input_file)
    keys_data = load_keys(keys_file)
    target_keys = normalize_keys(keys_data, key_fields)

    subset = [x for x in data if record_key(x, key_fields) in target_keys]

    if not subset:
        print("Warning: no records matched the provided keys.")

    transformation_name = infer_transformation_name(subset or data)

    if output_file is None:
        if output_dir is None:
            raise ValueError("Either output_file or output_dir must be provided.")
        output_file = build_output_in_transformation_folder(
            base_dir=output_dir,
            transformation_name=transformation_name,
            filename=f"{transformation_name}_to_regenerate.json",
            unique=unique_names
        )

    save_json(subset, output_file)
    print(f"Extracted {len(subset)} records to {output_file}")


def merge_records(
    original_file: Path,
    regenerated_file: Path,
    output_file: Path | None,
    output_dir: Path | None,
    key_fields: list[str],
    unique_names: bool = True
):
    original = load_json(original_file)
    regenerated = load_json(regenerated_file)

    regenerated_map = {record_key(x, key_fields): x for x in regenerated}

    new_data = []
    replaced = 0

    for record in original:
        key = record_key(record, key_fields)
        if key in regenerated_map:
            new_data.append(regenerated_map[key])
            replaced += 1
        else:
            new_data.append(record)

    transformation_name = infer_transformation_name(regenerated or original)

    if output_file is None:
        if output_dir is None:
            raise ValueError("Either output_file or output_dir must be provided.")
        output_file = build_output_in_transformation_folder(
            base_dir=output_dir,
            transformation_name=transformation_name,
            filename=f"{transformation_name}_mutants_cleaned.json",
            unique=unique_names
        )

    save_json(new_data, output_file)
    print(f"Replaced {replaced} records")
    print(f"Saved merged dataset to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Reusable tool for extracting and merging selected mutant records."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser.add_argument(
        "--key-fields",
        nargs="+",
        default=["example_id", "category"],
        help="Fields used to identify records uniquely. Default: example_id category"
    )
    parser.add_argument(
        "--no-unique-names",
        action="store_true",
        help="If set, do not append _1, _2, ... when output file already exists"
    )

    # check-keys
    check_parser = subparsers.add_parser(
        "check-keys",
        help="Check uniqueness of selected key fields in a JSON file"
    )
    check_parser.add_argument("--input", type=Path, required=True, help="Input JSON file")

    # extract
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract records to regenerate"
    )
    extract_parser.add_argument("--input", type=Path, required=True, help="Input JSON file")
    extract_parser.add_argument("--keys", type=Path, required=True, help="JSON file containing keys to extract")
    extract_parser.add_argument("--output", type=Path, help="Explicit output JSON file")
    extract_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/prompts/to_regenerate"),
        help="Base output directory; a transformation-specific subfolder will be created automatically"
    )

    # merge
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge regenerated records back into original file"
    )
    merge_parser.add_argument("--original", type=Path, required=True, help="Original JSON file")
    merge_parser.add_argument("--regenerated", type=Path, required=True, help="Regenerated JSON file")
    merge_parser.add_argument("--output", type=Path, help="Explicit merged output JSON file")
    merge_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/prompts/final"),
        help="Base output directory; a transformation-specific subfolder will be created automatically"
    )

    args = parser.parse_args()
    unique_names = not args.no_unique_names

    if args.command == "check-keys":
        data = load_json(args.input)
        check_keys(data, args.key_fields)

    elif args.command == "extract":
        extract_records(
            input_file=args.input,
            keys_file=args.keys,
            output_file=args.output,
            output_dir=args.output_dir,
            key_fields=args.key_fields,
            unique_names=unique_names
        )

    elif args.command == "merge":
        merge_records(
            original_file=args.original,
            regenerated_file=args.regenerated,
            output_file=args.output,
            output_dir=args.output_dir,
            key_fields=args.key_fields,
            unique_names=unique_names
        )


if __name__ == "__main__":
    main()