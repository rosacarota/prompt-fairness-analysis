from pathlib import Path
import json
from datasets import load_dataset, get_dataset_config_names, concatenate_datasets

DATASET_NAME = "heegyu/bbq"

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_bbq_subsets(dataset_name: str = DATASET_NAME):
    """
    Load all available BBQ subsets/configurations from Hugging Face
    and merge them into a single dataset.
    """
    config_names = get_dataset_config_names(dataset_name, trust_remote_code=True)

    datasets_list = []
    for config in config_names:

        ds = load_dataset(dataset_name, config, split="test", trust_remote_code=True)

        ds = ds.add_column("subset_name", [config] * len(ds))

        datasets_list.append(ds)

    full_dataset = concatenate_datasets(datasets_list)
    return full_dataset


def filter_disambiguated(dataset):
    """
    Keep only disambiguated examples.
    """
    return dataset.filter(lambda x: x["context_condition"] == "disambig")


def simplify_record(example):
    """
    Convert each raw BBQ example into a cleaner format
    that is easier to use in later experimental steps.

    The returned dictionary keeps only the fields that are useful
    for prompt generation and evaluation.
    """
    label = int(example["label"])

    # Store the three answer options in a dictionary
    answers = {
        "ans0": example["ans0"],
        "ans1": example["ans1"],
        "ans2": example["ans2"],
    }

    # Retrieve the correct answer text using the label
    gold_answer_text = answers[f"ans{label}"]

    # Optional metadata fields from the original dataset
    answer_info = example.get("answer_info", {})
    additional_metadata = example.get("additional_metadata", {})
    stereotyped_groups = additional_metadata.get("stereotyped_groups", [])

    return {
        "example_id": example["example_id"],
        "category": example["category"],
        "question_polarity": example["question_polarity"],
        "context": example["context"],
        "question": example["question"],
        "answers": answers,
        "label": label,
        "gold_answer": gold_answer_text,
        "answer_info": answer_info,
        "stereotyped_groups": stereotyped_groups
    }


def save_jsonl(records, output_path: Path):
    """
    Save the processed records in JSONL format.
    """
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(records, output_path: Path):
    """
    Save the processed records in JSONL format.
    """
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main():
    print("Loading all BBQ subsets from Hugging Face...")
    full_dataset = load_all_bbq_subsets()

    print(f"Total examples loaded: {len(full_dataset)}")

    print("Filtering disambiguated examples...")
    disambig_dataset = filter_disambiguated(full_dataset)

    print(f"Number of disambiguated examples: {len(disambig_dataset)}")

    print("Simplifying records...")
    processed_records = [simplify_record(x) for x in disambig_dataset]

    jsonl_path = OUTPUT_DIR / "bbq_disambiguated.jsonl"
    json_path = OUTPUT_DIR / "bbq_disambiguated.json"

    print("Saving processed dataset...")
    save_jsonl(processed_records, jsonl_path)
    save_json(processed_records, json_path)

    print(f"JSONL file saved to: {jsonl_path}")
    print(f"JSON file saved to: {json_path}")


if __name__ == "__main__":
    main()