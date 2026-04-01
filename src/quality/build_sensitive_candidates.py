import json
import csv
import re
from pathlib import Path
from collections import Counter, defaultdict

# =========================
# CONFIG
# =========================
INPUT_FILE = "data/prompts/baseline/baseline_prompts.json"
LEXICON_FILE = "data/lexicons/dictionaries/sensitive_lexicon_by_subset_2.json"
OUTPUT_FILE = "data/lexicons/candidate_runs/baseline_candidates.csv"

MIN_FREQ = 2
TOP_K_PER_SUBSET = 200

STOPWORDS = {
    "a", "an", "the", "of", "to", "and", "or", "in", "on", "at", "for",
    "with", "from", "by", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "he", "she", "they", "it",
    "his", "her", "their", "who", "what", "when", "where", "why", "how",
    "as", "but", "if", "than", "then", "because", "while", "into", "about",
    "person", "people", "someone", "somebody"
}

GENERIC_TOKENS = {
    "said", "asked", "told", "room", "person", "people", "someone", "somebody",
    "thing", "things", "group", "groups", "one", "two"
}

COMBINED_SUBSETS = {
    "Race_x_SES": ["Race_ethnicity", "SES"],
    "Race_x_gender": ["Race_ethnicity", "Gender_identity"]
}

# =========================
# HELPERS
# =========================
def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    return normalize_text(text).split()

def ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_records(path):
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = load_json(path)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
        raise ValueError("Formato JSON non riconosciuto.")
    elif p.suffix.lower() == ".jsonl":
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    elif p.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    else:
        raise ValueError("Formato input non supportato.")

def detect_field(record, candidates):
    for c in candidates:
        if c in record:
            return c
    return None

def get_active_terms(subset, lexicon):
    if subset in COMBINED_SUBSETS:
        terms = []
        for base in COMBINED_SUBSETS[subset]:
            terms.extend(lexicon.get(base, []))
        return set(normalize_text(t) for t in terms)
    return set(normalize_text(t) for t in lexicon.get(subset, []))

def get_active_regex_patterns(subset, lexicon):
    patterns = []

    if subset in COMBINED_SUBSETS:
        for base in COMBINED_SUBSETS[subset]:
            patterns.extend(lexicon.get(f"{base}_regex", []))
    else:
        patterns.extend(lexicon.get(f"{subset}_regex", []))

    return patterns

def get_seed_tokens(active_terms):
    toks = set()
    for term in active_terms:
        for tok in term.split():
            if tok not in STOPWORDS and len(tok) > 2:
                toks.add(tok)
    return toks

def strip_regex_matches(text, regex_patterns):
    cleaned = text
    for pattern in regex_patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def is_candidate_relevant(ng, active_terms, seed_tokens):
    if ng in active_terms:
        return False

    parts = ng.split()

    if any(p in STOPWORDS for p in parts):
        return False

    if any(len(p) == 1 and p.isalpha() for p in parts):
        return False

    if all(p in GENERIC_TOKENS for p in parts):
        return False

    if all(len(p) <= 2 for p in parts):
        return False

    # Tieni solo candidati che condividono almeno un token con il seed
    if not any(p in seed_tokens for p in parts):
        return False

    return True

def get_next_output_path(base_output_file: str) -> Path:
    """
    Se il file base non esiste, usa quello.
    Se esiste già, crea un nuovo file con suffisso progressivo:
    esempio:
      baseline_candidates.csv
      baseline_candidates_1.csv
      baseline_candidates_2.csv
      ...
    """
    base_path = Path(base_output_file)
    parent = base_path.parent
    stem = base_path.stem
    suffix = base_path.suffix

    parent.mkdir(parents=True, exist_ok=True)

    if not base_path.exists():
        return base_path

    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

# =========================
# MAIN
# =========================
def main():
    lexicon = load_json(LEXICON_FILE)
    records = load_records(INPUT_FILE)

    if not records:
        raise ValueError("Nessun record trovato.")

    sample = records[0]
    subset_field = detect_field(sample, ["subset", "category", "bias_type"])
    text_field = detect_field(sample, ["prompt_text", "prompt", "text", "input"])

    if subset_field is None or text_field is None:
        raise ValueError("Non trovo i campi subset/text nel file input.")

    counters = defaultdict(Counter)
    examples = defaultdict(dict)

    for rec in records:
        subset = rec.get(subset_field, "")
        text = str(rec.get(text_field, ""))
        if not subset or not text:
            continue

        active_terms = get_active_terms(subset, lexicon)
        active_regex_patterns = get_active_regex_patterns(subset, lexicon)
        seed_tokens = get_seed_tokens(active_terms)

        # Rimuove dal testo i pattern già coperti via regex
        cleaned_text = strip_regex_matches(text, active_regex_patterns)
        tokens = tokenize(cleaned_text)

        for n in [1, 2, 3]:
            for ng in ngrams(tokens, n):
                if not is_candidate_relevant(ng, active_terms, seed_tokens):
                    continue
                counters[subset][ng] += 1
                if ng not in examples[subset]:
                    examples[subset][ng] = text[:300]

    final_output_path = get_next_output_path(OUTPUT_FILE)

    with open(final_output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "candidate_term", "frequency", "example_text"])

        for subset, counter in counters.items():
            candidates = [
                (term, freq) for term, freq in counter.items()
                if freq >= MIN_FREQ
            ]
            candidates.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))

            for term, freq in candidates[:TOP_K_PER_SUBSET]:
                writer.writerow([subset, term, freq, examples[subset].get(term, "")])

    print(f"Salvato: {final_output_path}")

if __name__ == "__main__":
    main()