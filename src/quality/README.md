# Quality Control Scripts

This directory contains scripts for managing and validating the quality of prompt mutants and their regeneration. These scripts handle mutation validation, cleanup, duplicate detection, and candidate identification for fairness analysis datasets.

## Scripts Overview

### `mutant_quality.py`
**Purpose:** Validate and assess the quality of mutant prompts against baseline criteria.

**Key Functions:**
- Extracts and normalizes answer options (A, B, C) from prompts
- Validates structural integrity of mutants:
  - Presence of questions (`?` mark)
  - Presence of labels A, B, C
  - Exact preservation of answer options between original and mutated prompts
  - Presence of exactly three options
- Detects and handles escaped characters in prompt text
- Generates quality assessment reports

**Inputs:**
- Input mutants JSON file (path specified via CLI argument)
- Baseline or original prompts for comparison

**Outputs:**
- Quality JSON file containing validation results and metadata
- Output path: `experiments/quality_checks/{transformation_name}/{input_stem}_quality.json`

**Usage:**
```bash
python src/quality/mutant_quality.py --input <mutants_file> --output <output_file>
```

**Key Features:**
- Handles normalized text deserialization (real newlines vs escaped `\n`)
- Extracts answer options using regex patterns
- Compares structural elements between original and mutated prompts

---

### `clean_mutant_wrappers.py`
**Purpose:** Remove formatting artifacts (leading/trailing `---` markers) from rewritten prompt fields.

**Key Functions:**
- Detects and strips leading/trailing `---` wrapper lines
- Preserves legitimate prompt content
- Tracks number of records modified

**Inputs:**
- Mutants JSON file with potentially wrapped rewritten_prompt fields

**Outputs:**
- Cleaned JSON file without formatting wrappers
- Progress report showing number of modified records

**Usage:**
```bash
python src/quality/clean_mutant_wrappers.py --input <input_file> --output <output_file>
```

**Example Transformation:**
```
BEFORE:
"---
This is the mutated prompt content.
---"

AFTER:
"This is the mutated prompt content."
```

---

### `build_regenerate_keys.py`
**Purpose:** Extract records marked as invalid and create a regeneration key file for batch reprocessing.

**Key Functions:**
- Identifies invalid mutants (where `is_valid_mutant = 0`)
- Builds unique keys from example_id and category to avoid duplicates
- Generates transformation-specific regeneration task lists

**Inputs:**
- Quality JSON file with validation results

**Outputs:**
- Regeneration keys JSON file containing:
  - example_id
  - category
  - regeneration_reason (e.g., "invalid")
- Output structure: `data/prompts/to_regenerate/{transformation_name}/{transformation_name}_regenerate_keys.json`

**Usage:**
```bash
python src/quality/build_regenerate_keys.py --input <quality_file> --output-dir data/prompts/to_regenerate

# Or with explicit output path:
python src/quality/build_regenerate_keys.py --input <quality_file> --output <explicit_output_path>
```

**Options:**
- `--no-unique-names`: Skip appending `_1, _2, ...` to existing files

---

### `manage_regeneration.py`
**Purpose:** Manage the regeneration workflow by extracting and merging records during iterative refinement cycles.

**Key Functions:**
- **Extract records:** Filter data using regeneration keys and save targeted subsets
  - Matches records using key fields (e.g., example_id, category)
  - Creates transformation-specific output folders
  
- **Merge records:** Combine original data with updated/regenerated records
  - Updates existing records based on key matches
  - Preserves non-matching original records
  - Handles duplicate resolution

**Inputs:**
- Original mutants/data file
- Regeneration keys file (output from `build_regenerate_keys.py`)
- Updated/regenerated records file

**Outputs:**
- Extracted subset JSON (for targeted regeneration)
- Merged JSON file (combining original + regenerated records)

**Usage:**
```bash
# Extract records to regenerate
python src/quality/manage_regeneration.py extract-records \
  --input <data_file> \
  --keys <regeneration_keys_file> \
  --output-dir data/prompts/regenerate_workspace

# Merge regenerated records back
python src/quality/manage_regeneration.py merge-records \
  --original <original_file> \
  --updated <regenerated_file> \
  --output <merged_output_file>
```

**Key Features:**
- Transformation name inference from records
- Automatic folder creation based on transformation type
- Duplicate key handling via unique file naming
- Preserves original records not targeted for regeneration

---
