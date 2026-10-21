# src/

This folder contains all source code for data preparation, prompt generation, quality control, inference, and evaluation.

```
src/
├── 📁 data/          # Dataset download, filtering, and stratified sampling
├── 📁 prompts/       # Prompt construction and LLM-based rewriting
├── 📁 quality/       # Quality assurance for generated prompt variants
├── 📁 inference/     # Model inference
└── 📁 evaluation/    # Metric computation and aggregation
```

---

## data/

| Script | Description |
|---|---|
| `data_filter.py` | Downloads the BBQ dataset from Hugging Face and extracts all disambiguated instances into `data/processed/bbq_disambiguated.json`.|
| `sample_stratified_dataset.py` | Constructs the stratified 594-instance sample across 198 strata (bias category × question polarity × gold-answer position × target position). Outputs `data/processed/bbq_disambiguated_594_sample.json`. |

---

## prompts/

| Script | Description |
|---|---|
| `prompt_builder.py` | Core module defining prompt templates and formatting logic for all five strategies. |
| `prompt_rewriter.py` | Constructs meta-prompts and handles context cleaning before rewriting. |
| `generate_baseline_prompts.py` | Generates zero-shot baseline prompts for all 594 instances. |
| `generate_fixed_mutants.py` | Generates `role_based` and `chain_of_thought` prompt variants using Python templates. |
| `generate_metaprompts.py` | Generates meta-prompts for GPT-5-mini context rewriting. |
| `generate_mutants.py` | Calls the OpenRouter API to generate LLM-rewritten context variants. |

---

## quality/

| Script | Description |
|---|---|
| `mutant_quality.py` | Validates LLM-rewritten prompt variants against the baseline. Checks that (1) the question and answer options are preserved, (2) answer labels are intact, and (3) semantic similarity (cosine similarity via `all-MiniLM-L6-v2`) is ≥ 0.90. Outputs a QA report to `experiments/quality_checks/`. |
---

## inference/

| Script | Description |
|---|---|
| `model_runner.py` | Multi-run inference engine. Routes requests to **OpenRouter** (Llama-3.1 8B, Ministral-3 8B) or **SiliconFlow** (Qwen-3 8B) depending on the model alias. Handles automatic retries, incremental saves (`--save-every`), and automated run ID tracking (`run_01`, `run_02`, …). Requires `OPENROUTER_API_KEY` and `SILICONFLOW_API_KEY` in `.env`. |

Model aliases: `llama31_8b`, `qwen3_8b`, `ministral3_8b`

---

## evaluation/

| Script | Description |
|---|---|
| `compute_output_metrics.py` | Parses raw inference outputs and computes per-run metrics (accuracy, unknown ratio, sDIS, BER_nonaligned, valid_ratio). Outputs `experiments/evaluations/output_metrics_by_run.csv`. |
| `summarize_global_metrics.py` | Aggregates per-run metrics across runs for each model–strategy combination (mean, std, min, max). Outputs `experiments/evaluations/global_metrics_summary.csv`. |
| `summarize_category_metrics.py` | Computes and aggregates metrics broken down by BBQ bias category. Outputs `category_metrics_by_run.csv` and `category_metrics_summary.csv`. |
| `export_invalid_outputs.py` | Extracts all responses classified as invalid for manual inspection. Outputs `experiments/evaluations/invalid_outputs.csv`. |
| `prompt_metrics.py` | Computes linguistic and structural measures (token count, readability, POS counts, etc.) for all prompt variants using spaCy and textstat |
