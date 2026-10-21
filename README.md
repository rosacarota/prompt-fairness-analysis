# Fifty Shades of Bias: How General-Purpose Prompting Shapes LLM Accuracy and Fairness

> **Replication Package**
>📂 Each main folder includes its own `README.md` with further details.


## Overview

This replication package contains all code, data, and experiment outputs necessary to reproduce the empirical study. The study investigates how general-purpose prompt engineering strategies affect both **accuracy** and **stereotype-related behavior** in instruction-tuned LLMs, evaluated on the [BBQ benchmark](https://huggingface.co/datasets/heegyu/bbq) (Bias Benchmark for QA).


## Repository Structure

```
prompt-fairness-analysis/
├── 📁 data/                    # Datasets, prompts, and meta-prompts
│   ├── 📁 raw/                 # Raw BBQ dataset files
│   ├── 📁 processed/           # Filtered & stratified BBQ datasets
│   ├── 📁 meta-prompting/      # Meta-prompts for LLM-based context rewriting
│   └── 📁 prompts/             # All prompt variants (baseline, fixed, generated)
├── 📁 experiments/             # Inference outputs, evaluations, and QA reports
│   ├── 📁 outputs/             # Raw model responses: {model}/{prompt_type}/run_{id}/
│   ├── 📁 evaluations/         # Evaluation CSV summaries
│   ├── 📁 metrics/             # Prompt structural and readability metrics
│   └── 📁 quality_checks/      # QA reports for generated prompt variants
├── 📁 notebook/                # Jupyter notebooks for RQ1, RQ2, and all analyses
├── 📁 src/                     # All source code (data, prompts, inference, evaluation)
├── 📁 utils/                   # Utility scripts for data conversion and post-processing
├── 📄 README.md
└── 📄 requirements.txt
```

---

## Dataset

We use the **BBQ** (Bias Benchmark for QA) dataset by Parrish et al. (2022), available on [Hugging Face](https://huggingface.co/datasets/heegyu/bbq). BBQ contains 58,492 instances across 11 bias categories: *Age*, *Disability Status*, *Gender Identity*, *Nationality*, *Physical Appearance*, *Race/Ethnicity*, *Religion*, *Socioeconomic Status*, *Sexual Orientation*, and two intersectional categories combining *Gender Identity* with *Race/Ethnicity* and *Socioeconomic Status* with *Race/Ethnicity*.

We focused exclusively on the **29,246 disambiguated instances**, where the context provides sufficient evidence to identify the correct answer, allowing us to assess both task accuracy and stereotype-consistent errors.

### Stratified Sample (594 instances)

From the 29,246 disambiguated instances, we constructed a stratified sample by bias category, question polarity, gold-answer position, and target position, yielding **198 strata**. Three instances were randomly selected without replacement from each stratum, for a total of **594 instances**. The sample is:

- Balanced across 11 BBQ categories (54 instances each)
- Balanced between negative and non-negative questions (297 each)
- Balanced between stereotype-aligned and non-aligned instances (297 each)

Each instance was annotated with *target*, *non\_target*, and *unknown* labels using BBQ metadata and GPT-5.4 Thinking.

## Setup & Installation

### Prerequisites

- Python 3.10+
- API keys for the inference and prompt generation steps:
  - **[OpenRouter](https://openrouter.ai/)** — used for Llama-3.1 8B, Ministral-3 8B, and GPT-5-mini (context rewriting)
  - **[SiliconFlow](https://siliconflow.cn/)** — used for Qwen-3 8B

### Installation

```bash
# Clone the repository
git clone https://github.com/rosacarota/prompt-fairness-analysis.git
cd prompt-fairness-analysis

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.\.venv\Scripts\activate         # Windows (PowerShell)

# Install dependencies
pip install -r requirements.txt

# Download the spaCy English model
python -m spacy download en_core_web_sm
```

### Environment Configuration

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
SILICONFLOW_API_KEY=your_siliconflow_api_key_here
```

> **Note:** API keys are only needed for Stages 2 and 4 (prompt rewriting and inference). All pre-computed artifacts are already included in the repository; the analysis notebooks can be run directly without any API key.




## Analysis Notebooks

Open the notebooks with JupyterLab:

```bash
jupyter lab
```

| Notebook | Research Question | Description |
|---|---|---|
| `rq1_analysis.ipynb` | RQ1 | Main statistical analysis: paired cluster bootstrap-t comparing each strategy vs. baseline |
| `rq2_analysis.ipynb` | RQ2 | Main statistical analysis: GEE models linking prompt characteristics to response-level outcomes |

See [`notebook/README.md`](notebook/README.md) for details on how to run each notebook.

---

## Included Artifacts

All pre-computed artifacts are included; the analysis notebooks can be run without re-executing inference.

| Artifact | Location |
|---|---|
| Stratified 594-instance BBQ sample | `data/processed/` |
| All prompt variants (baseline, fixed, generated) | `data/prompts/` |
| Meta-prompts for context rewriting | `data/meta-prompting/` |
| Quality check reports | `experiments/quality_checks/` |
| Raw inference outputs (all models × strategies × runs) | `experiments/outputs/` |
| Evaluation CSVs (global, per-run, per-category) | `experiments/evaluations/` |
| Prompt structural & readability metrics | `experiments/metrics/` |
