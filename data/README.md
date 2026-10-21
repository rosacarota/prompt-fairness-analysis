# data/

This folder contains all datasets and prompt files used in the study.

```
data/
├── 📁 raw/                         # Original BBQ dataset files
│   └── 📁 bbq_official/                # One JSONL file per bias category (11 files)
├── 📁 processed/                   # Filtered and stratified BBQ datasets
│   ├── bbq_disambiguated.json          # All 29,246 disambiguated BBQ instances
│   ├── bbq_disambiguated_annotated.json    # Disambiguated instances with target/non_target/unknown labels
│   ├── bbq_disambiguated_annotated.xlsx    # Excel version of the annotated dataset
│   └── bbq_disambiguated_594_sample.json   # Final stratified sample of 594 instances
├── 📁 meta-prompting/              # Meta-prompts used to rewrite contexts via GPT-5 mini
│   ├── attribute_early_metaprompts.json
│   └── attribute_late_metaprompts.json
└── 📁 prompts/                     # All prompt variants used for inference
    ├── 📁 baseline/                    # Zero-shot baseline prompts
    ├── 📁 fixed/                       # Template-based variants (role_based, chain_of_thought)
    └── 📁 generated/                   # LLM-rewritten variants (attribute_early, attribute_late)
        ├── 📁 attribute_early/
        └── 📁 attribute_late/
```

---

## Dataset: BBQ (Bias Benchmark for QA)

We use the [BBQ dataset](https://huggingface.co/datasets/heegyu/bbq) by Parrish et al. (2022). The full dataset contains 58,492 instances across 11 social bias categories. We focus exclusively on **disambiguated** instances (29,246), where the context provides sufficient evidence to determine the correct answer.

Each instance includes:
- A **context** describing a social situation involving two individuals or groups
- A **question** about the individuals or groups
- **Three answer options**: one for each individual/group and one *unknown* option

## Annotations

Each instance in `bbq_disambiguated_annotated.json` carries three additional labels:

| Label | Description |
|---|---|
| `target` | The answer option that identifies the individual or group associated with the stereotype |
| `non_target` | The answer option that identifies the comparison individual or group |
| `unknown` | The answer option indicating insufficient information in the context |

## Stratified Sample (594 instances)

The sample in `bbq_disambiguated_594_sample.json` was constructed by stratifying the 29,246 instances across four dimensions:

- **Bias category** (11 levels)
- **Question polarity** (negative / non-negative)
- **Gold-answer position** (A, B, or C)
- **Target position** (A, B, or C)

This yields **198 strata**. Three instances were drawn without replacement from each stratum, producing 594 instances. The sample is balanced across categories (54 per category), polarities (297 each), and alignment (297 stereotype-aligned, 297 non-aligned).

## Prompts

Each prompt is stored as a JSON record including the original BBQ fields plus the formatted prompt string. The five prompt types are:

| Folder | Prompt type | Construction method |
|---|---|---|
| `baseline/` | `baseline` | Programmatic (Python) |
| `fixed/role_based/` | `role_based` | Programmatic (Python template) |
| `fixed/chain_of_thought/` | `chain_of_thought` | Programmatic (Python template) |
| `generated/attribute_early/` | `attribute_early` | GPT-5-mini rewriting |
| `generated/attribute_late/` | `attribute_late` | GPT-5-mini rewriting |

