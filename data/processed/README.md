# Processed Data

This directory contains processed versions of the BBQ (Bias in Bias Questionnaires) dataset used for fairness analysis.

## Files Overview

### `bbq_disambiguated.json`
**Description:** Complete processed dataset with all BBQ examples after disambiguation and processing.
**Size:** Full dataset
**Purpose:** Base dataset for analysis and model evaluation
**Use Case:** Comprehensive fairness analysis across all categories and demographic groups

### `bbq_disambiguated_sample_380.json`
**Description:** A stratified sample of 380 examples from the full disambiguated dataset.
**Purpose:** Smaller, manageable dataset for focused experiments and testing
**Use Case:** Quick experimentation, prompt generation, and initial model evaluation

### `bbq_disambiguated_sample_380_curated.json`
**Description:** Curated version of the 380-sample dataset with quality validation and filtering.
**Filtering Applied:** 
- Removed low-quality or problematic examples
- Validated structural integrity of questions and answers
- Ensured proper alignment between examples and stereotyped groups
**Purpose:** High-quality subset for rigorous fairness testing
**Use Case:** Primary dataset for experiments, model evaluation, and fairness metrics computation

### `bbq_disambiguated_sample_380_unmatched_stereotyped_groups.json`
**Description:** Examples from the 380-sample with unmatched or misaligned stereotyped groups.
**Contents:** Examples where:
- Stereotyped groups don't match available answer options
- "Unknown" or "Not answerable" responses appear
- Structural inconsistencies exist
**Purpose:** Quality control and error tracking
**Use Case:** Debugging, data quality assessment, and identifying problematic examples for re-processing
