# data/processed/

This directory contains processed versions of the BBQ dataset used in the study.

## Files

### `bbq_disambiguated.json`
**Description:** All 29,246 disambiguated BBQ instances after filtering from the full 58,492-instance dataset.

---

### `bbq_disambiguated_annotated.json`
**Description:** Disambiguated instances enriched with three additional labels per instance: `target`, `non_target`, and `unknown`. Labels were assigned using BBQ metadata and GPT-4.1 Thinking.  


### `bbq_disambiguated_annotated.xlsx`
**Description:** Excel version of `bbq_disambiguated_annotated.json` for manual inspection.  
**Purpose:** Human-readable format for reviewing annotations.

---

### `bbq_disambiguated_594_sample.json`
**Description:** Final stratified sample of **594 instances** drawn from the 29,246 disambiguated instances.  
