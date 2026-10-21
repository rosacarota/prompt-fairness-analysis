# notebook/

This folder contains all Jupyter notebooks used for data analysis and visualization. The notebooks address the two research questions of the paper.

Open all notebooks with:

```bash
# From the project root
jupyter lab
```

---

## RQ1 — Effects of Prompting Strategies

| Notebook | Description |
|---|---|
| `rq1_analysis.ipynb` | **Main RQ1 analysis.** Paired, category-stratified cluster bootstrap-t comparing each prompting strategy against the zero-shot baseline, separately for each LLM. Reports observed differences, 95% confidence intervals, and Holm-adjusted p-values (within-family and global). |

## RQ2 — Prompt Characteristics and Model Behavior

| Notebook | Description |
|---|---|
| `rq2_analysis.ipynb` | **Main RQ2 analysis.** Fits four binomial GEE models clustering responses by BBQ instance (`example_key`). Reports VIF checks, Benjamini–Hochberg correction (within-outcome and global), odds ratio plots, and QIC/QICu comparison of full vs. control-only models. |

---
