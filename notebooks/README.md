# Notebooks — exploration only

These notebooks are for **EDA and exploration**, not the pipeline. The pipeline
lives in `src/` and is run via `python src/train.py`. Anything that becomes part
of the workflow should be promoted into a module under `src/` and covered by a
test, not left in a notebook.

Suggested notebooks:
- `01_kiba_eda.ipynb` — affinity distribution, drug/target counts, scaffold
  diversity, why a scaffold split matters here.
- `02_scaffold_split.ipynb` — visualise the Bemis-Murcko scaffold groups and
  confirm zero train/test scaffold overlap.
