# Sustainable Behaviours Study

This repository contains the material needed to restart an analysis of the 2021 AVQ microdata on self-reported sustainable habits.  The aim is to rebuild the exploration, feature investigation, and modelling work from scratch, independently from the original master thesis.  The background material that accompanied the thesis (questionnaires, documentation, notes) is stored in the [`about/`](about/) folder so that you can consult it while designing the new study.

## Repository layout

```
about/                    → Survey documentation and thesis background notes
data/                     → Raw CSV files and processed datasets saved by the scripts
exploratory_analysis/     → Figures generated during the exploratory phase
trained_models/           → Plots exported during model fitting
exploratory_analysis.py   → Notebook-style exploratory cleaning script
features_selection.py     → t-SNE visualisation and clustering script
model_fitting.py          → Random forest training script
requirements.txt          → Python dependencies used by the scripts
```

## Getting started

1. (Optional) create a virtual environment and activate it.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux / macOS
   .venv\Scripts\activate     # Windows
   ```
2. Install the Python dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Review the materials in [`about/`](about/) to refresh the meaning of each survey variable before diving into the data.

## Running the existing scripts

The Python files were written as sequential scripts and rely on the raw dataset stored under `data/AVQ_Microdati_2021.csv`.  Run them directly with Python once you are ready to generate intermediate artefacts.

* **Exploratory preprocessing**
  ```bash
  python exploratory_analysis.py
  ```
  Saves cleaned datasets in `data/` and diagnostic plots in `exploratory_analysis/`.

* **Feature exploration**
  ```bash
  python features_selection.py
  ```
  Generates a t-SNE embedding, clustering plots, and stores them in `exploratory_analysis/`.

* **Model fitting**
  ```bash
  python model_fitting.py
  ```
  Trains baseline and balanced random forest models and exports feature importance plots in `trained_models/`.

## Suggested next steps

* Document any new hypotheses, alternative preprocessing ideas, or modelling goals directly in this repository as you iterate.
* Re-run the scripts after each change so you can track how the intermediate artefacts evolve.
* Once you are satisfied with the refreshed baseline, consider breaking the logic into notebooks or modules that better match your preferred workflow.

Happy analysing!
