# Sustainable Behaviours Study

This repository contains an exploratory data science study on self-reported sustainable behaviours using the 2021 ISTAT AVQ microdata. The project focuses on two target variables:

- `SPRACQUA`: attention to water-saving behaviours
- `SPRENER`: attention to energy-saving behaviours

Target coding is ordinal:

- `1` = high care / high attention
- `4` = low care / no attention

The objective is not to build a highly accurate predictive model. The goal is to understand which other behaviours, attitudes, and socio-demographic/spatial variables are associated with stronger or weaker attention to water and energy saving.

## Project Scope

The broader objective of the work is to understand the human characteristics that contribute to develop eco-friendly attitudes and behaviors. 
The main research question is whether attention to water saving and energy saving can be linked to a broader set of everyday behaviours, attitudes, and socio-demographic characteristics.
It is organised around three research goals:

1. **Goal 1 – human characteristics**: how socio-demographic variables and environmental attitudes relate to eco behaviours.
2. **Goal 2 – consumption**: which socioeconomic, motivational and preference factors explain variation in electricity and water consumption.
3. **Goal 3 – living area**: which spatial / local-service factors drive these behaviours.

## Current Repository Structure

```text
about/                               Survey documentation and contextual notes
data/                                Raw data, targets, and processed datasets
exploratory_analysis/                EDA outputs, heatmaps, boxplots, and clustering results
results/                             Random-forest outputs and explanatory-model results
feature_descriptions.py              Plain-language descriptions of survey columns (shared)
1_data_ingestion_first_dim_reduction.py
2_feature_selection.py
3_exploratory_analysis.py
4_clustering.py
5_rf_fitting.py
6_explanatory_models.py
requirements.txt
README.md
```

Note: For convenience, result directories (e.g., plots/, outputs/, etc.) have also been committed to the repository to facilitate result inspection, despite this going against the usual best practice of excluding generated outputs from version control.

## Data

The data are publicly available and have already been published in the repository. 
The analysis uses the 2021 AVQ survey microdata in [data/AVQ_Microdati_2021.csv](data/AVQ_Microdati_2021.csv). Script 1 builds the processed datasets used by every downstream step:

- [data/preprocessed_data_with_nan.csv](data/preprocessed_data_with_nan.csv): reduced feature matrix **before** imputation (299 features, contains NaN)
- [data/preprocessed_data_imputed.csv](data/preprocessed_data_imputed.csv): same matrix after median imputation (NaN-free)
- [data/preprocessed_data_final.csv](data/preprocessed_data_final.csv): ANOVA-reduced subset (65 features) used by scripts 3–5
- [data/tar_spracqua.csv](data/tar_spracqua.csv): target for water-saving care
- [data/tar_sprener.csv](data/tar_sprener.csv): target for energy-saving care

Data are splitted into four semantically meaningful variable blocks, that are protected from the cleaning of the generic statistical filters:

- `BLOCK_SOCIO` – sex, age, education, employment, income, household type/size, marital status
- `BLOCK_ATTITUDES` – the 15-item "environmental problems that worry you" checklist
- `BLOCK_CONSUMPTION` – energy/water behaviours and services
- `BLOCK_SPATIAL` – region, macro-area and perceived neighbourhood problems


## Output Policy

- **Data** (memberships, preprocessed matrices, raw scores) is written as **CSV**.
- **Results** (tables for the report) are written as **TXT tables** files so they can be read directly.
- **Plots** are written as **PNG**.

## Workflow

The project is organized as a simple linear pipeline.

1. [1_data_ingestion_first_dim_reduction.py](1_data_ingestion_first_dim_reduction.py)
   Loads the raw AVQ data, binarises the checklist batteries, protects the four goal-aligned blocks, and produces the reduced feature matrices and targets.

2. [2_feature_selection.py](2_feature_selection.py)
   Performs ANOVA-based feature filtering and generates the 65-feature subset and the ranking report used in the exploratory phase.

3. [3_exploratory_analysis.py](3_exploratory_analysis.py)
   Produces descriptive plots and bivariate summaries for the target variables.

4. [4_clustering.py](4_clustering.py)
   Clusters respondents (KMeans on the full 299-feature matrix, PCA-denoised) with bootstrap-stability evaluation and cluster–target cross-tables. Outputs to `exploratory_analysis/clustering/`.

5. [5_rf_fitting.py](5_rf_fitting.py)
   Uses Random Forest and Balanced Random Forest as exploratory models to rank the most relevant variables and summarise the shared correlates of the two targets. Outputs to `results/`.

6. [6_explanatory_models.py](6_explanatory_models.py)
   The formal explanatory stage with orderd models fitting. It runs four phases on a restricted **driver set** (SOCIO + ATTITUDES + SPATIAL; the 8 sibling behaviour items are excluded):
   - **Phase 1** – driver-set construction
   - **Phase 2** – L1 stability selection (bootstrap screening)
   - **Phase 3** – `OrderedModel` (proportional odds) with marginal effects + binary Logit robustness (1 vs 2–4)
   - **Phase 4** – validation via permutation importance and SHAP, aggregated by driver block
   Outputs to `results/explanatory/` (TXT tables + PNG plots).

## Main Outputs

### Exploratory outputs

The [exploratory_analysis/](exploratory_analysis/) folder contains descriptive plots, correlation views, and clustering-related outputs:

- `clustering/cluster_validation_metrics.txt` – silhouette/CH/DB plus bootstrap stability across k
- `clustering/cluster_summary.txt` – cluster sizes and target means
- `clustering/cluster_profiles.txt` – most distinctive features per cluster (with plain-language descriptions)
- `clustering/cluster_<target>_counts.txt` / `_proportions.txt` – target response distribution per cluster

Note: **Clustering** adds little information to the project. The respondent clustering (script 4) finds only soft clusters (silhouette 0.03 even at the best k) that separate people by age and digital/internet usage, while the two targets stay almost flat across clusters (means ~1.4-1.5 everywhere). No "sustainable-citizen" segment emerges, which is consistent with later findings of sustainability being a diffuse, cross-cutting disposition rather than a defining profile.


### Random-forest outputs (script 5)

Script 5 trains Random Forest and Balanced Random Forest classifiers to predict each target (care for energy / water) from the reduced-feature matrix. A random forest is an ensemble of many decision trees; "feature importance" measures how much each variable helps the trees separate the respondents. The Balanced Random Forest re-weights the classes so that the rare "no care" respondents (level 4 of the target variables) are not ignored.

All tables are pretty-printed TXT in [results/rf_study/](results/rf_study/):

- `class_distribution_<target>.txt` – how many respondents fall in each care level (1–4). Shows how skewed the target is (~70% at "always careful").
- `model_comparison_<target>.txt` – performance of a "dummy" model (always predict the majority class) vs Random Forest vs Balanced Random Forest: balanced accuracy (the average recall over the 4 classes; chance = 0.25) and MCC.
- `feature_importance_<target>.txt` – the full ranking of all 65 features by importance, with plain-language descriptions.
- `top_features_<target>.txt` – the top-20 of the ranking above.
- `descriptive_group_means_<target>.txt` – the average (z-standardised) value of the shared top features for each care level; this is the file that tells you the DIRECTION of the association (e.g. no-care respondents report reading labels less often).
- `shared_top_features.txt` – the features that rank high for BOTH targets, side by side.
- `features_importance_*.png`, `shared_feature_*.png` – visual versions of the tables above.

What this stage CAN and CANNOT tell you: importance ranks the variables, but it does not say whether a variable increases or decreases care, nor how significant the effect is. That is exactly what the next and final section adds.

### Explanatory-model outputs (script 6)

This section of the project wants to add a step to the analysis of the dataset, and to the DIRECTION analysis performed with the random forest fitting. Here, **ordinal models** are tested on the dataset.
These models are ideal when your dependent variable (your target) is categorical and has a natural ordering, exactly
as the SPRENER and SPRACQUA targets.
This ordered models measure the **DIRECTION, the STRENGTH and the statistical significance** of each association. It works on a RESTRICTED driver set of 38 background variables in three groups:

- **who you are** – sex, age, education, employment, income source, household type/size, marital status;
- **what worries you** – the 15 "environmental problems that worry you" items;
- **where you live** – region, macro-area, neighbourhood problems and services.

>**IMPORTANT**: The eight sibling behaviour variables from the same survey battery as the target variable are excluded. This prevents the models from trivially predicting one behaviour using closely related behaviours from the same Sustainability Behaviours section of the questionnaire.
Feature importance analyses using Random Forests showed that these sibling behaviours were among the strongest predictors. To better understand the underlying drivers of sustainability behaviours, rather than simply exploiting correlations within the same survey section, these variables were removed from the dataset. This is one of the main findings of the project.
In this revised analysis, all predictor variables are standardised (mean = 0, standard deviation = 1), making all estimated effects directly comparable on a per-standard-deviation basis.

The pipeline has 4 phases; each one writes its own TXT/PNG into [results/explanatory/](results/explanatory/):

1. **Driver set** -> `driver_set_summary.txt`. The 38 candidate drivers with plain-language descriptions.
2. **Shortlist** -> `l1_stability_<target>.txt`. "Does a driver keep showing up no matter which respondents we look at?" 60 times the script trains an L1-regularised logistic regression on a random 70% of respondents; the L1 penalty pushes irrelevant drivers' weights to exactly zero. A driver's selection frequency = the share of the 60 rounds in which it survived. Drivers stable in >= 80% of the rounds go to phase 3; the rest are treated as noise and dropped.
3. **Direction and strength** -> `ordinal_<target>.txt`, `binary_<target>.txt`. Because the target is ordered (1 = high care ... 4 = no care), a proportional-odds OrderedModel is fitted on the stable drivers. For each driver it reports: the coefficient (sign = direction), the odds ratio OR = exp(coef) per +1 SD (OR < 1 = driver pushes toward MORE care, OR > 1 = toward LESS care) and the average marginal effects (percentage-point change in the probability of each response class; "class 0" in the table = target value 1 = HIGH CARE). As a robustness check the target is collapsed to "care vs no-care" and refitted as a binary Logit; the signs of the two models are compared (agreement is counted only when both are significant).
4. **Validation** -> `validation_<target>.txt`, `shap_bar_<target>.png`, `block_importance_<target>.png`. A class-balanced random forest (no linearity assumption) ranks the drivers by permutation importance (how much the balanced accuracy drops when the driver's values are shuffled) and by mean |SHAP| (the average contribution of the driver to the model's predictions). Both are summed per driver block to show which of the three groups matters most overall. The test balanced accuracy (~0.37 for SPRENER, ~0.36 for SPRACQUA, vs a 0.25 chance baseline) shows that the driver set carries a weak but real signal.

`comparison_<target>.txt` is the best starting point: it puts ALL the evidence for one target side by side – L1 selection frequency, ordinal OR and p-value, binary OR, sign agreement, permutation importance and SHAP – in a single table.

## Main Findings So Far

**1. The targets share a common core.** Both targets are strongly skewed toward level 1 ("always careful"): about 68-70% of respondents, versus only ~5% at level 4 ("never careful"). The joint heatmap from script 3 shows a clear diagonal: high attention to water goes together with high attention to energy, and low with low. The two targets behave as one latent pro-environmental disposition rather than two isolated habits. (findings from the exploratory analsys [3_exploratory_analysis.py](3_exploratory_analysis.py))

**2. They share the same profile, but the signal is weak and diffuse.** In script 5, 24 features appear in both top-20 importance lists, and the top 5 are identical in order (USAGETT, ARUMOR, ALOCAL, ETICHET, BIOLOG); the shared-importance scatter hugs the diagonal. Yet no single feature dominates - the top one (USAGETT) accounts for only ~0.045 of total importance, and the Balanced Random Forest only lifts balanced accuracy to ~0.52-0.53 (chance = 0.25, MCC ~ 0.28). The importance ranking should therefore be read as a soft screening device rather than a strong separation. (findings from the random forest screening [5_rf_fitting.py](5_rf_fitting.py))

**3. The correlates cluster into three blocks.**

- Sustainable-consumption behaviours (sibling items from the same battery): `USAGETT`, `ARUMOR`, `ALOCAL`, `ETICHET`, `BIOLOG`, `TRASPO` - by far the strongest block.
- Institutional trust (0-10): `PUNTIFI10` (municipal government), `PUNTIFI3` (police), `FIDSCIE` (scientists), `PUNTIFI12` (fire brigade), `FIDMED`/`FIDINF` (health personnel).
- Cultural/social capital: `LIBFAM` (books in the household), `ETAMi` (age), `VOTOVI` (life satisfaction), `AMICI` (friends), `POLITI` (follows politics), `SICURO` (safety).

Notably, environmental-concern attitudes rank low (`CAMCLI`, "worried: climate change", is ~48/65) and spatial/SES variables are absent from the top: the profile is behavioural + trust-based, not attitude- or geography-driven.

**4. Direction.** No-care for energy/water respondents (level 4) show lower institutional trust, fewer books, and lower life satisfaction. The six consumption behaviours rise monotonically with the no-care level, meaning less-care respondents report doing them less often (reading labels, buying local/organic, avoiding noisy driving, using transport alternatives). Age is non-monotonic: the high-care group is the oldest, and level 4 sits near the average. [5_rf_fitting.py](5_rf_fitting.py)

**5. An exception: USAGETT.** Three items in the behaviours battery are not phrased as "pay attention to ...": `GCARTE` (throwing paper/cardboard in the street), `DOPFIL` (parking in double file), and `USAGETT` (using single-use/disposable products). The first two behave as expected for "poor behaviour" items: they show a **negative** association with the targets, so respondents who care less about water/energy report doing these things more often. `USAGETT` is the anomaly: its association with the targets is **positive**, i.e. it moves against the typical "good behaviour" gradient, so at face value the least-care respondents would be the ones using disposable products *less* often. Within an otherwise highly coherent pattern, `USAGETT` is a clear exception. [5_rf_fitting.py](5_rf_fitting.py)

**6. Consistency between the two stages.** Script 3 [3_exploratory_analysis.py](3_exploratory_analysis.py) already highlighted the coherent gradient of ALOCAL, ETICHET, BIOLOG, ARUMOR, and USAGETT across target classes, and flagged `QTSALE` (attention to salt intake) as a weaker, partly different pattern - dietary/health-related rather than environmental. Script 5 [5_rf_fitting.py](5_rf_fitting.py) confirms that shared gradient on a far larger feature set, adds the trust and cultural-capital blocks, and isolates USAGETT as the one behavioural item that goes against the good-behaviour gradient.

Note that this screening runs on the ANOVA-reduced matrix (65 features), which contains almost no SOCIO/ATTITUDES/SPATIAL drivers. Those drivers are examined directly in script 6 (points 8-9 below).

**8. Script 6 adds direction and significance.** The explanatory models (see `results/explanatory/comparison_<target>.txt` for all the evidence side by side; a fuller plain-language summary also sits at the bottom of [6_explanatory_models.py](6_explanatory_models.py)) confirm the picture above and give it direction:

- The two targets share the same driver profile: 23 (SPRENER) and 27 (SPRACQUA) of the 38 drivers are stable in >= 80% of the bootstrap resamples, with nearly identical ranking and signs.
- Age is by far the strongest driver: `ETAMi` is selected in 100% of the resamples and tops both SHAP and permutation importance for the two targets (OR per 1 SD 0.70 / 0.74; +1 SD of age raises P(high care) by ~7 points for SPRENER).
- Environmental concern is the second pillar: all 15 "worries" are stable, worry about climate change (`CAMCLI`) is the strongest of them, and every significant worry has OR < 1 (more worry -> more care).
- The socio block is mixed: women, more educated respondents and region of residence lean toward care; employment condition and income source lean against it.
- The spatial block is the weakest of the three (smallest block-level |SHAP| sums); only region and macro-area consistently matter, while heavy traffic and recycling-station presence lean toward LESS care.
- The ordinal and binary models agree in sign everywhere both are significant, so the results are robust to the model choice. (findings from the explanatory models [6_explanatory_models.py](6_explanatory_models.py))

**9. Putting it all together.** Script 5 [5_rf_fitting.py](5_rf_fitting.py) (with reduced features) says the strongest correlates are the sibling consumption behaviours (reading labels, buying local/organic, transport alternatives, avoiding disposable products) plus institutional trust and cultural capital. Script 6 [6_explanatory_models.py](6_explanatory_models.py) analyzing driver blocks says that among "who you are / what worries you / where you live", the profile is OLDER and ENVIRONMENTALLY-WORRIED, with living-area factors secondary.

The overall conclusion stands: saving water and energy does not behave like an isolated outcome - it is embedded in a broader pattern of sustainable consumption, everyday habits, and socio-demographic/spatial conditions.

## How To Run The Project

1. Create or activate a Python environment (the project is developed on a conda `forecasting` environment with Python 3.10).
2. Install the requirements.

```bash
pip install -r requirements.txt
```

3. Run the scripts in numeric order.

```bash
python 1_data_ingestion_first_dim_reduction.py
python 2_feature_selection.py
python 3_exploratory_analysis.py
python 4_clustering.py
python 5_rf_fitting.py
python 6_explanatory_models.py
```

## Interpretation Notes

The machine-learning results should be used as exploratory evidence, not as causal proof.

- Feature importance highlights variables associated with the targets.
- It does not show the direction of the relationship.
- It does not establish causality.
- The ordinal and binary models in script 6 provide the direction and significance of the associations; marginal effects give the magnitude on the response-class probabilities.

Within the scope of this project, the Random Forest stage works best as a final screening tool that helps identify the most plausible correlates to discuss in the study, while script 6 formalises those correlates with interpretable models.
