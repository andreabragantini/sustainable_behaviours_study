"""
================================================================================
Script 6 - Explanatory models for the drivers of sustainable behaviour
================================================================================

GOAL
----
Explain how socio-demographic, attitude and spatial variables are associated
with the two ordinal targets:

  - SPRENER : attention to avoiding electricity waste (1 = high, 4 = no care)
  - SPRACQUA: attention to avoiding water waste       (1 = high, 4 = no care)

Unlike script 5 (random-forest screening), this script works on a DELIBERATELY
RESTRICTED driver set.  The 8 sibling sustainable-behaviour items from the same
battery as the targets (ETICHET, BIOLOG, ALOCAL, GCARTE, DOPFIL, ARUMOR, TRASPO,
USAGETT) are EXCLUDED so that the analysis does not simply predict one habit
with its mirror habits.

    "USAGETT": "Uses single-use / disposable products",
    "ARUMOR": "Avoids noisy driving behaviour",
    "ALOCAL": "Buys local food/products",
    "ETICHET": "Reads ingredients on food-product labels",
    "BIOLOG": "Buys organic food/products",
    "TRASPO": "Chooses transport alternatives to the private car",
    "GCARTE": "Throws paper/cardboard in the street",
    "DOPFIL": "Parks the car in double file",

PIPELINE
-------------------------
The script answers one question per target: WHICH of the candidate "drivers"
(38 background variables about the respondent, their worries and their living
area) are reliably associated with how much people say they care about saving
energy / water, in WHICH direction, and with WHAT strength.

  Phase 1 - Choose the candidate drivers
      Build a restricted driver matrix from the three driver blocks:
        - SOCIO (8): sex, age, education, employment, income source,
          household type and size, marital status ("who you are");
        - ATTITUDES (15): the "environmental problems that worry you" items
          ("what worries you");
        - SPATIAL (15): region, macro-area and living-area problems/services
          ("where you live").
    WATCHOUT:
      The 8 sibling behaviour items from the same battery of questions as the targets are
      EXCLUDED (see list above): otherwise the script would simply predict one
      habit with its mirror habits.  
      
      Every driver is STANDARDISED (mean 0,
      standard deviation 1) so that all coefficients, odds ratios and marginal
      effects are comparable across drivers "per 1 SD".

  Phase 2 - Shortlist the drivers (L1 stability selection with multinomial logistic regression)
      Idea: a driver is trustworthy only if it keeps showing up as relevant no
      matter which random 70% of the respondents we train on.  The script draws
      60 such random subsets ("bootstrap resamples"); on each one it fits a
      multinomial L1 logistic regression (SAGA solver).  The L1 penalty pushes
      the weights of irrelevant drivers to exactly zero, so a driver "survives"
      a resample when its coefficient is not zero.  Its SELECTION FREQUENCY is
      the share of the 60 resamples in which it survived; drivers that are
      stable in >= 80% of resamples go on to Phase 3, the rest are treated as
      noise.  The penalty strength C is fixed at 0.01 (L1_SCREENING_C): the
      accuracy-optimal C keeps ALL 38 drivers (the effects are weak and diffuse)
      and would make the frequency ranking meaningless.  This is a screening
      step, not the final model.

  Phase 3 - Measure direction and strength (ordinal + binary models)
      The targets are ordered (1 = high care ... 4 = no care), so the natural
      model is a proportional-odds OrderedModel (statsmodels).  For each driver
      it reports:
        - the coefficient (sign = direction of the association);
        - the odds ratio OR = exp(coef): the multiplicative change in the odds
          of being more careful for each +1 SD of the driver (OR < 1 means the
          driver pushes TOWARD more care, OR > 1 toward less care);
        - average marginal effects (AME): the average percentage-point change
          in the probability of each response class for +1 SD of the driver
          (e.g. +1 SD of age raises P(high care) by ~7 points for SPRENER).
          statsmodels has no get_margeff for OrderedModel, so the AMEs are
          computed manually from the logistic derivatives, with standard errors
          from the delta method.
      Robustness check: the target is collapsed to "care vs no care" (1 vs
      2-4), refit with a plain binary Logit on the SAME drivers, and the signs
      of the two models are compared (agreement is counted only when BOTH are
      significant at p<0.05).  Two very different estimators that point in the
      same direction make the result trustworthy.

  Phase 4 - Double-check with a model-agnostic lens (validation)
      A class-balanced Random Forest is trained on the full driver set; unlike
      the logistic models it makes no linearity/odds assumptions.  Two
      importance measures rank the drivers:
        - permutation importance: how much the balanced accuracy drops when a
          driver's values are randomly shuffled (if shuffling does not hurt,
          the driver adds nothing);
        - mean |SHAP|: the average absolute contribution of each driver to the
          model's predictions across respondents.
      Both are summed per driver block (SOCIO / ATTITUDES / SPATIAL) to see
      which of the three groups matters most overall.

OUTPUTS (results/explanatory/, plain-text tables + PNG plots)
  - driver_set_summary.txt
  - l1_stability_<target>.txt
  - ordinal_<target>.txt            (coefficients, OR, CI, p, marginal effects)
  - binary_<target>.txt
  - validation_<target>.txt         (permutation importance, SHAP, block totals)
  - comparison_<target>.txt         (all evidence side by side)
  - shap_bar_<target>.png, block_importance_<target>.png

HOW TO RUN
----------
  python 6_explanatory_models.py    (from the repo root, forecasting env)
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
from scipy.stats import logistic, norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel

from feature_descriptions import DRIVER_BLOCKS, SIBLING_BEHAVIOURS, describe, render_table


warnings.simplefilter(action="ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")


RANDOM_STATE = 42
DATA_PATH = Path("data/preprocessed_data_imputed.csv")
TARGET_PATHS = {
    "sprener": Path("data/tar_sprener.csv"),
    "spracqua": Path("data/tar_spracqua.csv"),
}
OUTPUT_DIR = Path("results") / "explanatory"
BLOCK_ORDER = ["SOCIO", "ATTITUDES", "SPATIAL"]

# Stability selection settings.
N_BOOTSTRAP = 60
BOOTSTRAP_FRACTION = 0.7
CV_GRID_C = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
# The L1 penalty used for screening.  The CV-optimal C (~0.2) selects every
# driver (the driver effects are weak and diffuse), which makes the frequency
# ranking uninformative; at C <= 0.005 the model collapses toward the majority
# class.  C = 0.01 is the value where selection frequencies spread out, so it is
# used for the screening step.  The CV landscape is still printed as a
# diagnostic.
L1_SCREENING_C = 0.01
STABILITY_THRESHOLDS = {"50%": 0.50, "60%": 0.60, "80%": 0.80}

# Validation settings.
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
TEST_SIZE = 0.30
PERM_N_REPEATS = 5
PERM_SAMPLE = 5000
SHAP_SAMPLE = 1000


def load_data() -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Load the full imputed feature matrix and the two targets."""
    X = pd.read_csv(DATA_PATH)
    if X.isna().any().any():
        raise ValueError("Input features contain NaN values. Please impute before model fitting.")

    targets: dict[str, pd.Series] = {}
    for name, path in TARGET_PATHS.items():
        target_df = pd.read_csv(path)
        targets[name] = target_df.iloc[:, 0].astype(int)

    for name, y in targets.items():
        if len(y) != len(X):
            raise ValueError(f"Target {name} has {len(y)} rows, expected {len(X)}.")
    return X, targets


# -----------------------------------------------------------------------------
# Phase 1 - choose the candidate drivers
# -----------------------------------------------------------------------------

def build_driver_set(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Phase 1: keep only the three driver blocks and exclude sibling behaviours.

    Returns the (standardised) driver matrix and the per-block feature lists
    (restricted to the columns actually present in the data).
    """
    present_blocks: dict[str, list[str]] = {}
    for block_name, block_features in DRIVER_BLOCKS.items():
        present = [c for c in block_features if c in X.columns]
        present_blocks[block_name] = present

    drivers = [c for block in BLOCK_ORDER for c in present_blocks[block]]
    X_drivers_raw = X[drivers]

    absent = [c for block in DRIVER_BLOCKS.values() for c in block if c not in X.columns]
    if absent:
        print(f"Warning: driver columns not found in the data: {absent}")

    # Standardise so that ORs and marginal effects are expressed per 1 SD.
    scaler = StandardScaler()
    X_drivers = pd.DataFrame(
        scaler.fit_transform(X_drivers_raw),
        columns=X_drivers_raw.columns,
        index=X_drivers_raw.index,
    )
    return X_drivers, present_blocks


def write_driver_summary(X_drivers: pd.DataFrame, present_blocks: dict[str, list[str]]) -> None:
    """Save the driver-set table with descriptions."""
    rows = []
    for block_name in BLOCK_ORDER:
        for feature in present_blocks[block_name]:
            rows.append(
                {
                    "block": block_name,
                    "feature": feature,
                    "description": describe(feature),
                    "std_mean": X_drivers[feature].mean(),
                    "std_sd": X_drivers[feature].std(ddof=0),
                }
            )
    table = pd.DataFrame(rows)
    table["std_mean"] = 0.0
    table["std_sd"] = 1.0
    with (OUTPUT_DIR / "driver_set_summary.txt").open("w", encoding="utf-8") as fh:
        fh.write(
            render_table(
                table[["block", "feature", "description"]],
                title=(
                    f"Phase 1 - Driver set ({len(table)} features, standardised; "
                    "sibling behaviour items excluded)"
                ),
            )
            + "\n"
        )

    print(f"Driver set: {len(table)} features "
          f"({', '.join(f'{b}={len(present_blocks[b])}' for b in BLOCK_ORDER)}).")

# -----------------------------------------------------------------------------
# Phase 2 - Multinomial logistic regression with L1 stability selection
# -----------------------------------------------------------------------------

def choose_regularization(X: pd.DataFrame, y: pd.Series) -> float:
    """Report the CV landscape and return the fixed screening penalty.

    The CV balanced accuracy barely moves across C (the driver effects are
    weak), so the accuracy-optimal C selects every feature.  The screening
    therefore uses the documented constant ``L1_SCREENING_C``; the CV table is
    printed so the reader can see why.
    """
    subsample = min(15000, len(X))
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=subsample, replace=False)
    X_sub, y_sub = X.iloc[idx].to_numpy(), y.iloc[idx].to_numpy()

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for c in CV_GRID_C:
        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=c,
            multi_class="multinomial",
            max_iter=2000,
            tol=1e-3,
            random_state=RANDOM_STATE,
        )
        score = cross_val_score(model, X_sub, y_sub, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
        print(f"    L1 CV: C={c:.3f} -> balanced accuracy {score.mean():.4f} (SE {score.std(ddof=1) / np.sqrt(len(score)):.4f})")

    print(f"    Screening penalty used: C={L1_SCREENING_C:.3f}")
    return float(L1_SCREENING_C)


def l1_stability_selection(X: pd.DataFrame, y: pd.Series) -> tuple[pd.Series, float]:
    """Phase 2: selection frequency of each driver under L1-logistic bootstraps.

    A feature counts as selected in a resample when it receives a non-zero
    coefficient in at least one of the response classes.
    """
    print("\n  Running L1 stability selection...")
    best_c = choose_regularization(X, y)
    print(f"    Using C={best_c:.3f}")

    n = len(X)
    rng = np.random.default_rng(RANDOM_STATE)
    counts = pd.Series(0.0, index=X.columns)
    X_arr, y_arr = X.to_numpy(), y.to_numpy()

    for i in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=int(n * BOOTSTRAP_FRACTION), replace=False)
        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=best_c,
            multi_class="multinomial",
            max_iter=2000,
            tol=1e-3,
            random_state=RANDOM_STATE,
        )
        model.fit(X_arr[idx], y_arr[idx])
        selected = np.abs(model.coef_).sum(axis=0) > 1e-6
        counts += selected.astype(float)
        if (i + 1) % 10 == 0:
            print(f"    bootstrap {i + 1}/{N_BOOTSTRAP} done")

    frequencies = counts / N_BOOTSTRAP
    return frequencies, best_c


def write_l1_table(frequencies: pd.Series, target: str) -> pd.DataFrame:
    """Save the L1 selection-frequency table."""
    table = pd.DataFrame(
        {
            "feature": frequencies.index,
            "block": [next((b for b in BLOCK_ORDER if f in DRIVER_BLOCKS[b]), "?") for f in frequencies.index],
            "description": [describe(f) for f in frequencies.index],
            "selection_frequency": frequencies.values,
        }
    )
    table = table.sort_values("selection_frequency", ascending=False)
    with (OUTPUT_DIR / f"l1_stability_{target}.txt").open("w", encoding="utf-8") as fh:
        fh.write(
            render_table(
                table,
                title=(
                    f"Phase 2 - L1 stability selection for {target.upper()}\n"
                    f"{N_BOOTSTRAP} bootstrap resamples, penalty C={L1_SCREENING_C:.3f} "
                    "(see CV landscape in the console)"
                ),
                formats={"selection_frequency": ".3f"},
            )
            + "\n"
        )
    return table


def stable_driver_set(frequencies: pd.Series) -> list[str]:
    """Choose the features that will enter the ordinal/binary models."""
    for _, threshold in sorted(STABILITY_THRESHOLDS.items(), key=lambda kv: -kv[1]):
        stable = frequencies[frequencies >= threshold].index.tolist()
        if len(stable) >= 3:
            return stable
    return frequencies.sort_values(ascending=False).head(15).index.tolist()


# -----------------------------------------------------------------------------
# Phase 3 - ordinal and binary models
# -----------------------------------------------------------------------------
def ordered_marginal_effects_ame(res, X: pd.DataFrame) -> pd.DataFrame:
    """Average marginal effects for a proportional-odds OrderedModel.

    statsmodels does not implement ``get_margeff`` for ``OrderedModel``, so the
    effects are computed manually:

        P(y = c | x) = F(theta_c - x'b) - F(theta_{c-1} - x'b)
        dP(y=c)/dx_j = b_j * [ f(theta_{c-1} - x'b) - f(theta_c - x'b) ]

    with F/f the logistic cdf/pdf.  Standard errors come from the delta method
    applied to the sample-average effect.
    """
    X_arr = X.to_numpy(dtype=float)
    n, p = X_arr.shape
    k = res.model.k_levels
    # statsmodels orders OrderedModel params as [betas..., thresholds...].
    names = list(res.params.index)
    beta_names = [nm for nm in names if nm in X.columns]
    if len(beta_names) != p:
        raise ValueError("beta parameter names do not match the X columns.")
    order = [names.index(nm) for nm in beta_names] + [names.index(nm) for nm in names if nm not in X.columns]
    params = np.asarray(res.params, dtype=float)[order]
    V = np.asarray(res.cov_params(), dtype=float)[np.ix_(order, order)]
    beta = params[:p]
    theta = params[p:]
    n_params = len(params)

    Xb = X_arr @ beta

    fa = np.zeros((k, n))
    fb = np.zeros((k, n))
    fpa = np.zeros((k, n))
    fpb = np.zeros((k, n))
    for c in range(k):
        if c >= 1:
            a = theta[c - 1] - Xb
            fa[c] = logistic.pdf(a)
            fpa[c] = fa[c] * (1 - 2 * logistic.cdf(a))
        if c <= k - 2:
            b = theta[c] - Xb
            fb[c] = logistic.pdf(b)
            fpb[c] = fb[c] * (1 - 2 * logistic.cdf(b))

    class_labels = [int(v) for v in np.unique(np.asarray(res.model.endog).ravel())]

    records: list[dict] = []
    for c in range(k):
        d = fa[c] - fb[c]
        mean_d = d.mean()
        M = d @ X_arr / n  # mean(d * x_j)
        for j in range(p):
            jac = np.zeros(n_params)
            jac[:p] = -beta[j] * M
            jac[j] += mean_d
            if c >= 1:
                jac[p + c - 1] = beta[j] * fpa[c].mean()
            if c <= k - 2:
                jac[p + c] = beta[j] * (-fpb[c].mean())
            me = beta[j] * mean_d
            se = float(np.sqrt(jac @ V @ jac))
            z = me / se if se > 0 else np.nan
            p_value = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
            records.append(
                {
                    "class": f"class {class_labels[c]}",
                    "feature": X.columns[j],
                    "ME": me,
                    "SE": se,
                    "z": z,
                    "p": p_value,
                }
            )
    return pd.DataFrame(records)


def fit_ordinal_and_binary(X_stable: pd.DataFrame, y: pd.Series):
    """Phase 3: OrderedModel + binary Logit robustness for one target."""
    y_int = y.astype(int).reset_index(drop=True)
    X_stable = X_stable.reset_index(drop=True)

    # --- Ordinal model ---
    ordinal_model = OrderedModel(y_int, X_stable, distr="logit")
    ordinal_res = ordinal_model.fit(method="bfgs", disp=False)

    beta_mask = [nm in X_stable.columns for nm in ordinal_res.params.index]
    coef = ordinal_res.params.iloc[beta_mask]
    se = ordinal_res.bse.iloc[beta_mask]
    p_values = ordinal_res.pvalues.iloc[beta_mask]
    assert len(coef) == X_stable.shape[1], "unexpected number of beta parameters"
    or_value = np.exp(coef)
    or_lo = np.exp(coef - 1.96 * se)
    or_hi = np.exp(coef + 1.96 * se)

    ordinal_table = pd.DataFrame(
        {
            "feature": coef.index,
            "coef": coef.values,
            "OR": or_value.values,
            "OR_lo": or_lo.values,
            "OR_hi": or_hi.values,
            "z": (coef / se).values,
            "p": p_values.values,
            "sig": ["***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "" for p in p_values],
        }
    )

    me_table = ordered_marginal_effects_ame(ordinal_res, X_stable)

    # --- Binary robustness (1 vs 2-4) ---
    y_bin = (y_int > 1).astype(int)
    X_const = sm.add_constant(X_stable, has_constant="add")
    logit_res = sm.Logit(y_bin, X_const).fit(disp=False)

    logit_coef = logit_res.params.drop("const")
    logit_se = logit_res.bse.drop("const")
    logit_p = logit_res.pvalues.drop("const")
    bin_or = np.exp(logit_coef)
    bin_or_lo = np.exp(logit_coef - 1.96 * logit_se)
    bin_or_hi = np.exp(logit_coef + 1.96 * logit_se)

    binary_table = pd.DataFrame(
        {
            "feature": logit_coef.index,
            "coef": logit_coef.values,
            "OR": bin_or.values,
            "OR_lo": bin_or_lo.values,
            "OR_hi": bin_or_hi.values,
            "z": (logit_coef / logit_se).values,
            "p": logit_p.values,
            "sig": ["***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "" for p in logit_p],
        }
    )

    # Sign-agreement check between ordinal and binary coefficients.
    merged = ordinal_table[["feature", "coef", "p"]].merge(
        binary_table[["feature", "coef", "p"]], on="feature", suffixes=("_ord", "_bin")
    )
    merged["sign_agreement"] = np.sign(merged["coef_ord"]) == np.sign(merged["coef_bin"])
    both_sig = (merged["p_ord"] < 0.05) & (merged["p_bin"] < 0.05)
    merged.loc[~both_sig, "sign_agreement"] = False  # no meaningful agreement if either is n.s.

    return {
        "ordinal_res": ordinal_res,
        "ordinal_table": ordinal_table,
        "me_table": me_table,
        "binary_table": binary_table,
        "agreement": merged,
    }


def write_ordinal_outputs(target: str, results: dict) -> None:
    """Save the ordinal + marginal-effect tables as TXT."""
    with (OUTPUT_DIR / f"ordinal_{target}.txt").open("w", encoding="utf-8") as fh:
        fh.write(
            render_table(
                results["ordinal_table"],
                title=(
                    f"Phase 3 - OrderedModel (proportional odds) for {target.upper()}\n"
                    "Coefficients on standardised drivers; OR = exp(coef), per 1 SD. "
                    "sig: *** p<0.001, ** p<0.01, * p<0.05"
                ),
                formats={
                    "coef": ".4f", "OR": ".3f", "OR_lo": ".3f", "OR_hi": ".3f",
                    "z": ".3f", "p": ".4g",
                },
            )
            + "\n\n"
        )
        fh.write(
            render_table(
                results["me_table"],
                title="Average marginal effects on P(class) - logit scale probabilities",
                formats={"ME": ".5f", "SE": ".5f", "z": ".3f", "p": ".4g"},
            )
            + "\n"
        )


def write_binary_outputs(target: str, results: dict) -> None:
    """Save the binary robustness tables as TXT."""
    with (OUTPUT_DIR / f"binary_{target}.txt").open("w", encoding="utf-8") as fh:
        fh.write(
            render_table(
                results["binary_table"],
                title=f"Phase 3 - Binary robustness Logit (1 vs 2-4) for {target.upper()}",
                formats={
                    "coef": ".4f", "OR": ".3f", "OR_lo": ".3f", "OR_hi": ".3f",
                    "z": ".3f", "p": ".4g",
                },
            )
            + "\n\n"
        )
        fh.write(
            render_table(
                results["agreement"],
                title="Sign agreement between ordinal and binary coefficients (at p<0.05)",
                formats={"coef_ord": ".4f", "coef_bin": ".4f", "p_ord": ".4g", "p_bin": ".4g"},
            )
            + "\n"
        )


# -----------------------------------------------------------------------------
# Phase 4 - validation via permutation importance and SHAP
# -----------------------------------------------------------------------------
def run_validation(X: pd.DataFrame, y: pd.Series) -> dict:
    """Permutation importance + SHAP on a balanced random forest.

    The forest depth is capped so that SHAP's path-dependent TreeExplainer
    stays fast; permutation importance is computed on a subsample of the test
    set.  Both choices do not change the qualitative ranking of drivers.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    perm_index = np.random.default_rng(RANDOM_STATE).choice(
        len(X_test), size=min(PERM_SAMPLE, len(X_test)), replace=False
    )
    X_perm = X_test.iloc[perm_index]
    y_perm = y_test.iloc[perm_index]
    perm = permutation_importance(
        rf, X_perm, y_perm,
        n_repeats=PERM_N_REPEATS,
        random_state=RANDOM_STATE,
        scoring="balanced_accuracy",
        n_jobs=-1,
    )

    n_sample = min(SHAP_SAMPLE, len(X_train))
    X_sample = X_train.sample(n=n_sample, random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(rf)
    shap_values_raw = explainer.shap_values(X_sample)

    if isinstance(shap_values_raw, list):
        abs_shap = np.stack([np.abs(a) for a in shap_values_raw], axis=-1).mean(axis=-1)
    else:
        abs_shap = np.abs(shap_values_raw)
        if abs_shap.ndim == 3:
            abs_shap = abs_shap.mean(axis=2)

    shap_mean = pd.Series(abs_shap.mean(axis=0), index=X.columns)

    return {
        "rf": rf,
        "perm_mean": pd.Series(perm.importances_mean, index=X.columns),
        "perm_std": pd.Series(perm.importances_std, index=X.columns),
        "shap_mean": shap_mean,
        "test_balanced_acc": balanced_accuracy_score(y_test, rf.predict(X_test)),
    }


def write_validation_outputs(target: str, validation: dict, present_blocks: dict[str, list[str]]) -> pd.DataFrame:
    """Save validation tables and block-aggregation plots."""
    table = pd.DataFrame(
        {
            "feature": validation["shap_mean"].index,
            "block": [next((b for b in BLOCK_ORDER if f in present_blocks[b]), "?") for f in validation["shap_mean"].index],
            "description": [describe(f) for f in validation["shap_mean"].index],
            "perm_importance": validation["perm_mean"].values,
            "perm_sd": validation["perm_std"].values,
            "mean_abs_shap": validation["shap_mean"].values,
        }
    )
    table = table.sort_values("mean_abs_shap", ascending=False)

    with (OUTPUT_DIR / f"validation_{target}.txt").open("w", encoding="utf-8") as fh:
        fh.write(
            render_table(
                table,
                title=(
                    f"Phase 4 - Validation for {target.upper()}: permutation importance "
                    "(balanced-accuracy drop) and mean |SHAP|\n"
                    f"Test balanced accuracy: {validation['test_balanced_acc']:.4f}"
                ),
                formats={"perm_importance": ".4f", "perm_sd": ".4f", "mean_abs_shap": ".4f"},
            )
            + "\n\n"
        )

        block_table = pd.DataFrame(
            {
                "block": BLOCK_ORDER,
                "features": [len(present_blocks[b]) for b in BLOCK_ORDER],
                "sum_perm_importance": [
                    table.loc[table["block"] == b, "perm_importance"].sum() for b in BLOCK_ORDER
                ],
                "sum_mean_abs_shap": [
                    table.loc[table["block"] == b, "mean_abs_shap"].sum() for b in BLOCK_ORDER
                ],
            }
        )
        fh.write(
            render_table(
                block_table,
                title="Block-level aggregation",
                formats={"sum_perm_importance": ".4f", "sum_mean_abs_shap": ".4f"},
            )
            + "\n"
        )

    # SHAP bar plot (top 20), coloured by block.
    top = table.head(20).sort_values("mean_abs_shap")
    block_colors = {"SOCIO": "#4c78a8", "ATTITUDES": "#72b7b2", "SPATIAL": "#f58518"}
    colors = [block_colors.get(b, "#b6b6b6") for b in top["block"]]

    fig, ax = plt.subplots(figsize=(9, 8), dpi=250)
    ax.barh(top["feature"].tolist(), top["mean_abs_shap"].tolist(), color=colors)
    ax.set_xlabel("Mean |SHAP| (balanced RF, standardised drivers)")
    ax.set_title(f"Top drivers by SHAP importance - {target.upper()}")
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_bar_{target}.png")
    plt.close(fig)

    # Block-level plot.
    fig, ax = plt.subplots(figsize=(7, 5), dpi=250)
    block_cols = [block_colors[b] for b in block_table["block"]]
    ax.bar(block_table["block"], block_table["sum_mean_abs_shap"], color=block_cols)
    ax.set_ylabel("Sum of mean |SHAP|")
    ax.set_title(f"Driver-block importance - {target.upper()}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"block_importance_{target}.png")
    plt.close(fig)

    return table


def write_comparison(target: str, l1_table: pd.DataFrame, ordinal: dict, validation: pd.DataFrame) -> None:
    """Combine all evidence for one target into a single side-by-side table."""
    merged = (
        l1_table[["feature", "block", "selection_frequency"]]
        .merge(
            ordinal["ordinal_table"][["feature", "coef", "OR", "p", "sig"]],
            on="feature", how="outer",
        )
        .merge(
            ordinal["binary_table"][["feature", "OR", "p"]].rename(
                columns={"OR": "binary_OR", "p": "binary_p"}
            ),
            on="feature", how="outer",
        )
        .merge(
            ordinal["agreement"][["feature", "sign_agreement"]],
            on="feature", how="outer",
        )
        .merge(
            validation[["feature", "perm_importance", "mean_abs_shap"]],
            on="feature", how="outer",
        )
    )
    merged = merged.rename(
        columns={
            "coef": "ordinal_coef",
            "OR": "ordinal_OR",
            "p": "ordinal_p",
            "sig": "ordinal_sig",
        }
    )
    merged["description"] = merged["feature"].map(describe)
    merged = merged.sort_values("mean_abs_shap", ascending=False, na_position="last")

    with (OUTPUT_DIR / f"comparison_{target}.txt").open("w", encoding="utf-8") as fh:
        fh.write(
            render_table(
                merged,
                title=(
                    f"Combined evidence for {target.upper()}\n"
                    "L1 = selection frequency; ordinal = OrderedModel; "
                    "binary = Logit 1 vs 2-4; shap = mean |SHAP|"
                ),
                formats={
                    "selection_frequency": ".3f",
                    "ordinal_coef": ".4f",
                    "ordinal_OR": ".3f",
                    "ordinal_p": ".4g",
                    "binary_OR": ".3f",
                    "binary_p": ".4g",
                    "perm_importance": ".4f",
                    "mean_abs_shap": ".4f",
                },
            )
            + "\n"
        )


def analyse_target(
    target: str,
    X_drivers: pd.DataFrame,
    y: pd.Series,
    present_blocks: dict[str, list[str]],
) -> dict:
    """Run Phases 2-4 for a single target."""
    print("\n" + "=" * 78)
    print(f"Analysing target: {target.upper()}")
    print("=" * 78)

    #========================================================================================
    # Phase 2 - L1 stability selection
    frequencies, _ = l1_stability_selection(X_drivers, y)
    l1_table = write_l1_table(frequencies, target)
 
    stable_features = stable_driver_set(frequencies)
    print(f"\n  Stable driver subset ({len(stable_features)} features): {stable_features}")
    X_stable = X_drivers[stable_features]

    #========================================================================================
    # Phase 3 - ordinal + binary models
    ordinal = fit_ordinal_and_binary(X_stable, y)
    write_ordinal_outputs(target, ordinal)
    write_binary_outputs(target, ordinal)

    #========================================================================================
    # Phase 4 - validation
    validation = run_validation(X_drivers, y)
    validation_table = write_validation_outputs(target, validation, present_blocks)

    write_comparison(target, l1_table, ordinal, validation_table)
    return {"stable_features": stable_features, "l1_table": l1_table}


def main() -> None:
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Running explanatory models...")

    X, targets = load_data()

    #========================================================================================
    # Phase 1 - driver-set construction
    X_drivers, present_blocks = build_driver_set(X)
    write_driver_summary(X_drivers, present_blocks)

    #========================================================================================
    # Phases 2-4 - per-target analysis
    for target in TARGET_PATHS:
        analyse_target(target, X_drivers, targets[target], present_blocks)

    print(f"\nAll outputs written to: {OUTPUT_DIR}")
    print(f"Total time: {(time.time() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()


# ================================================================================
# FINDINGS AND HOW TO READ THE RESULTS
# ================================================================================
#
# Everything below summarises a successful run of this script
# All results live in results/explanatory/.
#
# --------------------------------------------------------------------------------
# 1. WHAT THE SCRIPT DOES (4 phases, per target)
# --------------------------------------------------------------------------------
#   Phase 1 - Driver-set construction (results/explanatory/driver_set_summary.txt)
#       Builds a restricted, standardised driver matrix of 38 features:
#       SOCIO (8: sex, age, education, employment, income source, household
#       type/size, marital status) + ATTITUDES (15 "environmental worries") +
#       SPATIAL (15 living-area problems and services).  The 8 sibling
#       behaviour items from the same battery as the targets are EXCLUDED, so
#       the script does not predict one habit with its mirror habits.
#
#   Phase 2 - L1 stability selection (results/explanatory/l1_stability_*.txt)
#       60 bootstrap subsamples, each fits a multinomial L1 logistic
#       regression at C=0.01; a driver is "stable" if it survives with a
#       non-zero coefficient in most resamples.  The CV landscape (printed on
#       the console) shows balanced accuracy ~0.25 everywhere, i.e. the driver
#       effects are weak and diffuse; C=0.01 is the point where the frequency
#       ranking becomes informative instead of selecting all 38 features.
#
#   Phase 3 - Ordinal + binary models
#       (results/explanatory/ordinal_*.txt, binary_*.txt)
#       A proportional-odds OrderedModel is fitted on the stable drivers;
#       odds ratios (OR = exp(coef), per 1 SD) and average marginal effects
#       (AME) per response class are reported.  statsmodels has no get_margeff
#       for OrderedModel, so the AMEs are computed manually with delta-method
#       standard errors.  Robustness: the same drivers are refitted as a
#       binary Logit (1 vs 2-4) and the sign agreement between the two models
#       is reported (only counted as agreement when BOTH are significant).
#
#       READING THE MARGINAL-EFFECTS TABLES:
#       the target is coded 1 = high care, 4 = no care, and the model sorts the
#       classes ascending, so "class 0" in the ME table = target value 1 =
#       HIGH CARE.  A negative ordinal coefficient (OR < 1) means the driver
#       shifts probability mass TOWARD high care; a positive ME on "class 0"
#       is the same thing on the probability scale (e.g. +1 SD of age raises
#       P(high care) by ~7 percentage points for SPRENER).
#
#   Phase 4 - Validation
#       (results/explanatory/validation_*.txt, shap_bar_*.png,
#       block_importance_*.png)
#       A class-balanced random forest on the full driver set; permutation
#       importance (drop in balanced accuracy) and mean |SHAP| per driver,
#       plus block-level totals.  Test balanced accuracy: 0.374 (SPRENER) and
#       0.355 (SPRACQUA) vs a ~0.25 near-chance baseline: the driver set
#       carries a modest but real signal.
#
# --------------------------------------------------------------------------------
# 2. MAIN FINDINGS
# --------------------------------------------------------------------------------
# (a) The two targets share the same driver profile.
#     L1 stability selects 23 stable drivers for SPRENER and 27 for SPRACQUA
#     (out of 38, 80% stability threshold); the ranking and the signs are
#     nearly identical across targets.  Energy and water care are driven by
#     the same socio-demographic, attitude and spatial conditions.
#
# (b) Age is by far the strongest driver.
#     ETAMi is selected in 100% of bootstraps and is first in both SHAP and
#     permutation importance for both targets.  Ordinal OR per 1 SD: 0.695
#     (SPRENER) and 0.739 (SPRACQUA); AME on P(high care) +0.0725 / +0.061.
#     Older respondents are much more likely to report high care for both
#     energy and water.
#
# (c) Environmental concern is the second pillar.
#     All 15 "worries" are selected at high frequency.  Among them, worry
#     about climate change (CAMCLI) is the strongest (SHAP/permutation rank
#     2 for both targets; OR 0.882 / 0.899), followed by waste (SMARIF),
#     pollution of rivers and seas (INQFIU), air pollution (IARIA), soil
#     pollution (INQSU) and catastrophes (CATASTR).  Every significant
#     attitude driver has OR < 1: more worry -> more care.
#
# (d) The socio-demographic block is mixed but meaningful.
#     Females are more likely to report high care (SESSO OR 0.896 / 0.879);
#     higher education (ISTRMi) and region of residence (REGMf) also lean
#     toward care.  In contrast, employment condition (CONDMi, OR 1.072 /
#     1.073) and income source (REDPRMi) point the other way: drivers with
#     "harder" living conditions (looking for work, housewife, retired...) are
#     slightly LESS likely to report care.  Household size (NCOMP) leans the
#     same way for SPRENER (OR 1.051).  Marital status and age-of-household
#     effects are essentially null in the ordinal model despite high L1
#     frequency - typical of a weak but repeatedly "on" signal.
#
# (e) The spatial block is the weakest of the three.
#     Block-level sum of mean |SHAP|: SPRENER SOCIO 0.070 / ATTITUDES 0.058 /
#     SPATIAL 0.034; SPRACQUA 0.062 / 0.062 / 0.035.  Region and macro-area
#     (REGMf, RIPMf) are the only spatial drivers that consistently matter.
#     Two interesting reversals: heavy traffic (TRAF, OR ~1.02-1.04, positive)
#     and recycling-station presence (ECOSTAZ, positive for SPRACQUA) lean
#     toward LESS care, while street litter (SPORCO) and unpleasant odours
#     (ODSGR) lean toward MORE care.
#
# (f) The ordinal and binary models agree.
#     Sign agreement is True for every driver that is significant in BOTH
#     models; the only "False" rows are drivers non-significant in one or both
#     (e.g. STCIVMi, REDPRMi, NCOMP, AMRUM for SPRACQUA).  There is no genuine
#     sign disagreement between the two estimators.
#
# (g) The overall story.
#     Attention to saving water and energy is NOT an isolated habit.  It is
#     the behaviour of older people who are generally worried about the
#     environment (climate, waste, pollution) and who live in regions where
#     these concerns are more salient; neighbourhood services and nuisances
#     play only a secondary role.  This echoes the cluster and RF findings:
#     care for water/energy is part of a broader profile rather than a
#     stand-alone choice.
#
# --------------------------------------------------------------------------------
# 3. QUICK INDEX TO THE OUTPUT FILES (all under results/explanatory/)
# --------------------------------------------------------------------------------
#   driver_set_summary.txt          Phase 1 driver set with descriptions
#   l1_stability_<target>.txt       Phase 2 selection frequencies (bootstrap L1)
#   ordinal_<target>.txt            Phase 3 coefficients, OR (1 SD), CI, p, AME per class
#   binary_<target>.txt             Phase 3 binary Logit robustness + sign agreement
#   validation_<target>.txt         Phase 4 permutation importance, mean |SHAP|, block totals
#   comparison_<target>.txt         ALL evidence side by side (the best single file per target)
#   shap_bar_<target>.png           Top-20 drivers by mean |SHAP|, coloured by block
#   block_importance_<target>.png   Sum of mean |SHAP| per driver block
#
#   <target> = sprener (energy) or spracqua (water).
# ================================================================================
