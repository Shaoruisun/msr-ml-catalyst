# -*- coding: utf-8 -*-
"""Step 2 -- train, select, and calibrate the MeOH-conversion model.

One self-contained script with three consecutive sub-steps:

1. Train and select.  Nine base candidates plus a training-OOF weighted blend
   under the pure row-random protocol: 80/20 split (seed 42), hyper-parameters
   and blend members chosen by training-only 5-fold CV, holdout scored only
   after selection is locked.  The whole protocol is then repeated over seeds
   42-51 for the stability table (``--seeds`` can narrow this).
2. Fit the uncertainty layer.  A conformalized quantile-GB pair (q16/q84,
   bracketed-conformal multiplier k_br, 68% nominal coverage) is calibrated on
   the training split of the deployed blend and written to
   ``output/uncertainty_model.pkl`` (+ ``output/reports/uncertainty_calibration.json``).
   The PSO stage consumes this to report predictive SD.
3. Export per-row out-of-fold predictions and predictive SD for the diagnostic
   panels -> ``output/tables/oof_predictions.csv``.

This file is self-contained: the feature contract, the release verifier and
the two uncertainty sub-steps are inlined below so the stage runs as a single
script.  The logic is identical to the original modular chain (model_train.py
+ fit_uncertainty.py + export_oof_uncertainty.py + pipeline_config.py +
v6_features.py + release_contract.py).

Usage:
    python ML/train/model_train.py [--seeds 42-51] [--input PATH] [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (KFold, RandomizedSearchCV, cross_validate,
                                     learning_curve, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

SCRIPT_DIR = Path(__file__).resolve().parent          # ML/train
OUTPUT_DIR = SCRIPT_DIR / "output"
PRIMARY_DATA = SCRIPT_DIR.parent / "data_clean" / "prep_output" / "ni_model_primary.csv"
DATASET_MANIFEST = PRIMARY_DATA.parent / "dataset_manifest.json"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_DIR = OUTPUT_DIR / "reports"
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
for _d in (TABLE_DIR, REPORT_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
STABILITY_SEEDS = tuple(range(42, 52))
TEST_SIZE = 0.20
CV_FOLDS = 5
SEARCH_ITERATIONS = 50
# Worker count for the CV searches and the learning curve.  Overridable so a
# run can be throttled on a memory-constrained machine.
CV_JOBS = max(1, int(os.environ.get("MSR_N_JOBS") or min(4, os.cpu_count() or 1)))
TARGET = "MeOH_Conversion_%"
ONEHOT_MIN_FREQUENCY = 5
MODEL_ORDER = {"SVR": 1, "Random Forest": 2, "Gradient Boosting": 3,
               "HistGradientBoosting": 4, "XGBoost": 5, "AdaBoost": 6,
               "Bagging": 7, "Extra Trees": 8}

# ── Feature contract (v6_scientific_features_3, 31 features) ────────────────
FEATURE_SCHEMA_VERSION = "v6_scientific_features_3"

RAW_NUMERIC_FEATURES = [
    "Reaction_Temp_C", "Ni_Loading_wt%", "Promoter_Loading_wt%",
    "Calcination_Temp_C", "Reduction_Temp_C", "Dry_Temp_C",
    "Calcination_Time_h", "Reduction_Time_h", "Dry_Time_h", "S_C_Ratio",
    "TOS_h", "Pressure_bar", "Reaction_Time_h",
]
DERIVED_NUMERIC_FEATURES = [
    "GHSV_log", "WHSV_log",
    "GHSV_h_inv_log", "LHSV_h_inv_log", "TOS_log",
    "prom_electronegativity", "prom_ionic_radius", "prom_atomic_mass",
    "prom_first_ionization", "support_acid_base",
]
NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES
CATEGORICAL_FEATURES = [
    "Support", "Promoter_Metal", "Metal_Loading_Method", "Precursor_Family",
    "Support_Prep_Method", "support_type",
]
BINARY_FEATURES = ["has_promoter", "support_reducible"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
FORBIDDEN_MODEL_FEATURES = {
    "Paper_ID", "Source_File", "raw_row_id", "record_id", "row_id",
    "original_row_sha256", "model_eligible", "review_method", "evidence_id",
    "page_num", "figure_binding_status", "data_source",
}


# ── Release verification (same gate as the cleaning stage) ──────────────────
def verify_release(source: Path) -> dict:
    """Verify the released database against its manifest.

    The manifest's ``official_input`` was recorded on the authors' machine, so
    only the file NAME is compared here -- the SHA-256 is the binding identity
    check.  The per-evidence-file hash loop of the original audit gate is
    skipped in this release: the underlying PDF/figure evidence is not part of
    the repository.
    """
    source = Path(source).resolve()
    info = json.loads((source.parent / "manifest.json").read_text(encoding="utf-8"))
    if info.get("release_status") != "verified":
        raise ValueError("Raw v6 has not passed source review")
    if Path(info.get("official_input", "")).name != source.name:
        raise ValueError("Official source identity mismatch")
    if hashlib.sha256(source.read_bytes()).hexdigest() != info.get("official_sha256"):
        raise ValueError("Official source hash mismatch")
    dependencies = info.get("input_hashes", {})
    required = {"corrections.csv", "evidence.csv", "extensions.csv", "additions.csv",
                "record_reviews.csv", "source_reviews.csv", "duplicate_relations.csv"}
    if not required.issubset(dependencies):
        raise ValueError("Release dependency manifest incomplete")
    for name, expected in dependencies.items():
        path = (source.parent / name).resolve()
        if path.parent != source.parent or not path.exists():
            raise ValueError("Release dependency missing: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Release dependency modified: " + name)
    return info


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class WeightedBlendRegressor(BaseEstimator, RegressorMixin):
    """Non-negative blend whose weights are fixed from training OOF predictions."""

    def __init__(self, estimators, weights, names):
        self.estimators = estimators
        self.weights = weights
        self.names = names

    def fit(self, X, y):
        self.estimators_ = [clone(model).fit(X, y) for model in self.estimators]
        return self

    def predict(self, X):
        matrix = np.column_stack([model.predict(X) for model in self.estimators_])
        return matrix @ np.asarray(self.weights, dtype=float)


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=ONEHOT_MIN_FREQUENCY))]), CATEGORICAL_FEATURES),
        ("bin", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True))]), BINARY_FEATURES),
    ], remainder="drop", verbose_feature_names_out=True)


def estimator(name: str, params: dict) -> object:
    if name == "Random Forest": return RandomForestRegressor(random_state=SEED, n_jobs=1, **params)
    if name == "Extra Trees": return ExtraTreesRegressor(random_state=SEED, n_jobs=1, **params)
    if name == "Gradient Boosting": return GradientBoostingRegressor(random_state=SEED, **params)
    if name == "HistGradientBoosting": return HistGradientBoostingRegressor(random_state=SEED, **params)
    if name == "XGBoost": return XGBRegressor(random_state=SEED, n_jobs=1, objective="reg:squarederror", verbosity=0, **params)
    raise ValueError(name)


def specs() -> list[dict]:
    # Capacity-hardened configurations: the first released-v6 run showed a large
    # training-only CV train-vs-validation gap (0.16-0.25) across tree families,
    # i.e. systematic overfitting.  Spaces are narrowed toward stronger
    # regularization using that train-side signal only; selection rules and any
    # test-set handling are unchanged.
    fixed = {
        "Random Forest": {"n_estimators": 600, "max_depth": 5, "min_samples_leaf": 40, "min_samples_split": 40, "max_features": 0.7, "bootstrap": True},
        "Gradient Boosting": {"n_estimators": 400, "learning_rate": 0.02, "max_depth": 2, "min_samples_leaf": 40, "min_samples_split": 40, "max_features": 0.8, "subsample": 0.75},
        "HistGradientBoosting": {"max_iter": 400, "learning_rate": 0.03, "max_leaf_nodes": 7, "min_samples_leaf": 60, "l2_regularization": 30.0},
        "XGBoost": {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 2, "min_child_weight": 30, "subsample": 0.75, "colsample_bytree": 0.8, "reg_alpha": 1.0, "reg_lambda": 300.0, "gamma": 2.0},
        "Extra Trees": {"n_estimators": 700, "max_depth": 5, "min_samples_leaf": 40, "min_samples_split": 40, "max_features": 0.8, "bootstrap": False},
    }
    out = [{"candidate": n, "model": n, "rank": MODEL_ORDER[n], "params": p, "estimator": estimator(n, p), "tune": False} for n, p in fixed.items()]
    trees_a = [DecisionTreeRegressor(max_depth=d, min_samples_leaf=l, random_state=SEED) for d in (2, 3, 4) for l in (10, 20, 40)]
    trees_b = [DecisionTreeRegressor(max_depth=d, min_samples_leaf=l, random_state=SEED) for d in (2, 3, 4) for l in (10, 20, 40)]
    out += [
        {"candidate": "SVR", "model": "SVR", "rank": 1, "params": {}, "estimator": SVR(kernel="rbf"), "tune": True, "space": {"model__C": [30, 100, 300, 1000, 3000], "model__epsilon": [0.05, 0.1, 0.2, 0.5, 1, 2], "model__gamma": ["scale", 0.003, 0.01, 0.03, 0.1]}},
        {"candidate": "AdaBoost", "model": "AdaBoost", "rank": 5, "params": {}, "estimator": AdaBoostRegressor(random_state=SEED), "tune": True, "space": {"model__estimator": trees_a, "model__n_estimators": [100, 200, 400, 700], "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2], "model__loss": ["linear", "square", "exponential"]}},
        {"candidate": "Bagging", "model": "Bagging", "rank": 6, "params": {}, "estimator": BaggingRegressor(random_state=SEED, n_jobs=-1), "tune": True, "space": {"model__estimator": trees_b, "model__n_estimators": [100, 200, 400, 700], "model__max_samples": [0.55, 0.7, 0.85, 1.0], "model__max_features": [0.45, 0.6, 0.8, 1.0], "model__bootstrap": [True]}},
    ]
    for spec in out:
        if spec["tune"]:
            continue
        spec["tune"] = True
        if spec["model"] in {"Random Forest", "Extra Trees"}:
            spec["space"] = {"model__max_depth": [4, 6, 8, None],
                             "model__min_samples_leaf": [10, 20, 40, 80],
                             "model__max_features": [0.4, 0.7, 1.0]}
        elif spec["model"] == "Gradient Boosting":
            spec["space"] = {"model__max_depth": [2, 3, 4],
                             "model__min_samples_leaf": [10, 20, 40],
                             "model__learning_rate": [0.01, 0.02, 0.03],
                             "model__subsample": [0.65, 0.85, 1.0]}
        elif spec["model"] == "HistGradientBoosting":
            spec["space"] = {"model__max_leaf_nodes": [7, 15],
                             "model__min_samples_leaf": [40, 60, 80],
                             "model__learning_rate": [0.01, 0.02, 0.03],
                             "model__l2_regularization": [10.0, 30.0, 100.0]}
        else:
            spec["space"] = {"model__max_depth": [2, 3, 4],
                             "model__min_child_weight": [15, 30, 60],
                             "model__reg_lambda": [50, 150, 300],
                             "model__learning_rate": [0.01, 0.02, 0.03]}
    return sorted(out, key=lambda x: x["rank"])


def metrics(y, p):
    return {"R2": float(r2_score(y, p)) if len(y) >= 2 and np.ptp(y) > 0 else float("nan"), "MAE": float(mean_absolute_error(y, p)), "RMSE": float(math.sqrt(mean_squared_error(y, p)))}


def fit_oof_blend(fitted, cv_table, X, y, cv):
    """Fit at most three strong, diverse candidates using training-only OOF weights."""
    ordered = cv_table.sort_values(["CV_R2_Mean", "CV_Abs_Gap_Mean"], ascending=[False, True])
    names = ordered.Candidate.head(3).tolist()
    estimators = [fitted[name][0] for name in names]
    oof = np.zeros((len(y), len(names)), dtype=float)
    fold_train, fold_valid = [], []
    splits = list(cv.split(X, y))
    for fold, (tr, va) in enumerate(splits):
        train_matrix, valid_matrix = [], []
        for estimator0 in estimators:
            model = clone(estimator0).fit(X.iloc[tr], y[tr])
            train_matrix.append(model.predict(X.iloc[tr]))
            valid_matrix.append(model.predict(X.iloc[va]))
        train_matrix = np.column_stack(train_matrix)
        valid_matrix = np.column_stack(valid_matrix)
        oof[va] = valid_matrix
        fold_train.append((tr, train_matrix))
        fold_valid.append((va, valid_matrix))
    weights = np.linalg.lstsq(oof, y, rcond=None)[0]
    weights = np.clip(weights, 0.0, None)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.zeros(len(names)); weights[0] = 1.0
    else:
        weights /= weights.sum()
    audit = {"train_r2": [], "test_r2": [], "mae": [], "rmse": []}
    for (tr, train_matrix), (va, valid_matrix) in zip(fold_train, fold_valid):
        train_m, valid_m = metrics(y[tr], train_matrix @ weights), metrics(y[va], valid_matrix @ weights)
        audit["train_r2"].append(train_m["R2"]); audit["test_r2"].append(valid_m["R2"])
        audit["mae"].append(valid_m["MAE"]); audit["rmse"].append(valid_m["RMSE"])
    audit = {key: np.asarray(value, dtype=float) for key, value in audit.items()}
    blend = WeightedBlendRegressor(tuple(estimators), tuple(float(x) for x in weights), tuple(names)).fit(X, y)
    params = {"components": names, "weights": [float(x) for x in weights],
              "weight_fit": "nonnegative_training_oof_lstsq"}
    return blend, audit, params


def fit_cv(spec, X, y, cv):
    base = clone(spec["estimator"])
    if "n_jobs" in base.get_params(deep=False):
        base.set_params(n_jobs=1)
    pipe = Pipeline([("preprocessor", preprocessor()), ("model", base)])
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
    search = RandomizedSearchCV(pipe, spec["space"], n_iter=SEARCH_ITERATIONS, scoring=scoring, refit="r2", cv=cv, random_state=SEED, n_jobs=CV_JOBS, return_train_score=True, error_score="raise")
    search.fit(X, y)
    i, r = search.best_index_, search.cv_results_
    audit = {"train_r2": np.array([r[f"split{k}_train_r2"][i] for k in range(CV_FOLDS)]), "test_r2": np.array([r[f"split{k}_test_r2"][i] for k in range(CV_FOLDS)]), "mae": np.array([-r[f"split{k}_test_mae"][i] for k in range(CV_FOLDS)]), "rmse": np.array([-r[f"split{k}_test_rmse"][i] for k in range(CV_FOLDS)])}
    tuned = {k[len("param_"):]: r[k][i] for k in r if k.startswith("param_")}
    params = {k.removeprefix("model__"): str(v) for k, v in tuned.items()}
    return search.best_estimator_, audit, params


def select_model(table: pd.DataFrame) -> int:
    top = table["CV_R2_Mean"].max()
    pool = table[table["CV_R2_Mean"] >= top - 0.02].copy()
    if "CV_Abs_Gap_Mean" not in pool:
        pool["CV_Abs_Gap_Mean"] = pool.CV_Gap_Mean.abs()
    return int(pool.sort_values(["CV_Abs_Gap_Mean", "CV_R2_SD", "Complexity_Rank", "CV_R2_Mean"], ascending=[True, True, True, False]).index[0])


# ── Sub-step 2: conformalized quantile-GB uncertainty layer ─────────────────
QUANTILE_ALPHAS = (0.16, 0.84)
NOMINAL_COVERAGE = 0.68


def quantile_pipe(alpha: float) -> Pipeline:
    return Pipeline([
        ("preprocessor", preprocessor()),
        ("model", GradientBoostingRegressor(
            loss="quantile", alpha=alpha, n_estimators=400, learning_rate=0.03,
            max_depth=2, min_samples_leaf=10, subsample=0.85, random_state=SEED)),
    ])


def fit_uncertainty_layer() -> dict:
    """Calibrate the conformal uncertainty layer on the deployed blend.

    Fits a q16/q84 quantile-GB pair on the TRAINING split of the primary set
    and calibrates the bracketed-conformal multiplier k_br by 5-fold OOF within
    the training rows (68% nominal coverage):
        sd(x) = k_br * max(q84(x) - q16(x), 0) / 2
    k_br is the 68th percentile of |y - mu| / (width/2) over OOF points with
    positive quantile width, where mu comes from per-fold refits of the
    deployed blend itself (cloned with its locked hyperparameters).  Nothing
    outside the training split is ever used.
    """
    uncertainty_pkl = OUTPUT_DIR / "uncertainty_model.pkl"
    calibration_path = REPORT_DIR / "uncertainty_calibration.json"
    df = pd.read_csv(PRIMARY_DATA, encoding="utf-8-sig", low_memory=False)
    with (OUTPUT_DIR / "best_model.pkl").open("rb") as fh:
        bundle = pickle.load(fh)
    blend = bundle["pipeline"]
    train_idx = np.asarray(bundle["train_indices"], dtype=int)
    X, y = df.iloc[train_idx][list(MODEL_FEATURES)].copy(), df.iloc[train_idx][TARGET].astype(float).to_numpy()
    print(f"[unc] training rows {len(X)} x {X.shape[1]} features", flush=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    mu_oof = np.zeros(len(y)); q16_oof = np.zeros(len(y)); q84_oof = np.zeros(len(y))
    for fold, (tr, va) in enumerate(kf.split(X)):
        b = clone(blend).fit(X.iloc[tr], y[tr])
        lo = quantile_pipe(QUANTILE_ALPHAS[0]).fit(X.iloc[tr], y[tr])
        hi = quantile_pipe(QUANTILE_ALPHAS[1]).fit(X.iloc[tr], y[tr])
        mu_oof[va] = b.predict(X.iloc[va])
        q16_oof[va] = lo.predict(X.iloc[va])
        q84_oof[va] = hi.predict(X.iloc[va])
        print(f"[unc] fold {fold + 1}/5 done", flush=True)

    width = np.maximum(q84_oof - q16_oof, 0.0)
    bracketed = width > 0
    ratios = np.abs(y[bracketed] - mu_oof[bracketed]) / (width[bracketed] / 2.0)
    k_br = float(np.quantile(ratios, NOMINAL_COVERAGE))
    coverage = float(np.mean(np.abs(y[bracketed] - mu_oof[bracketed]) <= k_br * width[bracketed] / 2.0))
    print(f"[unc] bracketed points {int(bracketed.sum())}/{len(y)}, k_br={k_br:.4f}, "
          f"empirical coverage={coverage:.3f}", flush=True)

    q16_full = quantile_pipe(QUANTILE_ALPHAS[0]).fit(X, y)
    q84_full = quantile_pipe(QUANTILE_ALPHAS[1]).fit(X, y)
    meta = {
        "method": "conformalized quantile-GB pair (q16/q84), bracketed-conformal",
        "sd_formula": "k_br * max(q84 - q16, 0) / 2",
        "conformal_k_bracketed": k_br,
        "nominal_coverage": NOMINAL_COVERAGE,
        "empirical_oof_coverage": coverage,
        "n_calibration_rows": int(len(y)),
        "n_bracketed_rows": int(bracketed.sum()),
        "fold_scheme": "5-fold KFold(shuffle, seed=42) within the training split",
        "quantile_alphas": list(QUANTILE_ALPHAS),
        "blend_components": list(getattr(blend, "names", [])),
    }
    with uncertainty_pkl.open("wb") as fh:
        pickle.dump({"q16": q16_full, "q84": q84_full, "meta": meta}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    calibration_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


# ── Sub-step 3: per-row out-of-fold predictions for diagnostics ─────────────
def export_oof_layer() -> dict:
    """Refit the deployed pipeline out-of-fold over the full primary set.

    Writes one row per record (row_id, OOF_Predicted, OOF_Residual,
    Abs_OOF_Residual, Pred_Std) to output/tables/oof_predictions.csv.  Pred_Std
    reuses the frozen k_br so the column is consistent with the deployed
    uncertainty model.  Diagnostic output only: these are out-of-fold residuals
    over the whole primary set, which is a different quantity from the seed-42
    holdout residuals in holdout_predictions.csv.
    """
    output_csv = TABLE_DIR / "oof_predictions.csv"
    calibration = json.loads((REPORT_DIR / "uncertainty_calibration.json").read_text(encoding="utf-8"))
    k_br = float(calibration["conformal_k_bracketed"])

    df = pd.read_csv(PRIMARY_DATA, encoding="utf-8-sig", low_memory=False)
    with (OUTPUT_DIR / "best_model.pkl").open("rb") as fh:
        blend = pickle.load(fh)["pipeline"]
    X = df[list(MODEL_FEATURES)].copy()
    y = df[TARGET].astype(float).to_numpy()
    print(f"[oof] {len(X)} rows x {X.shape[1]} raw features, k_br={k_br:.4f}", flush=True)

    folds = 5
    mu = np.full(len(y), np.nan)
    width = np.full(len(y), np.nan)
    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        fitted = clone(blend).fit(X.iloc[tr], y[tr])
        lo = quantile_pipe(QUANTILE_ALPHAS[0]).fit(X.iloc[tr], y[tr])
        hi = quantile_pipe(QUANTILE_ALPHAS[1]).fit(X.iloc[tr], y[tr])
        mu[va] = fitted.predict(X.iloc[va])
        width[va] = np.maximum(hi.predict(X.iloc[va]) - lo.predict(X.iloc[va]), 0.0)
        print(f"[oof] fold {fold}/{folds} done", flush=True)

    residual = y - mu
    out = pd.DataFrame({
        "row_id": df["row_id"].to_numpy(),
        "OOF_Predicted": mu,
        "OOF_Residual": residual,
        "Abs_OOF_Residual": np.abs(residual),
        "Pred_Std": k_br * width / 2.0,
    })
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    meta = {
        "method": "out-of-fold refit of the deployed pipeline over the full primary set",
        "fold_scheme": f"{folds}-fold KFold(shuffle, seed={SEED}) over all primary rows",
        "pred_std_formula": f"k_br * max(q84 - q16, 0) / 2 with frozen k_br={k_br}",
        "n_rows": int(len(out)),
        "oof_r2": float(1 - np.sum(residual ** 2) / np.sum((y - y.mean()) ** 2)),
        "oof_mae": float(np.abs(residual).mean()),
        "mean_pred_std": float(out["Pred_Std"].mean()),
        "note": ("diagnostic only; out-of-fold over the whole primary set, which differs "
                 "from the seed-42 holdout residuals in holdout_predictions.csv"),
    }
    (output_csv.parent / "oof_predictions_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


# ── Sub-step 1: training and selection ───────────────────────────────────────
def main(stability_seeds: tuple[int, ...] = STABILITY_SEEDS) -> None:
    started = time.time()
    df = pd.read_csv(PRIMARY_DATA, encoding="utf-8-sig")
    manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("source_review_passed") is not True:
        raise ValueError("Training requires released source review")
    raw_path = Path(manifest["raw_csv"])
    verify_release(raw_path)
    raw_manifest = json.loads((raw_path.parent / "manifest.json").read_text(encoding="utf-8"))
    if raw_manifest.get("release_status") != "verified" or sha256(raw_path) != manifest["raw_sha256"]:
        raise ValueError("Raw release revoked or modified")
    for filename, expected_hash in raw_manifest.get("input_hashes", {}).items():
        if sha256(raw_path.parent / filename) != expected_hash:
            raise ValueError("Source review dependency modified: " + filename)
    if sha256(PRIMARY_DATA) != manifest["primary_sha256"]: raise RuntimeError("Primary dataset hash mismatch")
    missing = [c for c in MODEL_FEATURES + [TARGET, "row_id"] if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Feature schema version mismatch")
    leaked = FORBIDDEN_MODEL_FEATURES.intersection(MODEL_FEATURES)
    if leaked: raise ValueError(f"Audit metadata leaked into features: {sorted(leaked)}")
    X, y = df[MODEL_FEATURES].copy(), df[TARGET].astype(float).to_numpy()
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED, shuffle=True)
    Xtr, Xte, ytr, yte = X.iloc[train_idx], X.iloc[test_idx], y[train_idx], y[test_idx]
    split = df[["row_id", "Paper_ID", "Source_File", "Canonical_Catalyst_ID", "Catalyst", TARGET]].copy()
    split["Set"] = "Train"; split.loc[test_idx, "Set"] = "Test"
    split.to_csv(TABLE_DIR / "split_manifest.csv", index=False, encoding="utf-8-sig")
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    rows, folds, fitted = [], [], {}
    model_specs = specs()
    for spec in model_specs:
        print(f"Training-only CV: seed={SEED}, model={spec['model']}",flush=True)
        pipe, audit, params = fit_cv(spec, Xtr, ytr, cv)
        train_m = metrics(ytr, pipe.predict(Xtr))
        # Holdout is not evaluated while candidates or parameters are selected.
        test_m = dict(R2=np.nan, MAE=np.nan, RMSE=np.nan)
        gap = audit["train_r2"] - audit["test_r2"]
        for k in range(CV_FOLDS): folds.append({"Candidate": spec["candidate"], "Fold": k + 1, "Train_R2": audit["train_r2"][k], "Validation_R2": audit["test_r2"][k], "CV_Gap": gap[k], "Validation_MAE": audit["mae"][k], "Validation_RMSE": audit["rmse"][k]})
        fitted[spec["candidate"]] = (pipe, spec, params)
        rows.append({"Candidate": spec["candidate"], "Model": spec["model"], "Feature_Set": "cleaned_no_metadata", "Complexity_Rank": spec["rank"], "CV_R2_Mean": audit["test_r2"].mean(), "CV_R2_SD": audit["test_r2"].std(ddof=1), "CV_Train_R2_Mean": audit["train_r2"].mean(), "CV_Gap_Mean": gap.mean(), "CV_Gap_SD": gap.std(ddof=1), "CV_MAE_Mean": audit["mae"].mean(), "CV_RMSE_Mean": audit["rmse"].mean(), "Train_R2": train_m["R2"], "Train_MAE": train_m["MAE"], "Train_RMSE": train_m["RMSE"], "Test_R2": test_m["R2"], "Test_MAE": test_m["MAE"], "Test_RMSE": test_m["RMSE"], "Train_Test_R2_Gap": train_m["R2"] - test_m["R2"], "Best_Params": json.dumps(params, ensure_ascii=False, sort_keys=True)})
    base_comparison = pd.DataFrame(rows)
    fold_table = pd.DataFrame(folds)
    base_comparison["CV_Abs_Gap_Mean"] = base_comparison.Candidate.map(
        fold_table.assign(Abs_Gap=lambda d:d.CV_Gap.abs()).groupby("Candidate").Abs_Gap.mean())
    blend, blend_audit, blend_params = fit_oof_blend(fitted, base_comparison, Xtr, ytr, cv)
    blend_gap = blend_audit["train_r2"] - blend_audit["test_r2"]
    for k in range(CV_FOLDS):
        folds.append({"Candidate": "OOF Weighted Blend", "Fold": k + 1,
                      "Train_R2": blend_audit["train_r2"][k], "Validation_R2": blend_audit["test_r2"][k],
                      "CV_Gap": blend_gap[k], "Validation_MAE": blend_audit["mae"][k],
                      "Validation_RMSE": blend_audit["rmse"][k]})
    blend_train = metrics(ytr, blend.predict(Xtr))
    fitted["OOF Weighted Blend"] = (blend, None, blend_params)
    rows.append({"Candidate":"OOF Weighted Blend", "Model":"OOF Weighted Blend",
                 "Feature_Set":"cleaned_no_metadata", "Complexity_Rank":len(model_specs)+1,
                 "CV_R2_Mean":blend_audit["test_r2"].mean(), "CV_R2_SD":blend_audit["test_r2"].std(ddof=1),
                 "CV_Train_R2_Mean":blend_audit["train_r2"].mean(), "CV_Gap_Mean":blend_gap.mean(),
                 "CV_Gap_SD":blend_gap.std(ddof=1), "CV_MAE_Mean":blend_audit["mae"].mean(),
                 "CV_RMSE_Mean":blend_audit["rmse"].mean(), "Train_R2":blend_train["R2"],
                 "Train_MAE":blend_train["MAE"], "Train_RMSE":blend_train["RMSE"],
                 "Test_R2":np.nan, "Test_MAE":np.nan, "Test_RMSE":np.nan,
                 "Train_Test_R2_Gap":np.nan,
                 "Best_Params":json.dumps(blend_params,ensure_ascii=False,sort_keys=True)})
    comparison = pd.DataFrame(rows)
    comparison["CV_Abs_Gap_Mean"] = comparison.Candidate.map(pd.DataFrame(folds).assign(Abs_Gap=lambda d:d.CV_Gap.abs()).groupby("Candidate").Abs_Gap.mean())
    selected_i = select_model(comparison); comparison["Selected"] = False; comparison.loc[selected_i, "Selected"] = True; comparison = comparison.sort_values(["Selected", "CV_R2_Mean"], ascending=[False, False]).reset_index(drop=True); comparison.insert(1, "CV_Rank", comparison["CV_R2_Mean"].rank(ascending=False, method="min").astype(int))
    comparison.to_csv(TABLE_DIR / "locked_training_selection.csv", index=False, encoding="utf-8-sig")
    for i, row in comparison.iterrows():
        result = metrics(yte, fitted[row["Candidate"]][0].predict(Xte))
        for metric, value in result.items():
            comparison.loc[i, "Test_" + metric] = value
        comparison.loc[i, "Train_Test_R2_Gap"] = row["Train_R2"] - result["R2"]
        comparison.loc[i, "Abs_Train_Test_R2_Gap"] = abs(row["Train_R2"] - result["R2"])
    comparison.to_csv(TABLE_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig"); pd.DataFrame(folds).to_csv(TABLE_DIR / "row_random_cv_fold_results.csv", index=False, encoding="utf-8-sig")
    chosen = comparison.loc[comparison["Selected"]].iloc[0]; name = chosen["Candidate"]; pipe = fitted[name][0]
    pred = np.empty(len(df)); pred[train_idx] = pipe.predict(Xtr); pred[test_idx] = pipe.predict(Xte)
    holdout = df[["row_id", "Paper_ID", "Source_File", "Canonical_Catalyst_ID", "Catalyst"]].copy(); holdout["Set"] = "Train"; holdout.loc[test_idx, "Set"] = "Test"; holdout["Actual"] = y; holdout["Predicted"] = pred; holdout["Residual"] = pred - y; holdout["Abs_Residual"] = abs(pred - y); holdout.to_csv(TABLE_DIR / "holdout_predictions.csv", index=False, encoding="utf-8-sig")
    stability, subtype_results = [], []
    for s in stability_seeds:
        tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=s, shuffle=True)
        seed_models, parameter_log = {}, []
        if s == SEED:
            for candidate_name, (p, _, params) in fitted.items():
                seed_models[candidate_name] = p
                parameter_log.append({"Candidate":candidate_name,"Split_Seed":s,
                                      "Best_Params":json.dumps(params,sort_keys=True)})
        else:
            seed_cv_rows=[]
            seed_fitted={}
            seed_cv=KFold(n_splits=CV_FOLDS,shuffle=True,random_state=s)
            for spec in model_specs:
                print(f"Stability training-only CV: seed={s}, model={spec['model']}",flush=True)
                p, audit, params = fit_cv(spec, X.iloc[tr], y[tr], seed_cv)
                seed_models[spec["candidate"]] = p
                seed_fitted[spec["candidate"]]=(p,spec,params)
                gaps=audit["train_r2"]-audit["test_r2"]
                seed_cv_rows.append(dict(Candidate=spec["candidate"],CV_R2_Mean=audit["test_r2"].mean(),
                                         CV_Abs_Gap_Mean=np.abs(gaps).mean()))
                parameter_log.append({"Candidate":spec["candidate"],"Split_Seed":s,
                                      "Best_Params":json.dumps(params,sort_keys=True)})
            seed_blend, _, seed_blend_params=fit_oof_blend(
                seed_fitted,pd.DataFrame(seed_cv_rows),X.iloc[tr],y[tr],seed_cv)
            seed_models["OOF Weighted Blend"]=seed_blend
            parameter_log.append({"Candidate":"OOF Weighted Blend","Split_Seed":s,
                                  "Best_Params":json.dumps(seed_blend_params,sort_keys=True)})
        pd.DataFrame(parameter_log).to_csv(TABLE_DIR / f"locked_parameters_seed_{s}.csv", index=False)
        seed_split = df[["row_id"]].copy()
        seed_split["Set"] = "Train"
        seed_split.loc[te, "Set"] = "Test"
        seed_split.to_csv(TABLE_DIR / f"split_seed_{s}.csv", index=False)
        for candidate_name, p in seed_models.items():
            predicted = p.predict(X)
            mt, mv = metrics(y[tr], predicted[tr]), metrics(y[te], predicted[te])
            for reaction in ("MSR", "APRM"):
                subset = te[df.iloc[te].Reaction_Type.eq(reaction).to_numpy()]
                values = metrics(y[subset], predicted[subset]) if len(subset) else dict(R2=np.nan, MAE=np.nan, RMSE=np.nan)
                subtype_results.append(dict(Candidate=candidate_name, Split_Seed=s, Reaction_Type=reaction, n=len(subset), **values))
            stability.append({"Candidate": candidate_name, "Split_Seed": s, "Train_n": len(tr), "Test_n": len(te),
                              **{"Train_" + k:v for k,v in mt.items()}, **{"Test_" + k:v for k,v in mv.items()},
                              "Train_Test_R2_Gap": mt["R2"] - mv["R2"]})
    all_st = pd.DataFrame(stability)
    pd.DataFrame(subtype_results).to_csv(TABLE_DIR / "reaction_type_test_metrics.csv", index=False)
    all_st["Abs_Train_Test_R2_Gap"] = all_st.Train_Test_R2_Gap.abs()
    all_st.to_csv(TABLE_DIR / "all_models_seed_stability.csv", index=False)
    _sum = all_st.groupby("Candidate")[["Train_R2", "Test_R2", "Train_MAE", "Train_RMSE", "Test_MAE", "Test_RMSE", "Abs_Train_Test_R2_Gap"]].agg(["mean", "std"])
    _sum.columns = pd.MultiIndex.from_tuples(_sum.columns)
    _sum.to_csv(TABLE_DIR / "all_models_seed_summary.csv")
    st = all_st.loc[all_st.Candidate.eq(name)].copy()
    st["Abs_Train_Test_R2_Gap"] = st["Train_Test_R2_Gap"].abs()
    st.to_csv(TABLE_DIR / "random_seed_stability.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([{**{"Metric": col}, **{"Mean": st[col].mean(), "SD": st[col].std(ddof=1), "Median": st[col].median(), "Min": st[col].min(), "Max": st[col].max()}} for col in ["Train_R2", "Test_R2", "Test_MAE", "Test_RMSE", "Train_Test_R2_Gap"]]); summary.to_csv(TABLE_DIR / "random_seed_summary.csv", index=False, encoding="utf-8-sig")
    sizes, trc, vac = learning_curve(clone(pipe), Xtr, ytr, cv=cv, scoring="r2", train_sizes=np.array([0.2, 0.35, 0.5, 0.7, 0.85, 1.0]), n_jobs=CV_JOBS, shuffle=True, random_state=SEED); pd.DataFrame({"Train_Size": sizes, "Train_R2_Mean": trc.mean(1), "Train_R2_SD": trc.std(1, ddof=1), "Validation_R2_Mean": vac.mean(1), "Validation_R2_SD": vac.std(1, ddof=1)}).to_csv(TABLE_DIR / "learning_curve.csv", index=False, encoding="utf-8-sig")
    final = clone(pipe).fit(X, y)
    if isinstance(final, Pipeline):
        pre = final.named_steps["preprocessor"]
        fitted_model = final.named_steps["model"]
        names = pre.get_feature_names_out().tolist()
        np.save(ARTIFACT_DIR / "X_full.npy", pre.transform(X))
        dump(pre, ARTIFACT_DIR / "preprocessor.joblib")
        evaluation_pre = pipe.named_steps["preprocessor"]
        evaluation_model = pipe.named_steps["model"]
        evaluation_names = evaluation_pre.get_feature_names_out().tolist()
    else:
        fitted_model = final
        # A bare blend still exposes its canonical fitted preprocessor through
        # any member pipeline; persist the transformed matrix so downstream
        # consumers (t-SNE, SHAP audits) never depend on the winning model type.
        pre = final.estimators_[0].named_steps["preprocessor"]
        names = pre.get_feature_names_out().tolist()
        np.save(ARTIFACT_DIR / "X_full.npy", pre.transform(X))
        dump(pre, ARTIFACT_DIR / "preprocessor.joblib")
        evaluation_pre = pipe.estimators_[0].named_steps["preprocessor"]
        evaluation_model = pipe
        evaluation_names = evaluation_pre.get_feature_names_out().tolist()
        pd.DataFrame({"Component": final.names, "Weight": final.weights}).to_csv(
            TABLE_DIR / "blend_components.csv", index=False, encoding="utf-8-sig")
    (ARTIFACT_DIR / "feature_names.txt").write_text("\n".join(names), encoding="utf-8")
    bundle = {"version": "pure_row_random_v2", "feature_schema_version":FEATURE_SCHEMA_VERSION,
              "data_sha256": manifest["primary_sha256"], "candidate": name, "pipeline": final,
              "preprocessor": pre, "model": fitted_model, "feat_names": names,
              "raw_feature_names": MODEL_FEATURES, "split_strategy": "pure_row_random",
              "selection_basis": "training_only_5fold_row_random_cv", "train_indices": train_idx,
              "test_indices": test_idx}
    with (OUTPUT_DIR / "evaluation_model.pkl").open("wb") as f: pickle.dump({**bundle, "pipeline": pipe, "preprocessor": evaluation_pre, "model": evaluation_model, "feat_names": evaluation_names}, f, protocol=pickle.HIGHEST_PROTOCOL)
    with (OUTPUT_DIR / "best_model.pkl").open("wb") as f: pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    imp = getattr(fitted_model, "feature_importances_", np.full(len(names), np.nan))
    pd.DataFrame({"Feature": names, "Importance": imp}).sort_values("Importance", ascending=False).to_csv(TABLE_DIR / "model_feature_importance.csv", index=False, encoding="utf-8-sig")
    test_row = summary.loc[summary["Metric"] == "Test_R2"].iloc[0]; gap_row = summary.loc[summary["Metric"] == "Train_Test_R2_Gap"].iloc[0]
    report = {"report_kind": "descriptive_performance_summary",
              "selected_candidate": name, "split_strategy": "pure_row_random",
              "feature_schema_version":FEATURE_SCHEMA_VERSION,
              "base_models_completed":len(model_specs), "blend_evaluated":True,
              "cv_r2_mean": float(chosen["CV_R2_Mean"]), "test_r2_seed42": float(chosen["Test_R2"]),
              "train_test_gap_seed42": float(chosen["Train_Test_R2_Gap"]),
              "multiseed_test_r2_mean": float(test_row["Mean"]),
              "multiseed_test_r2_sd": float(test_row["SD"]),
              "multiseed_gap_mean": float(gap_row["Mean"]),
              "multiseed_gap_sd": float(gap_row["SD"]),
              "stability_seeds": list(stability_seeds)}
    report.update(train_r2_seed42=float(chosen["Train_R2"]),
                  source_raw_sha256=manifest["raw_sha256"],primary_sha256=manifest["primary_sha256"],
                  multiseed_train_r2_mean=float(st.Train_R2.mean()),
                  multiseed_abs_gap_mean=float(st.Abs_Train_Test_R2_Gap.mean()),
                  multiseed_test_r2_by_seed=[float(v) for v in st.Test_R2])
    (REPORT_DIR / "performance_gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORT_DIR / "methodology_summary.md").write_text(
        "# Pure row-random evaluation\n\n"
        "Eight base models and one training-OOF weighted blend evaluated by five-fold training-only random CV. Parameters and selection "
        "saved before holdout evaluation. Splits: 80/20, seeds 42-51, no source stratification. "
        "The existing corpus has previously participated in development; reseeding is not new independent evidence.\n\n"
        + json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[train] total wall time {time.time() - started:.0f}s", flush=True)

    # Sub-steps 2 and 3: uncertainty layer + OOF diagnostic export.
    fit_uncertainty_layer()
    export_oof_layer()


def _parse_seeds(text: str) -> tuple[int, ...]:
    """Parse a seed list like "42", "42,43,44" or "42-51"."""
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("No seeds parsed from --seeds")
    return tuple(seeds)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Train/select the conversion model, fit the uncertainty layer, and export OOF diagnostics.")
    parser.add_argument("--seeds", type=str, default="42-51",
                        help="split seeds for the stability table, e.g. '42' or '42-51' (default: 42-51)")
    parser.add_argument("--input", type=Path, help="override the primary modeling table")
    parser.add_argument("--output-dir", type=Path, help="override the output directory")
    args = parser.parse_args()
    if args.input:
        if not args.output_dir:
            parser.error("--output-dir required with --input")
        PRIMARY_DATA = args.input.resolve()
        DATASET_MANIFEST = PRIMARY_DATA.parent / "dataset_manifest.json"
        info = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
        if info.get("source_review_passed") is not True:
            raise ValueError("Source review gate has not passed")
    if args.output_dir:
        OUTPUT_DIR = args.output_dir.resolve()
        TABLE_DIR, REPORT_DIR, ARTIFACT_DIR = [OUTPUT_DIR / x for x in ("tables", "reports", "artifacts")]
        for d in (TABLE_DIR, REPORT_DIR, ARTIFACT_DIR):
            d.mkdir(parents=True, exist_ok=True)
    main(_parse_seeds(args.seeds))
