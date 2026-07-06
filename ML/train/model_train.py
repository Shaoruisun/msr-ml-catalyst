"""Source-stratified row-random model training for the v3 cleaned dataset.

This branch keeps the simple row-random ML framing but stratifies the 80/20
split by Source_File so the holdout has a comparable literature-source mix.
Source_File is used only for splitting and never enters model features.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
PRIMARY_DATA = ML_ROOT / "data_clean" / "prep_output" / "ni_model_primary.csv"
DATASET_MANIFEST = ML_ROOT / "data_clean" / "prep_output" / "dataset_manifest.json"
TRAIN_DIR = SCRIPT_DIR
TABLE_DIR = TRAIN_DIR / "tables"
REPORT_DIR = TRAIN_DIR / "reports"
ARTIFACT_DIR = TRAIN_DIR / "artifacts"
REGULARIZED_TABLES = TABLE_DIR
for directory in (TRAIN_DIR, TABLE_DIR, REPORT_DIR, ARTIFACT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 42
STABILITY_SEEDS = tuple(range(42, 52))
TEST_SIZE = 0.20
CV_FOLDS = 5
SEARCH_ITERATIONS = 50
TARGET = "MeOH_Conversion_%"

NUMERIC_FEATURES = [
    "Reaction_Temp_C",
    "Ni_Loading_wt%",
    "Promoter_Loading_wt%",
    "Calcination_Temp_C",
    "Reduction_Temp_C",
    "Dry_Temp_C",
    "Calcination_Time_h",
    "Reduction_Time_h",
    "Dry_Time_h",
    "S_C_Ratio",
    "TOS_h",
    "Pressure_bar",
    "GHSV_log",
    "WHSV_log",
    "Catalyst_Label_Ni_Value",
    "calc_reduc_temp_diff",
    "total_treatment_time_h",
    "prom_electronegativity",
    "prom_ionic_radius",
    "prom_atomic_mass",
    "prom_first_ionization",
    "support_acid_base",
]

CATEGORICAL_FEATURES = [
    "Support",
    "Promoter_Metal",
    "Metal_Loading_Method",
    "Precursor_Family",
    "Support_Prep_Method",
    "support_type",
]

BINARY_FEATURES = [
    "has_promoter",
    "prom_missing",
    "support_reducible",
]

ONEHOT_MIN_FREQUENCY = 5
SOURCE_STRATIFY_MIN_ROWS = 4
NUMERIC_FEATURES_SS = [x for x in NUMERIC_FEATURES if x != "Catalyst_Label_Ni_Value"]
CATEGORICAL_FEATURES_SS = list(CATEGORICAL_FEATURES)
BINARY_FEATURES_SS = list(BINARY_FEATURES)
MODEL_FEATURES_SS = NUMERIC_FEATURES_SS + CATEGORICAL_FEATURES_SS + BINARY_FEATURES_SS
FEATURE_SET_NAME = "no_label_ni_mf5_no_num_indicators"
STANDARD_MODEL_ORDER = {
    "SVR": 1,
    "Random Forest": 2,
    "Gradient Boosting": 3,
    "XGBoost": 4,
    "AdaBoost": 5,
    "Bagging": 6,
    "Extra Trees": 7,
}
REFERENCE_MODEL_MAP = {
    "Random Forest": "RandomForest_regularized",
    "Gradient Boosting": "GradientBoosting_regularized",
    "XGBoost": "XGBoost_regularized",
    "Extra Trees": "ExtraTrees_regularized",
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_strata(source: pd.Series) -> pd.Series:
    counts = source.value_counts(dropna=False)
    return source.where(source.map(counts) >= SOURCE_STRATIFY_MIN_ROWS, "rare_sources")


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
                ("scaler", StandardScaler()),
            ]), NUMERIC_FEATURES_SS),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=ONEHOT_MIN_FREQUENCY)),
            ]), CATEGORICAL_FEATURES_SS),
            ("bin", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), BINARY_FEATURES_SS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, DecisionTreeRegressor):
        return {
            "DecisionTreeRegressor": {
                "max_depth": value.max_depth,
                "min_samples_leaf": value.min_samples_leaf,
                "random_state": value.random_state,
            }
        }
    return value


def clean_params(params: dict[str, object]) -> dict[str, object]:
    out = {}
    for key, value in params.items():
        clean_key = key.removeprefix("model__")
        out[clean_key] = json_safe(value)
    return out


def regularized_reference_rows() -> dict[str, pd.Series]:
    audit = pd.read_csv(REGULARIZED_TABLES / "regularized_model_comparison.csv")
    audit = audit[audit["Feature_Set"].eq(FEATURE_SET_NAME)].copy()
    if audit.empty:
        raise RuntimeError(f"No regularized audit rows for feature set {FEATURE_SET_NAME}")
    refs: dict[str, pd.Series] = {}
    for public_name, audit_name in REFERENCE_MODEL_MAP.items():
        match = audit[audit["Model"].eq(audit_name)].sort_values("CV_R2_Mean", ascending=False)
        if match.empty:
            raise RuntimeError(f"No regularized audit row for {audit_name}")
        refs[public_name] = match.iloc[0]
    return refs


def fixed_estimator(public_name: str, params: dict[str, object]):
    if public_name == "Random Forest":
        return RandomForestRegressor(random_state=SEED, n_jobs=1, **params)
    if public_name == "Extra Trees":
        return ExtraTreesRegressor(random_state=SEED, n_jobs=1, **params)
    if public_name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=SEED, **params)
    if public_name == "XGBoost":
        return XGBRegressor(random_state=SEED, n_jobs=1, objective="reg:squarederror", verbosity=0, **params)
    raise ValueError(f"Unsupported fixed model name: {public_name}")


def tuning_space(public_name: str) -> tuple[object, dict[str, list[object]]]:
    if public_name == "SVR":
        return SVR(kernel="rbf"), {
            "model__C": [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
            "model__epsilon": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            "model__gamma": ["scale", "auto", 0.003, 0.01, 0.03, 0.1, 0.3],
        }
    if public_name == "AdaBoost":
        trees = [
            DecisionTreeRegressor(max_depth=d, min_samples_leaf=leaf, random_state=SEED)
            for d in (2, 3, 4, 5)
            for leaf in (3, 5, 10, 15)
        ]
        return AdaBoostRegressor(random_state=SEED), {
            "model__estimator": trees,
            "model__n_estimators": [100, 200, 400, 700],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "model__loss": ["linear", "square", "exponential"],
        }
    if public_name == "Bagging":
        trees = [
            DecisionTreeRegressor(max_depth=d, min_samples_leaf=leaf, random_state=SEED)
            for d in (4, 6, 8, None)
            for leaf in (2, 5, 10, 15)
        ]
        return BaggingRegressor(random_state=SEED, n_jobs=1), {
            "model__estimator": trees,
            "model__n_estimators": [100, 200, 400, 700],
            "model__max_samples": [0.55, 0.70, 0.85, 1.0],
            "model__max_features": [0.45, 0.60, 0.80, 1.0],
            "model__bootstrap": [True],
        }
    raise ValueError(f"Unsupported tuned model name: {public_name}")


def model_specs() -> list[dict[str, object]]:
    refs = regularized_reference_rows()
    specs: list[dict[str, object]] = []
    for public_name in ["Random Forest", "Gradient Boosting", "XGBoost", "Extra Trees"]:
        row = refs[public_name]
        params = json.loads(row["Best_Params"])
        specs.append({
            "candidate": public_name,
            "model_name": public_name,
            "complexity_rank": STANDARD_MODEL_ORDER[public_name],
            "regularized_cv_r2": float(row["CV_R2_Mean"]),
            "regularized_gap": float(row["Train_Test_R2_Gap"]),
            "params": params,
            "estimator": fixed_estimator(public_name, params),
            "tune": False,
            "search_space": None,
        })
    for public_name in ["SVR", "AdaBoost", "Bagging"]:
        estimator, search_space = tuning_space(public_name)
        specs.append({
            "candidate": public_name,
            "model_name": public_name,
            "complexity_rank": STANDARD_MODEL_ORDER[public_name],
            "regularized_cv_r2": np.nan,
            "regularized_gap": np.nan,
            "params": {},
            "estimator": estimator,
            "tune": True,
            "search_space": search_space,
        })
    return sorted(specs, key=lambda x: x["complexity_rank"])


def fit_and_audit(spec: dict[str, object], X_train: pd.DataFrame, y_train: np.ndarray, cv: KFold) -> tuple[Pipeline, dict[str, np.ndarray], dict[str, object]]:
    pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", spec["estimator"])])
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}
    if spec.get("tune"):
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=spec["search_space"],
            n_iter=SEARCH_ITERATIONS,
            scoring=scoring,
            refit="r2",
            cv=cv,
            random_state=SEED,
            n_jobs=-1,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best = int(search.best_index_)
        results = search.cv_results_
        audit = {
            "train_r2": np.array([results[f"split{i}_train_r2"][best] for i in range(CV_FOLDS)], dtype=float),
            "test_r2": np.array([results[f"split{i}_test_r2"][best] for i in range(CV_FOLDS)], dtype=float),
            "test_mae": np.array([results[f"split{i}_test_mae"][best] for i in range(CV_FOLDS)], dtype=float),
            "test_rmse": np.array([results[f"split{i}_test_rmse"][best] for i in range(CV_FOLDS)], dtype=float),
        }
        return search.best_estimator_, audit, clean_params(search.best_params_)
    audit = cross_validate(
        clone(pipeline),
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
        error_score="raise",
    )
    pipeline.fit(X_train, y_train)
    return pipeline, audit, clean_params(spec["params"])


def metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "R2": float(r2_score(y_true, pred)),
        "MAE": float(mean_absolute_error(y_true, pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y_true, pred))),
    }


def select_model(comparison: pd.DataFrame) -> int:
    """Select only by training-CV evidence; use gap/SD only within a 0.02 CV-R2 window."""
    eligible = comparison.copy()
    max_cv = eligible["CV_R2_Mean"].max()
    pool = eligible[eligible["CV_R2_Mean"] >= max_cv - 0.02].copy()
    pool = pool.sort_values(["CV_Gap_Mean", "CV_R2_SD", "Complexity_Rank", "CV_R2_Mean"], ascending=[True, True, True, False])
    return int(pool.index[0])


def seed_summary(stability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric_name in ("Train_R2", "Test_R2", "Test_MAE", "Test_RMSE", "Train_Test_R2_Gap"):
        values = stability[metric_name]
        rows.append({
            "Metric": metric_name,
            "Mean": values.mean(),
            "SD": values.std(ddof=1),
            "Median": values.median(),
            "Min": values.min(),
            "Max": values.max(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    started = time.time()
    df = pd.read_csv(PRIMARY_DATA, encoding="utf-8-sig")
    manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    if file_sha256(PRIMARY_DATA) != manifest["primary_sha256"]:
        raise RuntimeError("Primary data hash differs from dataset_manifest.json")
    missing = [c for c in [*MODEL_FEATURES_SS, TARGET, "row_id", "Source_File"] if c not in df.columns]
    if missing:
        raise ValueError(f"Primary dataset is missing required columns: {missing}")

    X = df[MODEL_FEATURES_SS].copy()
    y = df[TARGET].astype(float).to_numpy()
    all_idx = np.arange(len(df))
    strata = source_strata(df["Source_File"])
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=strata,
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    split_manifest = df[["row_id", "Source_File", "Canonical_Catalyst_ID", "Catalyst", TARGET]].copy()
    split_manifest["Source_Stratum"] = strata
    split_manifest["Set"] = ""
    split_manifest.loc[train_idx, "Set"] = "Train"
    split_manifest.loc[test_idx, "Set"] = "Test"
    split_manifest.to_csv(TRAIN_DIR / "split_manifest.csv", index=False, encoding="utf-8-sig")
    split_manifest.to_csv(TABLE_DIR / "split_manifest.csv", index=False, encoding="utf-8-sig")

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    tuned: dict[str, dict] = {}
    fold_rows: list[dict] = []
    comparison_rows: list[dict] = []

    for spec in model_specs():
        candidate = str(spec["candidate"])
        pipeline, audit, best_params = fit_and_audit(spec, X_train, y_train, cv)
        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)
        train_m, test_m = metrics(y_train, train_pred), metrics(y_test, test_pred)
        cv_gap = audit["train_r2"] - audit["test_r2"]
        for fold in range(CV_FOLDS):
            fold_rows.append({
                "Candidate": candidate,
                "Model": spec["model_name"],
                "Fold": fold + 1,
                "Train_R2": float(audit["train_r2"][fold]),
                "Validation_R2": float(audit["test_r2"][fold]),
                "CV_Gap": float(cv_gap[fold]),
                "Validation_MAE": float(-audit["test_mae"][fold]),
                "Validation_RMSE": float(-audit["test_rmse"][fold]),
            })
        tuned[candidate] = {"pipeline": pipeline, "spec": spec, "audit": audit, "best_params": best_params}
        comparison_rows.append({
            "Candidate": candidate,
            "Model": spec["model_name"],
            "Feature_Set": FEATURE_SET_NAME,
            "Complexity_Rank": spec["complexity_rank"],
            "CV_R2_Mean": float(np.mean(audit["test_r2"])),
            "CV_R2_SD": float(np.std(audit["test_r2"], ddof=1)),
            "CV_Train_R2_Mean": float(np.mean(audit["train_r2"])),
            "CV_Gap_Mean": float(np.mean(cv_gap)),
            "CV_Gap_SD": float(np.std(cv_gap, ddof=1)),
            "CV_MAE_Mean": float(np.mean(-audit["test_mae"])),
            "CV_RMSE_Mean": float(np.mean(-audit["test_rmse"])),
            "Train_R2": train_m["R2"],
            "Train_MAE": train_m["MAE"],
            "Train_RMSE": train_m["RMSE"],
            "Test_R2": test_m["R2"],
            "Test_MAE": test_m["MAE"],
            "Test_RMSE": test_m["RMSE"],
            "Train_Test_R2_Gap": train_m["R2"] - test_m["R2"],
            "Regularized_Branch_CV_R2": spec["regularized_cv_r2"],
            "Regularized_Branch_Gap": spec["regularized_gap"],
            "Best_Params": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
        })

    comparison = pd.DataFrame(comparison_rows)
    selected_index = select_model(comparison)
    comparison["Selected_Source_Stratified"] = False
    comparison.loc[selected_index, "Selected_Source_Stratified"] = True
    comparison = comparison.sort_values(["Selected_Source_Stratified", "CV_R2_Mean", "CV_Gap_Mean"], ascending=[False, False, True]).reset_index(drop=True)
    comparison.insert(1, "CV_Rank", comparison["CV_R2_Mean"].rank(ascending=False, method="min").astype(int))
    comparison.to_csv(TABLE_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(TABLE_DIR / "source_stratified_model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(TABLE_DIR / "cv_fold_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(TABLE_DIR / "source_stratified_cv_fold_results.csv", index=False, encoding="utf-8-sig")

    selected_row = comparison.loc[comparison["Selected_Source_Stratified"]].iloc[0]
    selected_candidate = selected_row["Candidate"]
    review_path = REPORT_DIR / "selection_review_required.json"
    if selected_row["Model"] != "Gradient Boosting":
        review = {
            "status": "manual_review_required",
            "reason": "Training-CV model selection did not select Gradient Boosting, so final model artifacts were not overwritten.",
            "selected_candidate": str(selected_row["Candidate"]),
            "selected_model": str(selected_row["Model"]),
            "selected_cv_r2_mean": float(selected_row["CV_R2_Mean"]),
            "selected_test_r2_audit_only": float(selected_row["Test_R2"]),
            "gradient_boosting_rows": comparison.loc[comparison["Model"].eq("Gradient Boosting")].to_dict(orient="records"),
            "required_decision": "Accept the new selected model and rerun SHAP/PSO, or keep Gradient Boosting as the final model.",
        }
        review_path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError(
            "CV-selected model is not Gradient Boosting; wrote model comparison tables and stopped before overwriting best_model.pkl."
        )
    if review_path.exists():
        review_path.unlink()
    selected_pipe = tuned[selected_candidate]["pipeline"]
    selected_spec = tuned[selected_candidate]["spec"]
    pred_train = selected_pipe.predict(X_train)
    pred_test = selected_pipe.predict(X_test)
    holdout = pd.concat([
        df.iloc[train_idx][["row_id", "Source_File", "Canonical_Catalyst_ID", "Catalyst"]].assign(Set="Train", Actual=y_train, Predicted=pred_train),
        df.iloc[test_idx][["row_id", "Source_File", "Canonical_Catalyst_ID", "Catalyst"]].assign(Set="Test", Actual=y_test, Predicted=pred_test),
    ], ignore_index=True)
    holdout["Residual"] = holdout["Predicted"] - holdout["Actual"]
    holdout["Abs_Residual"] = holdout["Residual"].abs()
    holdout.to_csv(TABLE_DIR / "holdout_predictions.csv", index=False, encoding="utf-8-sig")

    stability_rows = []
    for split_seed in STABILITY_SEEDS:
        tr, te = train_test_split(
            all_idx,
            test_size=TEST_SIZE,
            random_state=split_seed,
            shuffle=True,
            stratify=strata,
        )
        pipe = clone(selected_pipe)
        pipe.fit(X.iloc[tr], y[tr])
        p_tr = pipe.predict(X.iloc[tr])
        p_te = pipe.predict(X.iloc[te])
        m_tr, m_te = metrics(y[tr], p_tr), metrics(y[te], p_te)
        stability_rows.append({
            "Split_Seed": split_seed,
            "Train_n": len(tr),
            "Test_n": len(te),
            "Train_Sources": int(df.iloc[tr]["Source_File"].nunique()),
            "Test_Sources": int(df.iloc[te]["Source_File"].nunique()),
            "Train_R2": m_tr["R2"],
            "Test_R2": m_te["R2"],
            "Test_MAE": m_te["MAE"],
            "Test_RMSE": m_te["RMSE"],
            "Train_Test_R2_Gap": m_tr["R2"] - m_te["R2"],
        })
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(TABLE_DIR / "random_seed_stability.csv", index=False, encoding="utf-8-sig")
    summary = seed_summary(stability)
    summary.to_csv(TABLE_DIR / "random_seed_summary.csv", index=False, encoding="utf-8-sig")

    train_sizes, lc_train, lc_val = learning_curve(
        clone(selected_pipe),
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
        train_sizes=np.array([0.20, 0.35, 0.50, 0.70, 0.85, 1.0]),
        n_jobs=-1,
        shuffle=True,
        random_state=SEED,
        error_score="raise",
    )
    lc = pd.DataFrame({
        "Train_Size": train_sizes,
        "Train_R2_Mean": lc_train.mean(axis=1),
        "Train_R2_SD": lc_train.std(axis=1, ddof=1),
        "Validation_R2_Mean": lc_val.mean(axis=1),
        "Validation_R2_SD": lc_val.std(axis=1, ddof=1),
    })
    lc.to_csv(TABLE_DIR / "learning_curve.csv", index=False, encoding="utf-8-sig")

    eval_pre = selected_pipe.named_steps["preprocessor"]
    eval_names = eval_pre.get_feature_names_out().tolist()
    np.save(ARTIFACT_DIR / "X_train.npy", eval_pre.transform(X_train))
    np.save(ARTIFACT_DIR / "X_test.npy", eval_pre.transform(X_test))
    np.save(ARTIFACT_DIR / "X_full.npy", eval_pre.transform(X))
    (ARTIFACT_DIR / "feature_names_evaluation.txt").write_text("\n".join(eval_names), encoding="utf-8")

    evaluation_bundle = {
        "version": "v3_source_stratified",
        "data_sha256": manifest["primary_sha256"],
        "name": selected_spec["model_name"],
        "candidate": selected_candidate,
        "feature_set": FEATURE_SET_NAME,
        "pipeline": selected_pipe,
        "preprocessor": eval_pre,
        "model": selected_pipe.named_steps["model"],
        "feat_names": eval_names,
        "raw_feature_names": MODEL_FEATURES_SS,
        "numeric_features": NUMERIC_FEATURES_SS,
        "categorical_features": CATEGORICAL_FEATURES_SS,
        "binary_features": BINARY_FEATURES_SS,
        "add_numeric_indicators": False,
        "onehot_min_frequency": ONEHOT_MIN_FREQUENCY,
        "best_params": tuned[selected_candidate]["best_params"],
        "selection_basis": "source_stratified_training_cv_then_cv_gap",
        "split_strategy": "row_random_source_stratified",
        "source_stratify_min_rows": SOURCE_STRATIFY_MIN_ROWS,
        "random_state": SEED,
        "train_indices": train_idx,
        "test_indices": test_idx,
    }
    with (TRAIN_DIR / "evaluation_model.pkl").open("wb") as fh:
        pickle.dump(evaluation_bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)

    final_pipe = clone(selected_pipe).fit(X, y)
    final_pre = final_pipe.named_steps["preprocessor"]
    final_names = final_pre.get_feature_names_out().tolist()
    final_matrix = final_pre.transform(X)
    np.save(ARTIFACT_DIR / "X_full_final.npy", final_matrix)
    (ARTIFACT_DIR / "feature_names.txt").write_text("\n".join(final_names), encoding="utf-8")
    final_bundle = {
        **evaluation_bundle,
        "pipeline": final_pipe,
        "preprocessor": final_pre,
        "model": final_pipe.named_steps["model"],
        "feat_names": final_names,
        "training_rows": len(df),
        "dataset_path": str(PRIMARY_DATA),
    }
    with (TRAIN_DIR / "best_model.pkl").open("wb") as fh:
        pickle.dump(final_bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    dump(final_pre, ARTIFACT_DIR / "preprocessor.joblib")

    model = final_pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(np.asarray(model.coef_)).ravel()
    else:
        importance = np.zeros(len(final_names))
    pd.DataFrame({"Feature": final_names, "Importance": importance}).sort_values(
        "Importance", ascending=False
    ).to_csv(TABLE_DIR / "model_feature_importance.csv", index=False, encoding="utf-8-sig")

    selected_current = comparison.loc[comparison["Selected_Source_Stratified"]].iloc[0]
    seed_test = summary.loc[summary["Metric"].eq("Test_R2")].iloc[0]
    seed_gap = summary.loc[summary["Metric"].eq("Train_Test_R2_Gap")].iloc[0]
    gate = {
        "selected_candidate": str(selected_current["Candidate"]),
        "selected_model": str(selected_current["Model"]),
        "selected_feature_set": str(selected_current["Feature_Set"]),
        "split_strategy": "unweighted_source_stratified_row_split",
        "cv_r2_mean": float(selected_current["CV_R2_Mean"]),
        "cv_r2_sd": float(selected_current["CV_R2_SD"]),
        "cv_gap_mean": float(selected_current["CV_Gap_Mean"]),
        "seed42_train_r2": float(selected_current["Train_R2"]),
        "seed42_test_r2": float(selected_current["Test_R2"]),
        "seed42_train_test_gap": float(selected_current["Train_Test_R2_Gap"]),
        "seed42_test_mae": float(selected_current["Test_MAE"]),
        "seed42_test_rmse": float(selected_current["Test_RMSE"]),
        "multiseed_test_r2_mean": float(seed_test["Mean"]),
        "multiseed_test_r2_sd": float(seed_test["SD"]),
        "multiseed_gap_mean": float(seed_gap["Mean"]),
        "multiseed_gap_sd": float(seed_gap["SD"]),
        "target_test_r2_met": bool(selected_current["Test_R2"] >= 0.70),
        "target_multiseed_r2_met": bool(seed_test["Mean"] >= 0.70),
        "target_gap_012_met": bool(selected_current["Train_Test_R2_Gap"] <= 0.12),
        "target_gap_015_met": bool(selected_current["Train_Test_R2_Gap"] <= 0.15),
        "publication_gate_passed": bool(
            selected_current["Test_R2"] >= 0.70 and seed_test["Mean"] >= 0.70 and seed_gap["Mean"] <= 0.15
        ),
    }
    (REPORT_DIR / "performance_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    methodology = f"""# v3 source-stratified methodology summary

- Dataset: {len(df)} primary experimental rows from {df['Source_File'].nunique()} source files.
- Holdout: row-random 80/20 split stratified by Source_File; sources with fewer than {SOURCE_STRATIFY_MIN_ROWS} rows are grouped as rare_sources.
- Source_File is used only for splitting and is not included as a model descriptor.
- Feature strategy: Catalyst_Label_Ni_Value removed, numeric missing indicators disabled, one-hot min_frequency={ONEHOT_MIN_FREQUENCY}.
- Candidate models: SVR, Random Forest, Gradient Boosting, XGBoost, AdaBoost, Bagging, and Extra Trees.
- Existing tree-ensemble branches use locked regularized-audit hyperparameters; SVR, AdaBoost, and Bagging are tuned only within five-fold CV on the source-stratified training set.
- Model selection uses training-CV evidence only; the holdout metrics are reported after selection and are not used to choose the model.
- Holdout test set is used only once after model selection.
- Stability: fixed selected model/hyperparameters over source-stratified split seeds {STABILITY_SEEDS[0]}–{STABILITY_SEEDS[-1]}; no seed selection.
- Final interpretation model: same locked model refit on all primary data.
- Data SHA-256: {manifest['primary_sha256']}.

## Performance gate

- Selected candidate: {gate['selected_candidate']}
- CV R²: {gate['cv_r2_mean']:.4f} ± {gate['cv_r2_sd']:.4f}
- CV gap: {gate['cv_gap_mean']:.4f}
- Seed 42 Train R²: {gate['seed42_train_r2']:.4f}
- Seed 42 Test R²: {gate['seed42_test_r2']:.4f}
- Seed 42 Train–Test R² gap: {gate['seed42_train_test_gap']:.4f}
- Seed 42 Test MAE/RMSE: {gate['seed42_test_mae']:.3f} / {gate['seed42_test_rmse']:.3f}
- Multi-seed Test R²: {gate['multiseed_test_r2_mean']:.4f} ± {gate['multiseed_test_r2_sd']:.4f}
- Multi-seed Train–Test gap: {gate['multiseed_gap_mean']:.4f} ± {gate['multiseed_gap_sd']:.4f}
- Gate passed: {gate['publication_gate_passed']}
"""
    (REPORT_DIR / "methodology_summary.md").write_text(methodology, encoding="utf-8")
    (TRAIN_DIR / "run_log.txt").write_text(
        f"selected={gate['selected_candidate']}\nrows={len(df)}\nelapsed_seconds={time.time()-started:.1f}\n"
        + json.dumps(gate, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
