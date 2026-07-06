"""SHAP interpretation and training-CV ablation for the source-stratified model."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
_saved_path = list(sys.path)
sys.path = [p for p in sys.path if Path(p or ".").resolve() != SCRIPT_DIR]
import shap  # noqa: E402
sys.path = _saved_path


PRIMARY_DATA = ML_ROOT / "data_clean" / "prep_output" / "ni_model_primary.csv"
TRAIN_DIR = ML_ROOT / "train"
ARTIFACT_DIR = TRAIN_DIR / "artifacts"
SHAP_OUT = SCRIPT_DIR / "output"

SEED = 42
CV_FOLDS = 5
TARGET = "MeOH_Conversion_%"
OKABE_ITO = {
    "train": "#0072B2",
    "test": "#E69F00",
    "promoted": "#009E73",
    "unpromoted": "#CC79A7",
    "reference": "#222222",
}


def make_preprocessor(nums: list[str], cats: list[str], bins: list[str], add_indicators: bool) -> ColumnTransformer:
    transformers = []
    if nums:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=add_indicators)),
            ("scaler", StandardScaler()),
        ]), nums))
    if cats:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)),
        ]), cats))
    if bins:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), bins))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def pretty(name: str) -> str:
    name = name.replace("num__", "").replace("cat__", "").replace("bin__", "")
    name = name.replace("missingindicator_", "Missing: ")
    prefix_replacements = [
        ("Support_Prep_Method_", "Support prep: "),
        ("Metal_Loading_Method_", "Loading method: "),
        ("Precursor_Family_", "Precursor: "),
        ("Promoter_Metal_", "Promoter: "),
        ("support_type_", "Support type: "),
        ("Support_", "Support: "),
    ]
    for old, new in prefix_replacements:
        name = name.replace(old, new)
    replacements = {
        "MeOH_Conversion_%": "MeOH conversion (%)",
        "Reaction_Temp_C": "Reaction temperature (°C)",
        "Ni_Loading_wt%": "Ni loading (wt%)",
        "Promoter_Loading_wt%": "Promoter loading (wt%)",
        "Calcination_Temp_C": "Calcination temperature (°C)",
        "Reduction_Temp_C": "Reduction temperature (°C)",
        "Dry_Temp_C": "Drying temperature (°C)",
        "Calcination_Time_h": "Calcination time (h)",
        "Reduction_Time_h": "Reduction time (h)",
        "Dry_Time_h": "Drying time (h)",
        "S_C_Ratio": "S/C ratio",
        "TOS_h": "TOS (h)",
        "Pressure_bar": "Pressure (bar)",
        "GHSV_log": "GHSV (log)",
        "WHSV_log": "WHSV (log)",
        "calc_reduc_temp_diff": "Calcination–reduction ΔT (°C)",
        "total_treatment_time_h": "Total treatment time (h)",
        "has_promoter": "Has promoter",
        "support_reducible": "Reducible support",
        "support_acid_base": "Support acid–base class",
        "Metal_Loading_Method": "Loading method",
        "Precursor_Family": "Precursor family",
        "Support_Prep_Method": "Support preparation method",
        "support_type": "Support type",
        "prom_atomic_mass": "Promoter atomic mass",
        "prom_first_ionization": "Promoter first ionization",
        "prom_electronegativity": "Promoter electronegativity",
        "prom_ionic_radius": "Promoter ionic radius",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name


def style_axis(ax: plt.Axes, guides: bool = False) -> None:
    ax.grid(False)
    if guides:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, linestyle=":", zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3)


def save(fig: plt.Figure, folder: Path, stem: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(folder / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(folder / f"{stem}.tif", dpi=600, bbox_inches="tight", pad_inches=0.02, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main() -> None:
    SHAP_OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none", "axes.unicode_minus": False})
    with (TRAIN_DIR / "best_model.pkl").open("rb") as fh:
        bundle = pickle.load(fh)
    with (TRAIN_DIR / "evaluation_model.pkl").open("rb") as fh:
        eval_bundle = pickle.load(fh)
    df = pd.read_csv(PRIMARY_DATA, encoding="utf-8-sig")
    matrix = np.load(ARTIFACT_DIR / "X_full_final.npy")
    names = list(bundle["feat_names"])
    if matrix.shape[1] != len(names):
        raise ValueError("Final feature matrix and feature-name count differ")

    explainer = shap.TreeExplainer(bundle["model"])
    values = explainer.shap_values(matrix)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    mean_abs = np.abs(values).mean(axis=0)

    importance = getattr(bundle["model"], "feature_importances_", np.zeros(len(names)))
    imp = pd.DataFrame({"Feature": names, "Feature_Label": [pretty(x) for x in names], "Importance": importance})
    imp = imp.sort_values("Importance", ascending=False).reset_index(drop=True)
    shap_imp = pd.DataFrame({"Feature": names, "Feature_Label": [pretty(x) for x in names], "Mean_Abs_SHAP": mean_abs})
    shap_imp = shap_imp.sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    imp.to_csv(SHAP_OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")
    shap_imp.to_csv(SHAP_OUT / "shap_mean_abs.csv", index=False, encoding="utf-8-sig")

    audit_columns: dict[str, np.ndarray] = {"row_id": df["row_id"].to_numpy()}
    for i, name in enumerate(names):
        audit_columns[f"Value::{name}"] = matrix[:, i]
        audit_columns[f"SHAP::{name}"] = values[:, i]
    pd.DataFrame(audit_columns).to_csv(SHAP_OUT / "shap_values_all.csv", index=False, encoding="utf-8-sig")

    top10 = shap_imp.head(10)
    beeswarm_parts = []
    for rank, row in top10.iterrows():
        idx = names.index(row["Feature"])
        beeswarm_parts.append(pd.DataFrame({
            "row_id": df["row_id"],
            "Feature": row["Feature"],
            "Feature_Label": row["Feature_Label"],
            "Rank": rank + 1,
            "Feature_Value_Transformed": matrix[:, idx],
            "SHAP_Value": values[:, idx],
        }))
    pd.concat(beeswarm_parts, ignore_index=True).to_csv(SHAP_OUT / "shap_beeswarm_source.csv", index=False, encoding="utf-8-sig")

    numerical_candidates = shap_imp[
        shap_imp["Feature"].str.startswith("num__") & ~shap_imp["Feature"].str.contains("missingindicator")
    ].copy()
    dependence_parts = []
    for _, row in numerical_candidates.head(6).iterrows():
        raw = row["Feature"].removeprefix("num__")
        idx = names.index(row["Feature"])
        dependence_parts.append(pd.DataFrame({
            "row_id": df["row_id"],
            "Feature": raw,
            "Feature_Label": pretty(row["Feature"]),
            "Feature_Value": pd.to_numeric(df[raw], errors="coerce"),
            "SHAP_Value": values[:, idx],
        }))
    pd.concat(dependence_parts, ignore_index=True).to_csv(SHAP_OUT / "shap_dependence_source.csv", index=False, encoding="utf-8-sig")

    train_df = df.iloc[np.asarray(eval_bundle["train_indices"], dtype=int)].copy()
    y_train = train_df[TARGET].to_numpy(float)
    base_model = clone(eval_bundle["model"])
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    raw_features = list(eval_bundle["raw_feature_names"])
    nums_all = list(eval_bundle["numeric_features"])
    cats_all = list(eval_bundle["categorical_features"])
    bins_all = list(eval_bundle["binary_features"])
    add_indicators = bool(eval_bundle["add_numeric_indicators"])

    def cv_score(drop_feature: str | None) -> tuple[float, float]:
        nums = [x for x in nums_all if x != drop_feature]
        cats = [x for x in cats_all if x != drop_feature]
        bins = [x for x in bins_all if x != drop_feature]
        cols = nums + cats + bins
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor(nums, cats, bins, add_indicators)),
            ("model", clone(base_model)),
        ])
        scores = cross_val_score(pipeline, train_df[cols], y_train, cv=cv, scoring="r2", n_jobs=-1, error_score="raise")
        return float(scores.mean()), float(scores.std(ddof=1))

    baseline_mean, baseline_sd = cv_score(None)
    ablation_rows = []
    for feature in raw_features:
        reduced_mean, reduced_sd = cv_score(feature)
        ablation_rows.append({
            "Feature": feature,
            "Feature_Label": pretty(feature),
            "Baseline_CV_R2": baseline_mean,
            "Baseline_CV_R2_SD": baseline_sd,
            "Reduced_CV_R2": reduced_mean,
            "Reduced_CV_R2_SD": reduced_sd,
            "Delta_CV_R2_After_Removal": reduced_mean - baseline_mean,
        })
    pd.DataFrame(ablation_rows).sort_values("Delta_CV_R2_After_Removal").to_csv(SHAP_OUT / "ablation_cv.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(4.8, 3.8), constrained_layout=True)
    show = shap_imp.head(15).iloc[::-1]
    ax.barh(show["Feature_Label"], show["Mean_Abs_SHAP"], color=OKABE_ITO["test"], alpha=0.86)
    ax.set_xlabel("Mean |SHAP value|")
    style_axis(ax)
    save(fig, SHAP_OUT / "fig_shap_bar", "shap_bar")

    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    for row_pos, (_, row) in enumerate(top10.iloc[::-1].iterrows()):
        idx = names.index(row["Feature"])
        val = matrix[:, idx]
        denom = np.nanmax(val) - np.nanmin(val)
        color = (val - np.nanmin(val)) / denom if denom > 0 else np.zeros_like(val)
        ax.scatter(values[:, idx], row_pos + rng.normal(0, 0.09, len(df)), c=color, cmap="coolwarm", vmin=0, vmax=1,
                   s=9, alpha=0.62, edgecolors="none", zorder=2)
    ax.axvline(0, color="#222222", linewidth=0.7)
    ax.set_yticks(range(len(top10)), top10.iloc[::-1]["Feature_Label"])
    ax.set_xlabel("SHAP value")
    style_axis(ax, guides=True)
    save(fig, SHAP_OUT / "fig_shap_beeswarm", "shap_beeswarm")

    metadata = {
        "model": bundle["name"],
        "candidate": bundle.get("candidate", ""),
        "feature_set": bundle.get("feature_set", ""),
        "split_strategy": bundle.get("split_strategy", ""),
        "rows": len(df),
        "transformed_features": len(names),
        "ablation_baseline_cv_r2": baseline_mean,
        "data_sha256": bundle["data_sha256"],
    }
    (SHAP_OUT / "shap_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
