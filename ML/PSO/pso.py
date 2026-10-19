# -*- coding: utf-8 -*-
"""Step 5 -- condition-space screening for the deployed blend.

The deployed model is the OOF Weighted Blend held in
``ML/train/output/best_model.pkl``; predictive uncertainty comes from the
conformalized quantile-GB pair (``ML/train/output/uncertainty_model.pkl``
written by model_train.py):  sd = k_br * (q84 - q16) / 2, k_br calibrated by
5-fold OOF within the training split (68% nominal coverage;
``output/reports/uncertainty_calibration.json``).

Task framing: the model is an interpolator over measured literature space, and
extrapolating to unseen compositions would repeat the applicability-domain
failure flagged in review.  This stage sweeps the operating conditions of each
catalyst group of the primary set itself (temperature screened over the
group's own measured range +/-60 C; all other features frozen at the group's
median/mode) and reports the predicted optimum with predictive uncertainty and
a kNN applicability-domain check against the training pool.  No
cross-composition extrapolation is performed.

This file is self-contained (feature contract, deployed-blend class and paths
inlined); the logic is identical to the original modular chain.  Run it after
ML/train/model_train.py.

Usage: python ML/PSO/pso.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

SCRIPT_DIR = Path(__file__).resolve().parent          # ML/PSO
PRIMARY_DATA = SCRIPT_DIR.parent / "data_clean" / "prep_output" / "ni_model_primary.csv"
MODEL_BUNDLE = SCRIPT_DIR.parent / "train" / "output" / "best_model.pkl"
UNCERTAINTY_PKL = SCRIPT_DIR.parent / "train" / "output" / "uncertainty_model.pkl"
UNCERTAINTY_CALIBRATION = SCRIPT_DIR.parent / "train" / "output" / "reports" / "uncertainty_calibration.json"
PSO_DIR = SCRIPT_DIR / "output"
PSO_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "MeOH_Conversion_%"

# ── Feature contract (v6_scientific_features_3, 31 features) ────────────────
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


class WeightedBlendRegressor(BaseEstimator, RegressorMixin):
    """Deployed blend class; defined here so the pickled bundle loads."""

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


AD_COLUMNS = [
    "Reaction_Temp_C", "Ni_Loading_wt%", "Promoter_Loading_wt%",
    "Calcination_Temp_C", "Reduction_Temp_C", "Dry_Temp_C",
    "Calcination_Time_h", "Reduction_Time_h", "Dry_Time_h",
    "S_C_Ratio", "TOS_h", "Pressure_bar", "GHSV_log", "WHSV_log",
]
# Screening bounds are derived from the corpus in main(), not fixed here: a
# hardcoded temperature ceiling silently censored the result two ways -- group
# optima pinned at the boundary, and groups whose own window lay entirely
# above the ceiling dropped without a log line.
TEMP_PAD_C = 60.0    # half-width of each group's own screening window
TEMP_STEP_C = 10.0


def group_base_row(g: pd.DataFrame) -> dict:
    """Frozen per-group feature values: numeric median, categorical mode."""
    base = {}
    for c in MODEL_FEATURES:
        if c == "Reaction_Temp_C":
            continue
        if c in CATEGORICAL_FEATURES:
            vals = g[c].dropna()
            base[c] = vals.mode().iloc[0] if len(vals) else np.nan
        else:
            base[c] = pd.to_numeric(g[c], errors="coerce").median()
    return base


def sweep_frame(base: dict, grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in grid:
        r = dict(base)
        r["Reaction_Temp_C"] = float(t)
        rows.append(r)
    return pd.DataFrame(rows, columns=list(MODEL_FEATURES))


def main() -> None:
    df = pd.read_csv(PRIMARY_DATA, encoding="utf-8-sig", low_memory=False)
    with MODEL_BUNDLE.open("rb") as fh:
        bundle = pickle.load(fh)
    with UNCERTAINTY_PKL.open("rb") as fh:
        unc = pickle.load(fh)
    blend = bundle["pipeline"]
    q16_pipe, q84_pipe = unc["q16"], unc["q84"]
    k_br = float(unc["meta"]["conformal_k_bracketed"])

    # ---- applicability domain over the training pool (kNN, 95th pct) ----
    ad_imp = SimpleImputer(strategy="median")
    ad_scl = RobustScaler()
    ad_X = ad_scl.fit_transform(ad_imp.fit_transform(df[AD_COLUMNS].astype(float)))
    nn = NearestNeighbors(n_neighbors=6).fit(ad_X)
    dist_pool, _ = nn.kneighbors(ad_X)
    ad_threshold = float(np.percentile(dist_pool[:, 1:].mean(axis=1), 95))

    def ad_ok(frame: pd.DataFrame) -> np.ndarray:
        Z = ad_scl.transform(ad_imp.transform(frame[AD_COLUMNS].astype(float)))
        d, _ = nn.kneighbors(Z)
        return d[:, 1:].mean(axis=1) <= ad_threshold

    # ---- per-catalyst condition sweep (groups from the primary set) ----
    # Corpus-wide screening bounds: wide enough that every group's own window
    # fits inside them.
    t_all = pd.to_numeric(df["Reaction_Temp_C"], errors="coerce").dropna()
    t_lo = float(t_all.min()) - TEMP_PAD_C
    t_hi = float(t_all.max()) + TEMP_PAD_C

    sweep_rows, best_rows = [], []
    n_drop_no_temp = n_drop_no_grid = 0
    for (paper, ckey), g in df.groupby(["Paper_ID", "Catalyst_Key"], sort=True):
        base = group_base_row(g)
        own = pd.to_numeric(g["Reaction_Temp_C"], errors="coerce").dropna()
        if own.empty:
            n_drop_no_temp += 1
            continue
        # Each group is screened over its own +/-60 C window, with the grid
        # anchored at that window's lower edge.  Building the grid per group
        # rather than intersecting a shared lattice makes it non-empty whenever
        # lo <= hi: no group can be dropped because the global lattice happened
        # to miss its window (and a single-point group measured at a
        # non-multiple-of-10 temperature would disappear regardless).
        lo = max(t_lo, float(own.min()) - TEMP_PAD_C)
        hi = min(t_hi, float(own.max()) + TEMP_PAD_C)
        grid = np.arange(lo, hi + 1e-9, TEMP_STEP_C)
        if len(grid) == 0:
            n_drop_no_grid += 1
            continue
        sweep = sweep_frame(base, grid)
        mu = np.asarray(blend.predict(sweep), dtype=float)
        sd = k_br * np.maximum(q84_pipe.predict(sweep) - q16_pipe.predict(sweep), 0.0) / 2.0
        ok = ad_ok(sweep)
        ref_row = g.loc[pd.to_numeric(g[TARGET], errors="coerce").idxmax()]
        ref_t = float(ref_row["Reaction_Temp_C"])
        ref_y = float(ref_row[TARGET])
        for t, m, s, a in zip(grid, mu, sd, ok):
            sweep_rows.append({
                "Paper_ID": paper, "Catalyst_Key": ckey,
                "Reaction_Temp_C": float(t), "Predicted_Conversion_%": float(np.clip(m, 0, 100)),
                "Pred_Std": float(s), "Within_AD": bool(a),
                "Reference_T": ref_t, "Reference_Conversion": ref_y,
            })
        cand = [w for w in zip(grid, mu, sd, ok) if w[3]]
        if not cand:
            cand = list(zip(grid, mu, sd, ok))
        t_b, m_b, s_b, a_b = max(cand, key=lambda w: w[1] - 0.5 * w[2])
        best_rows.append({
            "Paper_ID": paper, "Catalyst_Key": ckey, "Catalyst": str(ref_row.get("Catalyst", "")),
            "Optimal_T": float(t_b), "Predicted_Conversion_%": float(np.clip(m_b, 0, 100)),
            "Pred_Std": float(s_b), "Within_AD": bool(a_b),
            "Reference_T": ref_t, "Reference_Conversion": ref_y,
        })

    n_groups = int(df.groupby(["Paper_ID", "Catalyst_Key"]).ngroups)
    print(f"[pso] groups {n_groups}: screened {len(best_rows)}, "
          f"dropped {n_drop_no_temp} (no measured temperature) + "
          f"{n_drop_no_grid} (empty grid)", flush=True)

    sweep_df = pd.DataFrame(sweep_rows)
    best_df = pd.DataFrame(best_rows)
    sweep_df.to_csv(PSO_DIR / "temp_sweep_all.csv", index=False, encoding="utf-8-sig")
    best_df.to_csv(PSO_DIR / "per_catalyst_optimal_conditions.csv", index=False, encoding="utf-8-sig")
    ad_summary = pd.DataFrame([{
        "n_catalysts": len(best_df),
        "sweep_points": len(sweep_df),
        "sweep_within_AD_rate": float(sweep_df["Within_AD"].mean()),
        "best_point_within_AD_rate": float(best_df["Within_AD"].mean()),
        "mean_Pred_Std_at_best": float(best_df["Pred_Std"].mean()),
        "ad_threshold_95pct": ad_threshold,
    }])
    ad_summary.to_csv(PSO_DIR / "ad_summary.csv", index=False, encoding="utf-8-sig")

    (PSO_DIR / "pso_metadata.json").write_text(json.dumps({
        "model": str(bundle.get("candidate")), "model_file": str(MODEL_BUNDLE),
        "protocol": "verified-v6 pure row-random (OOF Weighted Blend)",
        "uncertainty": {
            "method": "conformalized quantile-GB pair (q16/q84)",
            "sd": f"(q84 - q16) / 2 * k_br, k_br={k_br:.4f} (bracketed-conformal, 68% nominal)",
            "calibration": str(UNCERTAINTY_CALIBRATION),
            "model_file": str(UNCERTAINTY_PKL),
        },
        "task": "per-catalyst condition screening (groups from the primary set itself)",
        "reference_note": "Group key = Paper_ID + Catalyst_Key; frozen features = group median/mode; "
                          "each group's grid spans its own measured temperature range +/-60 C, "
                          "anchored at that window's lower edge.",
        "note": "No cross-composition extrapolation: the model is only evaluated on existing "
                "catalysts' condition grids.",
        "groups_total": n_groups,
        "groups_screened": int(len(best_df)),
        "groups_dropped_no_temperature": n_drop_no_temp,
        "groups_dropped_empty_grid": n_drop_no_grid,
        "temp_grid": {
            "corpus_bounds_C": [t_lo, t_hi],
            "group_window_pad_C": TEMP_PAD_C,
            "step_C": TEMP_STEP_C,
            "construction": "per group: arange(own_min - pad, own_max + pad, step), clipped to corpus bounds",
        },
        "ad": "kNN(6) mean-distance, RobustScaler, 95th-pct threshold on training pool",
        "score": "argmax predicted conversion - 0.5 * Pred_Std (AD-restricted)",
    }, indent=2, ensure_ascii=False))
    print(ad_summary.to_string(index=False))
    print(f"[pso] {len(best_df)} catalysts x {len(sweep_df)} sweep points -> {PSO_DIR}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
