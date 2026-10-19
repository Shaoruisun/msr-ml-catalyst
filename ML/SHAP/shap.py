# -*- coding: utf-8 -*-
"""Step 4 -- Permutation-Shapley interpretation of the deployed model.

The shap package is unusable in this environment (numba/numpy ABI conflict),
so exact single-reference permutation Shapley values (Strumbelj-Kononenko) are
computed directly against the model's batch predict -- mathematically the same
estimator as shap's permutation mode.  Attribution runs on the 31 RAW
scientific features: the blend predicts on raw-column DataFrames, so no
transformed matrix or one-hot folding is needed and every Shapley parent is a
directly interpretable descriptor.  Every model feature is additionally
classified as an operating condition or a catalyst-chemistry descriptor, and
the two groups must partition the feature contract exactly (asserted below).

This file is self-contained (feature contract, deployed-blend class and paths
inlined); the logic is identical to the original modular chain.  Run it after
ML/train/model_train.py -- it loads ``ML/train/output/best_model.pkl``.

Usage: python ML/SHAP/shap.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone

SCRIPT_DIR = Path(__file__).resolve().parent          # ML/SHAP
PRIMARY_DATA = SCRIPT_DIR.parent / "data_clean" / "prep_output" / "ni_model_primary.csv"
MODEL_BUNDLE = SCRIPT_DIR.parent / "train" / "output" / "best_model.pkl"
SHAP_DIR = SCRIPT_DIR / "output"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

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


# ── Condition vs catalyst-chemistry attribution ────────────────────────────
# Every model feature is assigned to exactly one side, and the two sides must
# partition MODEL_FEATURES exactly (asserted below).  The share is then a plain
# mass ratio over the two groups.
#
# Classification rule, applied to the descriptor's physical role:
#   condition  = how the measurement was run (temperature, feed, space
#                velocity, pressure, time-on-stream, reactor configuration)
#   chemistry  = what the catalyst is and how it was prepared (loadings,
#                promoter identity and properties, support, preparation
#                treatments such as calcination/reduction/drying)
#
# Temperature is represented once, in degrees Celsius (the contract's sole
# temperature form), so the share split is not diluted by equivalent encodings.
CONDITION_FEATURES = {
    # thermal (degrees Celsius is the sole temperature representation)
    "Reaction_Temp_C",
    # feed and reactor charge
    "S_C_Ratio",
    # space velocity (raw and log forms of the same quantities)
    "GHSV_log", "GHSV_h_inv_log",
    "WHSV_log", "LHSV_h_inv_log",
    # pressure
    "Pressure_bar",
    # time on stream / duration
    "TOS_h", "TOS_log", "Reaction_Time_h",
}
CHEMISTRY_FEATURES = {
    # metal loadings
    "Ni_Loading_wt%", "Promoter_Loading_wt%",
    # promoter identity and properties
    "Promoter_Metal", "has_promoter",
    "prom_electronegativity", "prom_ionic_radius", "prom_atomic_mass",
    "prom_first_ionization",
    # support identity and properties
    "Support", "support_type", "support_acid_base", "support_reducible",
    # preparation route and treatments
    "Metal_Loading_Method", "Precursor_Family", "Support_Prep_Method",
    "Calcination_Temp_C", "Calcination_Time_h",
    "Reduction_Temp_C", "Reduction_Time_h",
    "Dry_Temp_C", "Dry_Time_h",
}
_unclassified = set(MODEL_FEATURES) - CONDITION_FEATURES - CHEMISTRY_FEATURES
_overlap = CONDITION_FEATURES & CHEMISTRY_FEATURES
if _unclassified:
    raise ValueError(
        f"Model features not classified as condition or chemistry: {sorted(_unclassified)}. "
        "Add each to CONDITION_FEATURES or CHEMISTRY_FEATURES in shap.py.")
if _overlap:
    raise ValueError(f"Model features classified on both sides: {sorted(_overlap)}")

N_PERM = 20
ROBUST_N = 200
TOP_K = 15


def permutation_shapley(predict_fn, X: np.ndarray, cols: list[str],
                        row_pos: list[int], n_perm: int, seed: int) -> np.ndarray:
    """Exact single-reference permutation Shapley on the raw feature matrix.

    X is an object array (rows of the 31 raw columns); predict_fn receives a
    DataFrame.  All n_perm walks of one record are stacked into a single batch
    predict call (call overhead dominates an individual 31-row walk).
    Returns (len(row_pos), d) attribution matrix.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    phi = np.zeros((len(row_pos), d))
    for r, i in enumerate(row_pos):
        x = X[i]
        mats = []
        widths = []
        for _ in range(n_perm):
            b = X[rng.integers(n)]
            order = rng.permutation(d)
            base = len(mats)
            mats.append(b.copy())
            cur = b.copy()
            for j in order:
                cur = cur.copy()
                cur[j] = x[j]
                mats.append(cur.copy())
            widths.append((base, order))
        preds = np.asarray(predict_fn(pd.DataFrame(np.asarray(mats), columns=cols)), dtype=float)
        for base, order in widths:
            phi[r, order] += np.diff(preds[base:base + d + 1])
        phi[r] /= n_perm
        if (r + 1) % 100 == 0:
            print(f"  [shap] rows {r + 1}/{len(row_pos)}", flush=True)
    return phi


def two_segment(x: np.ndarray, y: np.ndarray):
    """Best two-piece linear split at deciles (segments >=30 pts)."""
    best = None
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n = len(xs)
    for q in np.arange(0.15, 0.86, 0.05):
        k = int(n * q)
        if k < 30 or n - k < 30:
            continue
        sse = 0.0
        slopes = []
        for seg in (slice(0, k), slice(k, n)):
            xa, ya = xs[seg], ys[seg]
            if len(np.unique(xa)) < 2:
                slopes.append(np.nan)
                continue
            s = np.polyfit(xa, ya, 1)
            slopes.append(s[0])
            sse += float(np.sum((np.polyval(s, xa) - ya) ** 2))
        if any(np.isnan(slopes)):
            continue
        r2 = 1.0 - sse / float(np.sum((ys - ys.mean()) ** 2))
        if best is None or r2 > best["r2"]:
            best = {"threshold": float(xs[k]), "slope_lo": slopes[0], "slope_hi": slopes[1], "r2": r2}
    return best


def main() -> None:
    df = pd.read_csv(PRIMARY_DATA, encoding="utf-8-sig", low_memory=False)
    with MODEL_BUNDLE.open("rb") as fh:
        bundle = pickle.load(fh)
    blend = bundle["pipeline"]
    cols = list(MODEL_FEATURES)
    X = df[cols].to_numpy(dtype=object)
    assert X.shape == (len(df), len(cols)), "raw feature matrix shape mismatch"
    print(f"[shap] matrix {X.shape}, model={bundle.get('candidate')}", flush=True)

    def predict_fn(frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(blend.predict(frame), dtype=float)

    base = predict_fn(pd.DataFrame(X, columns=cols))

    def shapley_rows(row_pos, seed):
        return permutation_shapley(predict_fn, X, cols, row_pos, N_PERM, seed)

    phi = shapley_rows(list(range(len(df))), seed=42)
    additivity = float(np.mean(np.abs(phi.sum(axis=1) - (base - base.mean()))))
    print(f"[shap] additivity residual check: {additivity:.4f}", flush=True)

    parents = cols  # raw features are the parents by construction
    mean_abs = np.abs(phi).mean(axis=0)
    imp = (pd.DataFrame({"Feature": parents, "MeanAbsSHAP": mean_abs})
           .sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True))
    imp.to_csv(SHAP_DIR / "shap_mean_abs.csv", index=False, encoding="utf-8-sig")
    col_imp = (pd.DataFrame({"Column": parents, "Parent": parents,
                             "MeanAbsSHAP": mean_abs})
               .sort_values("MeanAbsSHAP", ascending=False))
    col_imp.to_csv(SHAP_DIR / "shap_mean_abs_columns.csv", index=False, encoding="utf-8-sig")

    vals = pd.DataFrame(phi, columns=parents)
    vals.insert(0, "row_id", df["row_id"].to_numpy())
    vals.insert(1, "Paper_ID", df["Paper_ID"].to_numpy())
    vals.insert(2, "Actual", df[TARGET].astype(float).to_numpy())
    vals.to_csv(SHAP_DIR / "shap_values_all.csv", index=False, encoding="utf-8-sig")

    # Condition vs chemistry share.  Both sides are summed explicitly over the
    # asserted partition so that a mis-bucketed descriptor changes the reported
    # number instead of hiding in the 1 - cond_share complement.
    total = float(imp["MeanAbsSHAP"].sum())
    cond_sum = float(imp.loc[imp["Feature"].isin(CONDITION_FEATURES), "MeanAbsSHAP"].sum())
    chem_sum = float(imp.loc[imp["Feature"].isin(CHEMISTRY_FEATURES), "MeanAbsSHAP"].sum())
    if abs(cond_sum + chem_sum - total) > 1e-9 * max(total, 1.0):
        raise ValueError(
            f"condition + chemistry attribution ({cond_sum + chem_sum:.6f}) does not account "
            f"for the full total ({total:.6f}); the partition is incomplete")
    cond_share = cond_sum / total
    chem_share = chem_sum / total
    pd.DataFrame([{"condition_share": cond_share, "chemistry_share": chem_share,
                   "total_mean_abs_shap": total}]).to_csv(
        SHAP_DIR / "condition_vs_chemistry_share.csv", index=False, encoding="utf-8-sig")

    # beeswarm (top TOP_K raw features).  Numeric features colored by their
    # 5-95 percentile-normalized value; categoricals by category code.
    top = imp.head(TOP_K)["Feature"].tolist()
    pos = {c: j for j, c in enumerate(cols)}
    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    rng = np.random.default_rng(0)
    for k, f in enumerate(top):
        v = phi[:, pos[f]]
        if f in CATEGORICAL_FEATURES:
            codes = pd.factorize(df[f].fillna("Unknown").astype(str))[0].astype(float)
            c = codes / max(codes.max(), 1.0)
            ok = np.ones(len(v), dtype=bool)
        else:
            c = pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=float)
            lo, hi = np.nanpercentile(c, 5), np.nanpercentile(c, 95)
            c = np.clip((c - lo) / (hi - lo + 1e-9), 0, 1)
            ok = np.isfinite(c)
        sc = ax.scatter(v[ok], k + rng.uniform(-0.28, 0.28, ok.sum()), s=9,
                        c=c[ok], cmap="coolwarm", alpha=0.8, vmin=0, vmax=1)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (pp conversion)")
    plt.colorbar(sc, ax=ax, pad=0.02,
                 label="feature value (5-95 pct normalized; categorical = code)")
    fig.savefig(SHAP_DIR / "fig_shap_beeswarm.svg", bbox_inches="tight")
    fig.savefig(SHAP_DIR / "fig_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.barh(imp.head(TOP_K)["Feature"][::-1], imp.head(TOP_K)["MeanAbsSHAP"][::-1], color="#0072B2")
    ax.set_xlabel("mean |SHAP| (pp conversion)")
    fig.savefig(SHAP_DIR / "fig_shap_bar.svg", bbox_inches="tight")
    fig.savefig(SHAP_DIR / "fig_shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Dependence + quantitative-threshold audit.  The panel set spans both sides
    # of the split: the most influential numerical chemistry descriptors plus the
    # most influential operating condition.
    numeric_chem = [f for f in imp.head(14)["Feature"]
                    if f in NUMERIC_FEATURES and f in CHEMISTRY_FEATURES][:5]
    cond_ranked = [f for f in imp.head(14)["Feature"]
                   if f in NUMERIC_FEATURES and f in CONDITION_FEATURES]
    numeric_cond = cond_ranked[:1]
    numeric_top = numeric_chem + numeric_cond
    dep_rows, insight_rows = [], []
    # Drop the previous run's per-feature dependence panels.  The selected set
    # changes whenever the ranking does, and a leftover panel of a feature that
    # is no longer selected is indistinguishable from a current one.
    for stale in SHAP_DIR.glob("fig_shap_dep_*"):
        stale.unlink()
    for f in numeric_top:
        xv = pd.to_numeric(df[f], errors="coerce").to_numpy()
        yv = phi[:, pos[f]]
        ok = np.isfinite(xv) & np.isfinite(yv)
        if ok.sum() < 40:
            continue
        dep_rows.append(pd.DataFrame({"Feature": f, "Value": xv[ok], "SHAP": yv[ok]}))
        srt = pd.DataFrame({"v": xv[ok], "s": yv[ok]}).sort_values("v")
        trend = srt["s"].rolling(60, min_periods=20, center=True).median()
        fig, ax = plt.subplots(figsize=(5.6, 4.2))
        ax.scatter(srt["v"], srt["s"], s=9, alpha=0.45, color="#555555")
        ax.plot(srt["v"], trend, color="#D55E00", lw=2)
        ax.set_xlabel(f); ax.set_ylabel("SHAP (pp)")
        fig.savefig(SHAP_DIR / f"fig_shap_dep_{f.replace('%','pct').replace('/','_')}.svg", bbox_inches="tight")
        fig.savefig(SHAP_DIR / f"fig_shap_dep_{f.replace('%','pct').replace('/','_')}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        seg = two_segment(srt["v"].to_numpy(), srt["s"].to_numpy())
        overall = float(np.polyfit(srt["v"], srt["s"], 1)[0]) if srt["v"].nunique() > 1 else np.nan
        insight_rows.append({
            "Feature": f, "N": int(ok.sum()),
            "Overall_Slope_pp_per_unit": overall,
            "Segment_Threshold": seg["threshold"] if seg else np.nan,
            "Slope_Below": seg["slope_lo"] if seg else np.nan,
            "Slope_Above": seg["slope_hi"] if seg else np.nan,
            "TwoSegment_R2": seg["r2"] if seg else np.nan,
        })
    if dep_rows:
        pd.concat(dep_rows, ignore_index=True).to_csv(SHAP_DIR / "shap_dependence.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(insight_rows).to_csv(SHAP_DIR / "shap_quantitative_insights.csv", index=False, encoding="utf-8-sig")

    # robustness: second background seed on subsample, rank agreement
    sub = list(np.random.default_rng(7).choice(len(df), size=ROBUST_N, replace=False))
    phi2 = shapley_rows(sub, seed=99)
    ma1 = mean_abs
    ma2 = np.abs(phi2).mean(axis=0)
    rho = float(spearmanr(ma1, ma2).statistic)
    pd.DataFrame([{"spearman_parent_ranking": rho, "subsample_n": ROBUST_N,
                   "n_perm": N_PERM}]).to_csv(SHAP_DIR / "shap_robustness.csv", index=False, encoding="utf-8-sig")

    meta = {
        "model": str(bundle.get("candidate")), "model_file": str(MODEL_BUNDLE),
        "blend_components": list(getattr(blend, "names", [])),
        "blend_weights": [float(w) for w in getattr(blend, "weights", [])],
        "method": "exact single-reference permutation Shapley (Strumbelj-Kononenko) on raw features",
        "n_rows": int(len(df)), "n_raw_features": int(len(cols)), "n_permutations": N_PERM,
        "additivity_residual": additivity, "condition_share": cond_share,
        "chemistry_share": chem_share,
        "robustness_spearman": rho, "top5": imp.head(5).to_dict(orient="records"),
    }
    (SHAP_DIR / "shap_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
