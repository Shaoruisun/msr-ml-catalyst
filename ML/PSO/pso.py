"""Applicability-domain constrained PSO using the frozen model Pipeline."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
PRIMARY_DATA = ML_ROOT / "data_clean" / "prep_output" / "ni_model_primary.csv"
TRAIN_DIR = ML_ROOT / "train"
PSO_DIR = SCRIPT_DIR / "output"

SEED = 42


TEMP_FIXED_LIST = [300, 350, 400, 450, 500]
N_PARTICLES = 35
N_ITER = 110
RANDOM_SAMPLES_PER_TRACK_TEMP = 2500
MIN_FAMILY_COUNT = 10

OPT_COLUMNS = [
    "Ni_Loading_wt%", "Promoter_Loading_wt%", "Calcination_Temp_C",
    "Reduction_Temp_C", "Dry_Temp_C", "Calcination_Time_h",
    "Reduction_Time_h", "Dry_Time_h",
]

# The applicability domain is evaluated in the continuous experimental space.
# Observed categorical families are enforced separately by the eligible-family table;
# including sparse one-hot combinations in kNN distance would reject every novel
# continuous candidate even when it lies inside the measured operating domain.
AD_COLUMNS = [
    "Reaction_Temp_C", "Ni_Loading_wt%", "Promoter_Loading_wt%",
    "Calcination_Temp_C", "Reduction_Temp_C", "Dry_Temp_C",
    "Calcination_Time_h", "Reduction_Time_h", "Dry_Time_h",
    "S_C_Ratio", "TOS_h", "Pressure_bar", "GHSV_log", "WHSV_log",
]

PHYSICAL_BOUNDS = {
    "Ni_Loading_wt%": (0.01, 80.0), "Promoter_Loading_wt%": (0.0, 80.0),
    "Calcination_Temp_C": (100.0, 900.0), "Reduction_Temp_C": (100.0, 900.0),
    "Dry_Temp_C": (20.0, 200.0), "Calcination_Time_h": (0.1, 48.0),
    "Reduction_Time_h": (0.1, 48.0), "Dry_Time_h": (0.1, 48.0),
}

PROMOTER_PROPS = {
    "Ce": (1.12, 1.01, 140.12, 5.539), "Co": (1.88, 0.745, 58.93, 7.881),
    "Cu": (1.90, 0.73, 63.55, 7.726), "In": (1.78, 0.80, 114.82, 5.786),
    "La": (1.10, 1.032, 138.91, 5.577), "Mg": (1.31, 0.72, 24.31, 7.646),
    "Mo": (2.16, 0.59, 95.95, 7.092), "Pd": (2.20, 0.86, 106.42, 8.337),
    "Pt": (2.28, 0.80, 195.08, 9.000), "Sn": (1.96, 0.69, 118.71, 7.344),
    "Zn": (1.65, 0.74, 65.38, 9.394),
}

SUPPORT_PROPS = {
    "Al2O3": ("oxide", 2, 0), "Al2O4": ("oxide", 2, 0), "CN": ("nitride", 0, 0),
    "h-BN": ("nitride", 0, 0), "CNTs": ("carbon", 0, 0), "Carbon": ("carbon", 0, 0),
    "CeO2": ("oxide", 3, 1), "Cement-Clay": ("composite", 0, 0), "LDH": ("ldh", 3, 0),
    "MgAl": ("ldh", 3, 0), "MgO": ("oxide", 3, 0), "Mo2C": ("carbide", 0, 1),
    "NaF": ("salt", 0, 0), "SnO2": ("oxide", 2, 1), "TiO2": ("oxide", 2, 1),
    "TiO2-CeO2": ("composite", 2, 1), "ZnO": ("oxide", 2, 1), "ZrO2": ("oxide", 2, 1),
    "Perovskite": ("oxide", 2, 1), "Spinel": ("oxide", 2, 1),
    "Other": ("unknown", 0, 0), "Unknown": ("unknown", 0, 0),
}


def observed_bounds(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    low, high, rows = [], [], []
    for col in OPT_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        plow, phigh = PHYSICAL_BOUNDS[col]
        if values.nunique() >= 2:
            lo = max(plow, float(values.quantile(0.01)))
            hi = min(phigh, float(values.quantile(0.99)))
        elif len(values):
            lo = max(plow, float(values.iloc[0]) * 0.9)
            hi = min(phigh, float(values.iloc[0]) * 1.1 + 1e-6)
        else:
            lo, hi = plow, phigh
        if hi <= lo:
            hi = min(phigh, lo + max(1.0, abs(lo) * 0.1))
        low.append(lo); high.append(hi)
        rows.append({"Parameter": col, "Lower": lo, "Upper": hi, "Physical_Lower": plow, "Physical_Upper": phigh})
    return np.array(low), np.array(high), pd.DataFrame(rows)


def family_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = df.copy()
    work["Promoter_Key"] = work["Promoter_Metal"].fillna("Unknown").astype(str)
    work["Support"] = work["Support"].fillna("Unknown").astype(str)
    work["Metal_Loading_Method"] = work["Metal_Loading_Method"].fillna("Unknown").astype(str)
    counts = work.groupby(["Support", "Promoter_Key", "Metal_Loading_Method"], dropna=False).size().reset_index(name="Dataset_Count")
    counts["Track"] = np.where(counts["Promoter_Key"].str.lower().eq("unknown"), "unpromoted", "promoted")
    counts["Family"] = counts["Support"] + " | " + counts["Promoter_Key"] + " | " + counts["Metal_Loading_Method"]
    result = {}
    for track in ("promoted", "unpromoted"):
        sub = counts[(counts["Track"].eq(track)) & (counts["Dataset_Count"] >= MIN_FAMILY_COUNT)].copy()
        if sub.empty:
            sub = counts[counts["Track"].eq(track)].nlargest(5, "Dataset_Count").copy()
        result[track] = sub.reset_index(drop=True)
    return result


def assumptions(df: pd.DataFrame) -> dict[str, object]:
    def median(col: str, fallback: float) -> float:
        value = pd.to_numeric(df[col], errors="coerce").median()
        return float(value) if pd.notna(value) else fallback
    def mode(col: str, fallback: str) -> str:
        values = df[col].dropna().astype(str)
        return values.mode().iloc[0] if not values.empty else fallback
    return {
        "S_C_Ratio": median("S_C_Ratio", 2.0), "TOS_h": median("TOS_h", 10.0),
        "Pressure_bar": median("Pressure_bar", 1.0), "GHSV_log": median("GHSV_log", 0.0),
        "WHSV_log": median("WHSV_log", 0.0), "Catalyst_Label_Ni_Value": median("Catalyst_Label_Ni_Value", 5.0),
        "Precursor_Family": mode("Precursor_Family", "Unknown"),
        "Support_Prep_Method": mode("Support_Prep_Method", "Unknown"),
    }


def build_frame(positions: np.ndarray, families: pd.DataFrame, temp: float, fixed: dict[str, object], track: str) -> pd.DataFrame:
    n = len(positions)
    family_idx = np.clip(np.rint(positions[:, -1]).astype(int), 0, len(families) - 1)
    fam = families.iloc[family_idx].reset_index(drop=True)
    out = pd.DataFrame(positions[:, :len(OPT_COLUMNS)], columns=OPT_COLUMNS)
    if track == "unpromoted":
        out["Promoter_Loading_wt%"] = 0.0
    out["Reaction_Temp_C"] = temp
    out["S_C_Ratio"] = fixed["S_C_Ratio"]
    out["TOS_h"] = fixed["TOS_h"]
    out["Pressure_bar"] = fixed["Pressure_bar"]
    out["GHSV_log"] = fixed["GHSV_log"]
    out["WHSV_log"] = fixed["WHSV_log"]
    out["Catalyst_Label_Ni_Value"] = fixed["Catalyst_Label_Ni_Value"]
    out["calc_reduc_temp_diff"] = out["Calcination_Temp_C"] - out["Reduction_Temp_C"]
    out["total_treatment_time_h"] = out[["Calcination_Time_h", "Reduction_Time_h", "Dry_Time_h"]].sum(axis=1)
    out["Support"] = fam["Support"].to_numpy()
    promoter_values = fam["Promoter_Key"].astype(object).to_numpy(copy=True)
    promoter_values[promoter_values == "Unknown"] = np.nan
    out["Promoter_Metal"] = promoter_values
    out["Metal_Loading_Method"] = fam["Metal_Loading_Method"].to_numpy()
    out["Precursor_Family"] = fixed["Precursor_Family"]
    out["Support_Prep_Method"] = fixed["Support_Prep_Method"]
    out["has_promoter"] = (track == "promoted")
    out["Family"] = fam["Family"].to_numpy()
    out["Dataset_Count"] = fam["Dataset_Count"].to_numpy()
    out["Track"] = track

    props = out["Promoter_Metal"].map(PROMOTER_PROPS)
    for pos, col in enumerate(("prom_electronegativity", "prom_ionic_radius", "prom_atomic_mass", "prom_first_ionization")):
        out[col] = props.map(lambda x: x[pos] if isinstance(x, tuple) else np.nan)
    out["prom_missing"] = props.isna().astype(int)
    support = out["Support"].map(SUPPORT_PROPS)
    out["support_type"] = support.map(lambda x: x[0] if isinstance(x, tuple) else "unknown")
    out["support_acid_base"] = support.map(lambda x: x[1] if isinstance(x, tuple) else 0)
    out["support_reducible"] = support.map(lambda x: x[2] if isinstance(x, tuple) else 0)
    return out


def evaluate(frame: pd.DataFrame, bundle: dict, nn: NearestNeighbors,
             ad_imputer: SimpleImputer, ad_scaler: RobustScaler,
             ad_threshold: float, low: np.ndarray, high: np.ndarray) -> pd.DataFrame:
    features = bundle["raw_feature_names"]
    raw = bundle["pipeline"].predict(frame[features])
    ad_matrix = ad_scaler.transform(ad_imputer.transform(frame[AD_COLUMNS]))
    distances = nn.kneighbors(ad_matrix, n_neighbors=5, return_distance=True)[0].mean(axis=1)
    within = distances <= ad_threshold
    out_of_range = (raw < 0) | (raw > 100)
    numeric = frame[OPT_COLUMNS].to_numpy(float)
    span = np.maximum(high - low, 1e-9)
    boundary = ((numeric - low) / span <= 0.02).any(axis=1) | ((high - numeric) / span <= 0.02).any(axis=1)
    result = frame.copy()
    result["Raw_Prediction"] = raw
    result["Predicted_Conversion_%"] = np.clip(raw, 0, 100)
    result["AD_Mean_5NN_Distance"] = distances
    result["Within_AD"] = within
    result["Out_of_Physical_Range"] = out_of_range
    result["Boundary_Flag"] = boundary
    result["Unknown_Label_Flag"] = False
    result["Optimization_Score"] = raw - (~within) * 100.0 - out_of_range * 200.0
    return result


def run_pso(rng: np.random.Generator, bundle: dict, families: pd.DataFrame, temp: float, fixed: dict,
            track: str, low: np.ndarray, high: np.ndarray, nn: NearestNeighbors,
            ad_imputer: SimpleImputer, ad_scaler: RobustScaler, ad_threshold: float) -> pd.DataFrame:
    fam_low, fam_high = 0.0, float(max(0, len(families) - 1))
    lower = np.r_[low, fam_low]
    upper = np.r_[high, fam_high]
    if track == "unpromoted":
        promoter_idx = OPT_COLUMNS.index("Promoter_Loading_wt%")
        lower[promoter_idx] = upper[promoter_idx] = 0.0
    pos = rng.uniform(lower, np.where(upper > lower, upper, lower + 1e-12), size=(N_PARTICLES, len(lower)))
    vel = np.zeros_like(pos)
    pbest = pos.copy(); pbest_score = np.full(N_PARTICLES, -np.inf)
    gbest = pos[0].copy(); gbest_score = -np.inf
    for _ in range(N_ITER):
        scored = evaluate(build_frame(pos, families, temp, fixed, track), bundle, nn, ad_imputer, ad_scaler, ad_threshold, low, high)
        score = scored["Optimization_Score"].to_numpy()
        improved = score > pbest_score
        pbest[improved] = pos[improved]
        pbest_score[improved] = score[improved]
        best_idx = int(np.argmax(pbest_score))
        if pbest_score[best_idx] > gbest_score:
            gbest_score = float(pbest_score[best_idx]); gbest = pbest[best_idx].copy()
        r1 = rng.random(pos.shape); r2 = rng.random(pos.shape)
        vel = 0.72 * vel + 1.45 * r1 * (pbest - pos) + 1.45 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, lower, upper)
    result = evaluate(build_frame(pbest, families, temp, fixed, track), bundle, nn, ad_imputer, ad_scaler, ad_threshold, low, high)
    result["Search"] = "PSO"
    return result


def main() -> None:
    PSO_DIR.mkdir(parents=True, exist_ok=True)
    table_dir = PSO_DIR / "table_temp_sweep"; table_dir.mkdir(exist_ok=True)
    with (TRAIN_DIR / "best_model.pkl").open("rb") as fh:
        bundle = pickle.load(fh)
    df = pd.read_csv(PRIMARY_DATA, encoding="utf-8-sig")
    ad_imputer = SimpleImputer(strategy="median")
    ad_scaler = RobustScaler()
    train_matrix = ad_scaler.fit_transform(ad_imputer.fit_transform(df[AD_COLUMNS]))
    nn = NearestNeighbors(n_neighbors=6).fit(train_matrix)
    train_dist = nn.kneighbors(train_matrix, return_distance=True)[0][:, 1:].mean(axis=1)
    ad_threshold = float(np.quantile(train_dist, 0.95))
    low, high, bounds_df = observed_bounds(df)
    bounds_df.to_csv(PSO_DIR / "optimization_bounds.csv", index=False, encoding="utf-8-sig")
    fixed = assumptions(df)
    pd.DataFrame([fixed]).to_csv(PSO_DIR / "operating_assumptions.csv", index=False, encoding="utf-8-sig")
    families = family_tables(df)
    pd.concat([x for x in families.values()], ignore_index=True).to_csv(PSO_DIR / "eligible_families.csv", index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(SEED)
    pso_parts, random_parts = [], []
    for temp in TEMP_FIXED_LIST:
        for track in ("promoted", "unpromoted"):
            fam = families[track]
            pso_parts.append(run_pso(rng, bundle, fam, temp, fixed, track, low, high, nn, ad_imputer, ad_scaler, ad_threshold))
            lower = np.r_[low, 0.0]; upper = np.r_[high, float(max(0, len(fam) - 1))]
            if track == "unpromoted":
                idx = OPT_COLUMNS.index("Promoter_Loading_wt%")
                lower[idx] = upper[idx] = 0.0
            positions = rng.uniform(lower, np.where(upper > lower, upper, lower + 1e-12), size=(RANDOM_SAMPLES_PER_TRACK_TEMP, len(lower)))
            random = evaluate(build_frame(positions, fam, temp, fixed, track), bundle, nn, ad_imputer, ad_scaler, ad_threshold, low, high)
            random["Search"] = "Random"
            random_parts.append(random)

    pso = pd.concat(pso_parts, ignore_index=True)
    random = pd.concat(random_parts, ignore_index=True)
    pso = pso.sort_values(["Reaction_Temp_C", "Optimization_Score"], ascending=[True, False]).reset_index(drop=True)
    pso["Rank"] = pso.groupby("Reaction_Temp_C").cumcount() + 1
    random = random.sort_values(["Reaction_Temp_C", "Optimization_Score"], ascending=[True, False]).reset_index(drop=True)
    random["Rank"] = random.groupby("Reaction_Temp_C").cumcount() + 1
    pso.to_csv(PSO_DIR / "pso_temp_sweep_all.csv", index=False, encoding="utf-8-sig")
    random.to_csv(PSO_DIR / "random_search_scores.csv", index=False, encoding="utf-8-sig")

    valid_pso = pso[pso["Within_AD"] & ~pso["Out_of_Physical_Range"]]
    top = valid_pso.groupby("Reaction_Temp_C", group_keys=False).head(10).copy()
    top.to_csv(table_dir / "pso_tableS10_top_candidates.csv", index=False, encoding="utf-8-sig")
    summary = valid_pso.groupby("Reaction_Temp_C", as_index=False).agg(
        Best_Predicted_Conversion=("Predicted_Conversion_%", "max"),
        Median_Predicted_Conversion=("Predicted_Conversion_%", "median"),
        Valid_Candidates=("Within_AD", "size"),
        Boundary_Flag_Rate=("Boundary_Flag", "mean"),
    )
    summary.to_csv(table_dir / "pso_table4_main.csv", index=False, encoding="utf-8-sig")
    frequency = top.groupby(["Family", "Track"], as_index=False).agg(
        Top10_Count=("Family", "size"), Temperatures=("Reaction_Temp_C", lambda s: ", ".join(map(str, sorted(set(s))))),
        Best_Prediction=("Predicted_Conversion_%", "max"),
    ).sort_values(["Top10_Count", "Best_Prediction"], ascending=False)
    frequency.to_csv(table_dir / "pso_tableS11_top20_frequency.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "model": bundle["name"], "data_sha256": bundle["data_sha256"],
        "ad_method": "mean 5-nearest-neighbor distance in robust-scaled continuous experimental space",
        "ad_columns": AD_COLUMNS,
        "ad_threshold_mean_5nn": ad_threshold, "n_particles": N_PARTICLES,
        "n_iterations": N_ITER, "random_samples_per_track_temperature": RANDOM_SAMPLES_PER_TRACK_TEMP,
        "temperatures": TEMP_FIXED_LIST, "fixed_operating_conditions": fixed,
        "pso_rows": len(pso), "random_rows": len(random),
    }
    (PSO_DIR / "pso_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
