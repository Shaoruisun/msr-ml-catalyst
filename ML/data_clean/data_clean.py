"""Build the auditable v3 Ni-catalyst modelling dataset from dataset_all.csv.

The raw CSV is never modified by this script.  Source-level CSV repairs must be
applied before cleaning; this script records downstream exclusions and
deterministic/domain cleaning only.  Medians, one-hot encoders and scalers are
deliberately deferred to the training Pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ML_ROOT.parent
RAW_CSV = PROJECT_ROOT / "data" / "dataset_all.csv"
PREP_DIR = SCRIPT_DIR / "prep_output"
ALL_CLEAN_DATA = PREP_DIR / "ni_all_cleaned.csv"
PRIMARY_DATA = PREP_DIR / "ni_model_primary.csv"
DATASET_MANIFEST = PREP_DIR / "dataset_manifest.json"

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

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


EXPECTED_COLUMNS = 56
NO_LABELS = {"", "nan", "none", "unknown", "n/s", "n/a", "na", "not specified"}
INVALID_PROMOTER_COMPONENTS = NO_LABELS | {"o", "n", "c", "h", "s", "f", "p", "cl", "br", "i", "al"}

VALID_RANGES = {
    "Ni_Loading_wt%": (0.01, 80.0),
    "Promoter_Loading_wt%": (0.0, 80.0),
    "Calcination_Temp_C": (100.0, 900.0),
    "Calcination_Time_h": (0.1, 48.0),
    "Reduction_Temp_C": (100.0, 900.0),
    "Reduction_Time_h": (0.1, 48.0),
    "Dry_Temp_C": (20.0, 200.0),
    "Dry_Time_h": (0.1, 48.0),
    "Reaction_Temp_C": (150.0, 700.0),
    "MeOH_Conversion_%": (0.0, 100.0),
    "S_C_Ratio": (0.5, 10.0),
    "TOS_h": (0.0, 500.0),
    "Pressure_bar": (0.5, 50.0),
    "GHSV_mL_g_h": (1.0, 200000.0),
    "WHSV_h_inv": (0.01, 500.0),
}

PROMOTER_PROPS = {
    "Ce": (1.12, 1.01, 140.12, 5.539),
    "Co": (1.88, 0.745, 58.93, 7.881),
    "Cu": (1.90, 0.73, 63.55, 7.726),
    "In": (1.78, 0.80, 114.82, 5.786),
    "La": (1.10, 1.032, 138.91, 5.577),
    "Mg": (1.31, 0.72, 24.31, 7.646),
    "Mo": (2.16, 0.59, 95.95, 7.092),
    "Pd": (2.20, 0.86, 106.42, 8.337),
    "Pt": (2.28, 0.80, 195.08, 9.000),
    "Sn": (1.96, 0.69, 118.71, 7.344),
    "Zn": (1.65, 0.74, 65.38, 9.394),
}

SUPPORT_PROPS = {
    "Al2O3": ("oxide", 2, 0), "Al2O4": ("oxide", 2, 0),
    "CN": ("nitride", 0, 0), "h-BN": ("nitride", 0, 0),
    "CNTs": ("carbon", 0, 0), "Carbon": ("carbon", 0, 0),
    "CeO2": ("oxide", 3, 1), "Cement-Clay": ("composite", 0, 0),
    "LDH": ("ldh", 3, 0), "MgAl": ("ldh", 3, 0),
    "MgO": ("oxide", 3, 0), "Mo2C": ("carbide", 0, 1),
    "NaF": ("salt", 0, 0), "SnO2": ("oxide", 2, 1),
    "TiO2": ("oxide", 2, 1), "TiO2-CeO2": ("composite", 2, 1),
    "ZnO": ("oxide", 2, 1), "ZrO2": ("oxide", 2, 1),
    "Perovskite": ("oxide", 2, 1), "Spinel": ("oxide", 2, 1),
    "Other": ("unknown", 0, 0), "Unknown": ("unknown", 0, 0),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    true_values = {"true", "1", "yes", "y"}
    false_values = {"false", "0", "no", "n", "", "nan", "none"}
    s = series.fillna("").astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out[s.isin(true_values)] = True
    out[s.isin(false_values)] = False
    return out


def normalize_support(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    s = str(value).replace("₂", "2").replace("₃", "3").replace("₄", "4")
    sl = re.sub(r"\s+", "", s.lower())
    # Composite/specific forms must be checked before their components.
    if any(x in sl for x in ("tio2-ceo2", "tio2ceo2", "ceo2-tio2")): return "TiO2-CeO2"
    if "activatedcarbon" in sl: return "Carbon"
    if any(x in sl for x in ("cnt", "carbonnanotube")): return "CNTs"
    if any(x in sl for x in ("carbon", "activated")): return "Carbon"
    if "al2o4" in sl: return "Al2O4"
    if any(x in sl for x in ("al2o3", "alumina")): return "Al2O3"
    if "ceo2" in sl: return "CeO2"
    if "zro2" in sl: return "ZrO2"
    if "tio2" in sl or "ti-o" in sl: return "TiO2"
    if "zno" in sl: return "ZnO"
    if any(x in sl for x in ("mgal", "mg-al", "mg/al")): return "MgAl"
    if "mgo" in sl: return "MgO"
    if any(x in sl for x in ("mo2c", "alpha-moc", "moc")): return "Mo2C"
    if any(x in sl for x in ("g-c3n4", "c3n4", "carbonnitride")) or s.strip().upper() == "CN": return "CN"
    if any(x in sl for x in ("h-bn", "boronnitride")): return "h-BN"
    if "sno2" in sl: return "SnO2"
    if any(x in sl for x in ("perovskite", "lacemno", "lamno")): return "Perovskite"
    if "spinel" in sl: return "Spinel"
    if any(x in sl for x in ("ldh", "hydrotalcite")): return "LDH"
    if any(x in sl for x in ("naf", "sodiumfluoride")): return "NaF"
    if any(x in sl for x in ("cement", "clay")): return "Cement-Clay"
    return "Unknown" if sl in NO_LABELS else "Other"


def normalize_method(value: object, catalyst: object) -> str:
    text = " ".join(x for x in (str(value) if pd.notna(value) else "", str(catalyst) if pd.notna(catalyst) else "") if x).lower()
    if "co-precip" in text or "coprecip" in text or "copre" in text: return "Co-precipitation"
    if "sol-gel" in text or "sol gel" in text: return "Sol-gel"
    if "incipient" in text or "wetness" in text or "impreg" in text or "impre" in text: return "Impregnation"
    if "hydrothermal" in text or "hydro" in text: return "Hydrothermal"
    if "ball" in text and "mill" in text: return "Ball milling"
    if "polyol" in text: return "Polyol"
    return "Unknown" if not text.strip() else "Other"


def normalize_promoter(value: object) -> object:
    if pd.isna(value) or str(value).strip().lower() in NO_LABELS:
        return np.nan
    parts = [p.strip() for p in re.split(r"[,/;]", str(value)) if p.strip()]
    parts = [p for p in parts if p.lower() != "ni" and p.lower() not in INVALID_PROMOTER_COMPONENTS]
    return ",".join(parts) if parts else np.nan


def catalyst_label_ni_value(value: object) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value)
    patterns = [r"(?i)(\d+(?:\.\d+)?)\s*wt\s*%[^\n]*ni", r"(?i)ni\s*(\d+(?:\.\d+)?)"]
    for pattern in patterns:
        match = re.search(pattern, s)
        if match:
            val = float(match.group(1))
            return val if 0 < val <= 80 else np.nan
    return np.nan


def append_exclusions(log: list[dict], df: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    for _, row in df.loc[mask, ["raw_row_id", "Source_File", "Canonical_Catalyst_ID", "Catalyst"]].iterrows():
        log.append({**row.to_dict(), "Reason": reason})


def main() -> None:
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    exclusions: list[dict] = []
    repair_log: list[str] = []

    df = pd.read_csv(RAW_CSV, low_memory=False)
    if df.shape[1] != EXPECTED_COLUMNS:
        raise ValueError(f"dataset_all.csv must contain {EXPECTED_COLUMNS} columns, found {df.shape[1]}")
    df.insert(0, "raw_row_id", np.arange(len(df), dtype=int))
    raw_rows = len(df)

    # The 2022_Al2O4_NiCu shifted-provenance issue is repaired in the raw
    # database.  Keep a guard here so the cleaning stage never silently performs
    # this source-level fix again.
    lit_text = df["is_literature_comparison"].fillna("").astype(str)
    shifted_provenance = df["Source_File"].eq("2022_Al2O4_NiCu_main.pdf") & ~lit_text.str.lower().isin({"true", "false", "0", "1", "nan", ""})
    if shifted_provenance.any():
        bad_rows = df.loc[shifted_provenance, "raw_row_id"].astype(str).tolist()
        raise ValueError(
            "dataset_all.csv still contains shifted provenance fields for "
            f"2022_Al2O4_NiCu_main.pdf raw rows: {', '.join(bad_rows)}. "
            "Apply the raw-database repair before running data_clean_v3.py."
        )

    for col in ("is_approximate_value", "is_range_like_value", "is_literature_comparison"):
        df[col] = as_bool(df[col])

    # Row-level Ni identity.  Mixed broadcast labels require Ni in the catalyst label.
    active = df["Active_Metal"].fillna("").astype(str)
    catalyst = df["Catalyst"].fillna("").astype(str)
    active_has_ni = active.str.contains(r"(?i)(?:^|[^A-Za-z])Ni(?:[^A-Za-z]|$)", regex=True)
    catalyst_has_ni = catalyst.str.contains(r"(?i)(?:^|[^A-Za-z])Ni(?:[^A-Za-z]|$)", regex=True)
    active_is_simple_ni = active.str.strip().str.fullmatch(r"(?i)Ni(?:[-/ ].*)?", na=False)
    # Most rows are reliable when Active_Metal is Ni.  Three comparison papers
    # broadcast a metal list/label across non-Ni series, so those require an
    # explicit Ni token in the row-level catalyst label.  The Cu-Ni-Al paper
    # uses the shorthand C0.95N0.05 and is the documented exception.
    ni_mask = active_has_ni & (
        active_is_simple_ni
        | catalyst_has_ni
        | df["Source_File"].eq("2020_Cu-Ni-Al_main.pdf")
    )
    mixed_broadcast_sources = {
        "2014_Mo2C_Pt Fe Co Ni_main.pdf",
        "2015_TiO2_Pd Ni Zn Co Cu Sn_main.pdf",
        "2020_Al2O3_Mn Fe Co Ni Cu Zn_main.pdf",
    }
    mixed_source = df["Source_File"].isin(mixed_broadcast_sources)
    ni_mask = ni_mask & (~mixed_source | catalyst_has_ni)
    append_exclusions(exclusions, df, ~ni_mask, "not_row_level_Ni_catalyst")
    df = df.loc[ni_mask].copy()
    ni_candidate_rows = len(df)
    ni_candidate_sources = int(df["Source_File"].nunique())

    # Numeric casting and conservative known patches.
    numeric_columns = list(VALID_RANGES)
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mo2c = df["Source_File"].eq("2014_Mo2C_Ni_main.pdf") & df["Ni_Loading_wt%"].isna()
    parsed = df.loc[mo2c, "Catalyst"].fillna("").str.extract(r"\((\d+(?:\.\d+)?)\)\s*$", expand=False)
    df.loc[mo2c, "Ni_Loading_wt%"] = pd.to_numeric(parsed, errors="coerce")
    zro = df["Source_File"].eq("2015_ZrO2_Ni-Cu_main.pdf") & df["Ni_Loading_wt%"].isna()
    df.loc[zro, "Ni_Loading_wt%"] = 20.0

    # Reaction-temperature propagation only when a unique value exists in the group.
    for keys in (["Source_File", "Canonical_Catalyst_ID"], ["Source_File"]):
        for _, idx in df.groupby(keys, dropna=False).groups.items():
            values = df.loc[idx, "Reaction_Temp_C"].dropna().unique()
            if len(values) == 1:
                df.loc[idx, "Reaction_Temp_C"] = df.loc[idx, "Reaction_Temp_C"].fillna(values[0])

    # Record and null implausible values; they remain visible in the audit files.
    for col, (low, high) in VALID_RANGES.items():
        bad = df[col].notna() & ~df[col].between(low, high)
        df[f"{col}_range_flag"] = bad.astype(int)
        if bad.any():
            repair_log.append(f"Set {int(bad.sum())} out-of-range {col} values to missing ({low}..{high})")
            df.loc[bad, col] = np.nan

    # Required response fields.
    required_bad = df[TARGET].isna() | df["Reaction_Temp_C"].isna()
    append_exclusions(exclusions, df, required_bad, "missing_or_invalid_target_or_reaction_temperature")
    df = df.loc[~required_bad].copy()

    # Domain normalization and deterministic feature engineering.
    df["Promoter_Metal"] = df["Promoter_Metal"].apply(normalize_promoter)
    no_promoter = df["Promoter_Metal"].isna()
    df.loc[no_promoter, "Promoter_Loading_wt%"] = 0.0
    df["Support"] = df["Support_Normalized"].where(df["Support_Normalized"].notna(), df["Support"]).apply(normalize_support)
    df["Metal_Loading_Method"] = [normalize_method(v, c) for v, c in zip(df["Metal_Loading_Method"], df["Catalyst"])]
    for col in ("Precursor_Family", "Support_Prep_Method"):
        df[col] = df[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        if col == "Precursor_Family":
            df[col] = df[col].str.lower().replace({"nitrate": "Nitrate", "chloride": "Chloride", "acetate": "Acetate", "organometallic": "Organometallic", "other": "Other", "unknown": "Unknown"})

    df["Catalyst_Label_Ni_Value"] = df["Catalyst"].apply(catalyst_label_ni_value)
    df["GHSV_log"] = np.log1p(df["GHSV_mL_g_h"])
    df["WHSV_log"] = np.log1p(df["WHSV_h_inv"])
    df["calc_reduc_temp_diff"] = df["Calcination_Temp_C"] - df["Reduction_Temp_C"]
    df["total_treatment_time_h"] = df[["Calcination_Time_h", "Reduction_Time_h", "Dry_Time_h"]].sum(axis=1, min_count=1)
    df["has_promoter"] = (~no_promoter).astype(int)

    first_promoter = df["Promoter_Metal"].fillna("").astype(str).str.split(",").str[0]
    props = first_promoter.map(PROMOTER_PROPS)
    for pos, col in enumerate(("prom_electronegativity", "prom_ionic_radius", "prom_atomic_mass", "prom_first_ionization")):
        df[col] = props.map(lambda x: x[pos] if isinstance(x, tuple) else np.nan)
    df["prom_missing"] = props.isna().astype(int)

    support_props = df["Support"].map(SUPPORT_PROPS)
    df["support_type"] = support_props.map(lambda x: x[0] if isinstance(x, tuple) else "unknown")
    df["support_acid_base"] = support_props.map(lambda x: x[1] if isinstance(x, tuple) else 0)
    df["support_reducible"] = support_props.map(lambda x: x[2] if isinstance(x, tuple) else 0)

    # Exclude records missing more than two central synthesis descriptors.
    key_missing_cols = ["Ni_Loading_wt%", "Calcination_Temp_C", "Reduction_Temp_C", "Calcination_Time_h"]
    excessive_missing = df[key_missing_cols].isna().sum(axis=1) > 2
    append_exclusions(exclusions, df, excessive_missing, "more_than_two_core_synthesis_fields_missing")
    df = df.loc[~excessive_missing].copy()

    # Keep comparison rows in the all-clean and audit outputs, not in the primary model.
    df["is_literature_comparison"] = df["is_literature_comparison"].fillna(False).astype(bool)
    all_clean = df.copy()
    comparison = df.loc[df["is_literature_comparison"]].copy()
    primary = df.loc[~df["is_literature_comparison"]].copy()

    # Experimental-condition collision audit. Catalyst text is included because it
    # often contains structured variants that the old canonical id collapsed.
    primary["Catalyst_Key"] = primary["Catalyst"].fillna(primary["Canonical_Catalyst_ID"]).fillna("Unknown").astype(str).str.strip().str.lower()
    collision_key = [
        "Source_File", "Catalyst_Key", "Support", "Promoter_Metal",
        "Ni_Loading_wt%", "Promoter_Loading_wt%", "Reaction_Temp_C",
        "S_C_Ratio", "TOS_h", "Pressure_bar", "GHSV_mL_g_h", "WHSV_h_inv",
        "Calcination_Temp_C", "Reduction_Temp_C", "Calcination_Time_h",
        "Reduction_Time_h", "Dry_Temp_C", "Dry_Time_h", "Metal_Loading_Method",
    ]
    grouped = primary.groupby(collision_key, dropna=False, sort=False)
    keep_rows: list[pd.Series] = []
    collision_rows: list[pd.DataFrame] = []
    for _, group in grouped:
        target_range = float(group[TARGET].max() - group[TARGET].min())
        if len(group) == 1:
            row = group.iloc[0].copy()
            row["Replicate_Count"] = 1
            row["Replicate_Target_Range"] = 0.0
            keep_rows.append(row)
        elif target_range <= 5.0:
            row = group.iloc[0].copy()
            row[TARGET] = float(group[TARGET].median())
            row["Replicate_Count"] = int(len(group))
            row["Replicate_Target_Range"] = target_range
            row["Original_Row_IDs"] = ";".join(map(str, group["raw_row_id"].tolist()))
            keep_rows.append(row)
        else:
            audit = group.copy()
            audit["Collision_Target_Range"] = target_range
            audit["Collision_Reason"] = "same_recorded_conditions_target_range_gt_5pp"
            collision_rows.append(audit)
            append_exclusions(exclusions, group, pd.Series(True, index=group.index), "unresolved_condition_collision_gt_5pp")

    primary_final = pd.DataFrame(keep_rows).reset_index(drop=True)
    primary_final.insert(0, "row_id", np.arange(len(primary_final), dtype=int))
    all_clean = all_clean.reset_index(drop=True)
    comparison = comparison.reset_index(drop=True)
    collision_audit = pd.concat(collision_rows, ignore_index=True) if collision_rows else pd.DataFrame(columns=list(primary.columns) + ["Collision_Target_Range", "Collision_Reason"])

    missing_features = [c for c in MODEL_FEATURES if c not in primary_final.columns]
    if missing_features:
        raise RuntimeError(f"Missing model features after cleaning: {missing_features}")

    # Stable outputs.
    all_clean.to_csv(ALL_CLEAN_DATA, index=False, encoding="utf-8-sig")
    primary_final.to_csv(PRIMARY_DATA, index=False, encoding="utf-8-sig")
    comparison.to_csv(PREP_DIR / "ni_comparison_audit.csv", index=False, encoding="utf-8-sig")
    collision_audit.to_csv(PREP_DIR / "condition_collision_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(exclusions).to_csv(PREP_DIR / "row_exclusion_log.csv", index=False, encoding="utf-8-sig")

    feature_schema = {
        "target": TARGET,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "binary_features": BINARY_FEATURES,
        "model_features": MODEL_FEATURES,
        "units": {
            "Reaction_Temp_C": "degC", "Ni_Loading_wt%": "wt%",
            "Promoter_Loading_wt%": "wt%", "S_C_Ratio": "molar ratio",
            "TOS_h": "h", "Pressure_bar": "bar", "GHSV_mL_g_h": "mL g-1 h-1",
            "WHSV_h_inv": "h-1", TARGET: "%",
        },
        "valid_ranges": VALID_RANGES,
        "collision_policy": "median if target range <=5 percentage points; otherwise audit/exclude",
    }
    (PREP_DIR / "feature_schema.json").write_text(json.dumps(feature_schema, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "version": "v3",
        "raw_csv": str(RAW_CSV),
        "raw_sha256": sha256(RAW_CSV),
        "raw_rows": raw_rows,
        "raw_columns": EXPECTED_COLUMNS,
        "ni_candidate_rows": ni_candidate_rows,
        "ni_candidate_sources": ni_candidate_sources,
        "all_clean_rows": int(len(all_clean)),
        "all_clean_sources": int(all_clean["Source_File"].nunique()),
        "comparison_rows": int(len(comparison)),
        "primary_rows": int(len(primary_final)),
        "primary_sources": int(primary_final["Source_File"].nunique()),
        "collision_rows": int(len(collision_audit)),
        "excluded_rows_logged": int(len(exclusions)),
        "primary_sha256": sha256(PRIMARY_DATA),
        "repairs": repair_log,
    }
    DATASET_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# v3 data quality report", "",
        f"- Raw database: {raw_rows} rows x {EXPECTED_COLUMNS} columns",
        f"- Row-level Ni candidates: {ni_candidate_rows} rows from {ni_candidate_sources} sources",
        f"- All cleaned records: {len(all_clean)} rows from {all_clean['Source_File'].nunique()} sources",
        f"- Literature-comparison audit: {len(comparison)} rows",
        f"- Primary modelling data: {len(primary_final)} rows from {primary_final['Source_File'].nunique()} sources",
        f"- Unresolved collision audit: {len(collision_audit)} rows",
        f"- Logged exclusions: {len(exclusions)}", "",
        "## Repairs and range checks", "",
        *[f"- {line}" for line in repair_log], "",
        "## Modelling contract", "",
        "Statistical imputation, one-hot encoding, scaling, splitting and model selection are not performed here.",
        "The training script must fit all statistical preprocessing inside each CV fold.",
    ]
    (PREP_DIR / "data_quality_report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
