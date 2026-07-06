# NiMSR Miner: ML-Driven Inverse Design of Ni-Based Methanol Steam Reforming Catalysts


## Overview

A multimodal literature-mining and machine-learning workflow for the rational design of Ni-based methanol steam reforming (MSR) catalysts. LLM-based extraction builds a curated experimental dataset; a source-stratified 80/20 split trains seven regression models under a gap-constrained acceptance rule; the selected Gradient Boosting surrogate is interpreted with SHAP and t-SNE and embedded in an applicability-domain-constrained particle swarm optimizer (PSO) that proposes high-conversion catalyst families across the 300–500 °C operating range.

## Installation

```bash
git clone https://github.com/<user>/msr-ml-catalyst
cd msr-ml-catalyst
conda create -n nim_sr python=3.10
conda activate nim_sr
pip install -r requirements.txt
```

Python 3.10+ is required (scikit-learn 1.6 / numpy 2.1).

## Dataset

- **Source**: 81 peer-reviewed publications extracted by `extraction/NiMSR-Miner.py` into `data/dataset_all.csv` (3401 raw records, 56 columns).
- **Curated subsets** (produced by `ML/data_clean/data_clean.py`):
  - 1229 cleaned Ni-MSR records (42 sources)
  - 579 primary records (33 sources) used for modelling
- **Target**: `MeOH_Conversion_%`
- **Features**: 41 descriptors organized into interpretable layers:
  - 22 numerical descriptors (reaction conditions, Ni/promoter loading, treatment temperatures/times, engineered features, promoter elemental properties)
  - 6 categorical descriptors (Support, Promoter_Metal, Metal_Loading_Method, Precursor_Family, Support_Prep_Method, support_type)
  - 3 binary descriptors (has_promoter, prom_missing, support_reducible)
- **Split**: 80:20 source-stratified row-random split (`Source_File` used only for stratification, not as a feature).

## Usage

Run the scripts in order from the repository root:

```bash
# Step 1: build the auditable modelling dataset
python ML/data_clean/data_clean.py

# Step 2: train and select the publication model (~50 s)
python ML/train/model_train.py

# Step 3: t-SNE visualization of the chemical space
python ML/t-SNE/tsne.py

# Step 4: SHAP interpretability + training-CV ablation
python ML/SHAP/shap.py

# Step 5: PSO inverse catalyst design (5 temperature tracks)
python ML/PSO/pso.py
```

Each script is self-contained and writes its outputs to a sibling `output/` (or `prep_output/`, `reports/`, `artifacts/`) directory. Run-time assumes 8 GB RAM and a single CPU.

## Key Results

Seven regression candidates were compared under source-stratified 5-fold CV plus a one-shot 80:20 holdout. The Gradient Boosting model was selected under a publication gate requiring CV R² ≥ 0.75, holdout Test R² ≥ 0.75, and Train–Test R² gap < 0.12 (gap = 0.1090).

| CV Rank | Model | CV R² | Test R² | Test MAE | Test RMSE | Train–Test Gap | Selected |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 1 | Bagging | 0.7519 ± 0.0645 | 0.7731 | 10.32 | 17.20 | 0.1388 | |
| 2 | **Gradient Boosting** | **0.7502 ± 0.0606** | **0.7884** | **10.97** | **16.61** | **0.1090** | **✓** |
| 3 | SVR | 0.7376 ± 0.0427 | 0.7242 | 10.66 | 18.96 | 0.1595 | |
| 4 | AdaBoost | 0.6993 ± 0.0574 | 0.7120 | 13.51 | 19.37 | 0.1009 | |
| 5 | XGBoost | 0.6785 ± 0.0819 | 0.7462 | 13.15 | 18.19 | 0.0432 | |
| 6 | Random Forest | 0.6502 ± 0.0824 | 0.7106 | 13.91 | 19.42 | 0.0346 | |
| 7 | Extra Trees | 0.6445 ± 0.0696 | 0.7371 | 13.25 | 18.51 | −0.0138 | |

Multi-seed stability (10 random split seeds, fixed selected model/hyperparameters): Test R² = 0.7637 ± 0.0548, gap = 0.1225 ± 0.0581.

### PSO inverse design

Applicability-domain-constrained PSO (35 particles × 110 iterations, 2500 random samples per temperature) over five fixed reaction temperatures yields the following top catalyst family per temperature track:

| Reaction Temp (°C) | Best Predicted Conversion (%) | Top Catalyst Family | Track |
|---:|---:|---|---|
| 300 | 99.997 | Unknown support · Cu · Other loading | promoted |
| 350 | 99.9997 | CeO2 · (no promoter) · Impregnation | unpromoted |
| 400 | 99.999 | Unknown support · Cu · Other loading | promoted |
| 450 | 99.998 | Al2O3 · Cu · Impregnation | promoted |
| 500 | 99.994 | MgO · (no promoter) · Impregnation | unpromoted |

PSO candidate scores are Gradient Boosting surrogate predictions used for ranking within the literature-derived design domain; they are not experimental validation results.

## Repository Structure

```
msr-ml-catalyst/
├── README.md
├── requirements.txt
├── LICENSE                              # MIT
├── data/
│   └── dataset_all.csv                  # 3401 raw records (input)
├── extraction/
│   └── NiMSR-Miner.py                   # LLM-assisted PDF extraction (optional)
└── ML/
    ├── data_clean/
    │   └── data_clean.py                # Step 1: build modelling dataset
    ├── train/
    │   ├── model_train.py               # Step 2: train + select model
    ├── t-SNE/
    │   └── tsne.py                      # Step 3: t-SNE visualization
    ├── SHAP/
    │   └── shap.py                      # Step 4: SHAP + CV ablation
    └── PSO/
        └── pso.py                       # Step 5: PSO inverse design
```

Outputs (preprocessed datasets, model artifacts, SHAP values, PSO candidates, figures) are regenerated under each script's sibling folder and are excluded from version control via `.gitignore`.

## Data Provenance

`data/dataset_all.csv` SHA-256: `6beb6038e441f71d52376b43fe8dac1069093fd7f9ef00ce955a68bf3918f929`

The primary modelling dataset written by `data_clean.py` carries SHA-256: `53fa9e2dd741d9b86e6d776bd1a398004d2951d38e628de698980ccf558e83b9`, which `model_train.py` validates against `dataset_manifest.json` before fitting.


## License

This project is licensed under the MIT License — see `LICENSE`.
