# Identity-Resolved Multimodal Literature Mining for Evidence-Constrained Temperature Screening of Ni-Based Methanol Steam Reforming Catalysts
## Overview

A multimodal literature-mining and machine-learning workflow for the rational design of Ni-based
methanol steam reforming (MSR) catalysts. LLM-based extraction builds a source-review-verified
experimental database (6,341 records from 50 peer-reviewed sources); rule-based cleaning folds
evidence-approved duplicates and produces a modeling table of 1,091 rows from 46 papers. A pure
row-random 80/20 protocol trains nine regression candidates plus a training-OOF weighted blend
(selected on training-only 5-fold CV). The deployed blend reaches
**Test R² = 0.808 on the seed-42 holdout (0.767 ± 0.040 over seeds 42–51)** and is interpreted
with exact permutation-Shapley attribution, then used for applicability-domain-
constrained per-catalyst condition screening (particle-swarm-style sweeps) with conformalized
predictive uncertainty.

## Installation

```bash
git clone https://github.com/<user>/msr-ml-catalyst
cd msr-ml-catalyst
conda create -n nim_sr python=3.13
conda activate nim_sr
pip install -r requirements.txt
```

## Usage

Run the stages in order from the repository root (steps 2–4 need step 1's table; step 2 also
writes the uncertainty model used by step 4):

```bash
python extraction/NiMSR-Miner.py       # Step 0 (optional): LLM extraction of your own PDF corpus
python ML/data_clean/data_clean.py     # Step 1: build the modeling table
python ML/train/model_train.py         # Step 2: train + select, fit uncertainty, export OOF table
python ML/SHAP/shap.py                 # Step 3: permutation-Shapley interpretation
python ML/PSO/pso.py                   # Step 4: AD-constrained condition screening
```

Each stage is self-contained and writes its outputs to a sibling `output/` (or `prep_output/`)
folder. The shipped `output/` folders already contain the results of the authors' original run,
so steps 1, 3 and 4 can also be re-executed selectively on the shipped data and artifacts. The
training stage (step 2) carries a provenance gate bound to the authors' internal release
manifest and cannot be re-run from the public package alone. The shipped results were produced
with the exact package versions pinned in `requirements.txt`.

`extraction/NiMSR-Miner.py` calls an LLM API: set `MSR_API_KEY` (and optionally `MSR_BASE_URL`,
`MSR_TEXT_MODEL`, `MSR_VISION_MODEL`) in your environment — no key is stored in the repository.
Point `MSR_IN_DIR` at a folder of your PDFs (default: `data/pdfs/`).

## Dataset

- **Source**: 50 peer-reviewed publications, extracted by `extraction/NiMSR-Miner.py` into
  `data/dataset_all.csv` (6,341 verified records after source review; the exact 57-column
  export of the extraction script — the PDFs themselves are not redistributed).
- **Curated**: 1,091 primary Ni-MSR rows from 46 papers, produced by `ML/data_clean/data_clean.py`
  into `ML/data_clean/prep_output/ni_model_primary.csv`. The cleaning stage reads **only** `data/dataset_all.csv`:
  the published modeling selection (which rows enter the model and how evidence-confirmed
  duplicate representations fold together) is embedded in the script and bound to the dataset's
  SHA-256, so a changed dataset is rejected instead of silently reused.
- **Target**: `MeOH_Conversion_%` (31-feature scientific contract, all temperatures in °C).

## Results (deployed OOF weighted blend)

| Protocol | Train R² | Test R² | \|gap\| |
|---|---|---|---|
| Seed-42 single split | 0.887 | 0.808 | 0.079 |
| 10 seeds (42–51), mean ± SD | 0.887 ± 0.021 | 0.767 ± 0.040 | 0.119 ± 0.041 |

Per-seed values: `ML/train/output/tables/random_seed_stability.csv`;
per-row holdout predictions: `ML/train/output/tables/holdout_predictions.csv`.

## Repository Structure

```
msr-ml-catalyst/
├── data/                                  # released verified database (input)
│   └── dataset_all.csv                    # 6,341 records, 57-column extraction export
├── extraction/NiMSR-Miner.py              # LLM-assisted PDF extraction
└── ML/
    ├── data_clean/
    │   ├── data_clean.py                  # Step 1: cleaning (single-file edition)
    │   └── prep_output/ni_model_primary.csv
    ├── train/
    │   ├── model_train.py                 # Step 2: train/select + uncertainty + OOF export
    │   └── output/                        # models (.pkl), artifacts/, reports/, tables/
    ├── SHAP/shap.py         + output/     # Step 3: permutation-Shapley
    └── PSO/pso.py           + output/     # Step 4: condition screening
```

## Notes

- The shipped `output/` artifacts come from the authors' original run. Rerunning a stage
  regenerates its `*_metadata.json` provenance record (with your own local paths); these
  records are intentionally not part of the published repository.
- `uncertainty_model.pkl` (predictive intervals for the PSO stage) and `oof_predictions.csv`
  (per-row out-of-fold residual diagnostics) are both regenerated by `ML/train/model_train.py`.

## License

MIT — see `LICENSE`.
