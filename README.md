# NiMSR Miner: ML-Driven Inverse Design of Ni-Based Methanol Steam Reforming Catalysts


## Overview

A multimodal literature-mining and machine-learning workflow for the rational design of Ni-based methanol steam reforming (MSR) catalysts. LLM-based extraction builds a curated experimental dataset; a source-stratified 80/20 split trains seven regression candidates under a gap-constrained acceptance rule; the selected Gradient Boosting surrogate (Test R² = 0.79) is interpreted with SHAP and t-SNE and embedded in an applicability-domain-constrained particle swarm optimizer that proposes high-conversion catalyst families across the 300–500 °C operating range.

## Installation

```bash
git clone https://github.com/<user>/msr-ml-catalyst
cd msr-ml-catalyst
conda create -n nim_sr python=3.10
conda activate nim_sr
pip install -r requirements.txt
```

## Usage

Run the scripts in order from the repository root:

```bash
python ML/data_clean/data_clean.py        # Step 1: build modelling dataset
python ML/train/model_train.py            # Step 2: train and select model
python ML/t-SNE/tsne.py                   # Step 3: t-SNE visualization
python ML/SHAP/shap.py                    # Step 4: SHAP + CV ablation
python ML/PSO/pso.py                      # Step 5: PSO inverse design
```

Each script is self-contained and writes its outputs to a sibling folder (`prep_output/`, `reports/`, `artifacts/`, `output/`).

## Dataset

- **Source**: 81 peer-reviewed publications, extracted by `extraction/NiMSR-Miner.py` into `data/dataset_all.csv` (3401 raw records).
- **Curated**: 579 primary Ni-MSR records from 33 sources, produced by `ML/data_clean/data_clean.py`.
- **Target**: `MeOH_Conversion_%`.

## Repository Structure

```
msr-ml-catalyst/
├── data/dataset_all.csv                  # raw dataset
├── extraction/NiMSR-Miner.py             # LLM-assisted PDF extraction
└── ML/
    ├── data_clean/data_clean.py
    ├── train/
    │   ├── model_train.py
    │   └── tables/regularized_model_comparison.csv
    ├── t-SNE/tsne.py
    ├── SHAP/shap.py
    └── PSO/pso.py
```


## License

MIT — see `LICENSE`.
