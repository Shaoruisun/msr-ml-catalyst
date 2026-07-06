"""t-SNE projection aligned to the saved v3 holdout manifest."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


SCRIPT_DIR = Path(__file__).resolve().parent
ML_ROOT = SCRIPT_DIR.parent
PRIMARY_DATA = ML_ROOT / "data_clean" / "prep_output" / "ni_model_primary.csv"
TRAIN_DIR = ML_ROOT / "train"
TSNE_DIR = SCRIPT_DIR / "output"
ARTIFACT_DIR = TRAIN_DIR / "artifacts"
TABLE_DIR = TRAIN_DIR / "tables"

SEED = 42
OKABE_ITO = {
    "train": "#0072B2",
    "test": "#E69F00",
    "promoted": "#009E73",
    "unpromoted": "#CC79A7",
    "reference": "#222222",
}


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
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
    TSNE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "svg.fonttype": "none", "axes.unicode_minus": False,
    })
    matrix = np.load(ARTIFACT_DIR / "X_full.npy")
    data = pd.read_csv(PRIMARY_DATA, encoding="utf-8-sig")
    split = pd.read_csv(TABLE_DIR / "split_manifest.csv", encoding="utf-8-sig")
    pred = pd.read_csv(TABLE_DIR / "holdout_predictions.csv", encoding="utf-8-sig")
    if len(matrix) != len(data):
        raise ValueError("X_full.npy and ni_model_primary.csv row counts differ")
    if set(split["row_id"]) != set(data["row_id"]) or set(pred["row_id"]) != set(data["row_id"]):
        raise ValueError("t-SNE inputs do not cover the same row_id values")

    perplexity = min(30, max(5, (len(data) - 1) // 3))
    coords = TSNE(
        n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
        max_iter=1000, random_state=SEED,
    ).fit_transform(matrix)
    out = data[["row_id", "Source_File", "Canonical_Catalyst_ID", "Catalyst", "MeOH_Conversion_%"]].copy()
    out["TSNE1"] = coords[:, 0]
    out["TSNE2"] = coords[:, 1]
    out = out.merge(split[["row_id", "Set"]], on="row_id", how="left", validate="one_to_one")
    out = out.merge(pred[["row_id", "Actual", "Predicted", "Residual", "Abs_Residual"]], on="row_id", how="left", validate="one_to_one")
    out.to_csv(TSNE_DIR / "tsne_source.csv", index=False, encoding="utf-8-sig")

    # Train/Test coverage.
    fig, ax = plt.subplots(figsize=(4.6, 3.7), constrained_layout=True)
    for label, color, marker in [("Train", OKABE_ITO["train"], "o"), ("Test", OKABE_ITO["test"], "s")]:
        sub = out[out["Set"].eq(label)]
        ax.scatter(sub["TSNE1"], sub["TSNE2"], s=18 if label == "Train" else 24,
                   c=color, marker=marker, alpha=0.58 if label == "Train" else 0.82,
                   edgecolors="#222222", linewidths=0.25, label=f"{label} (n = {len(sub)})")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.legend(frameon=False)
    style_axis(ax)
    save(fig, TSNE_DIR / "fig_tsne_train_test", "tsne_train_test")

    # Conversion map.
    fig, ax = plt.subplots(figsize=(4.6, 3.7), constrained_layout=True)
    sc = ax.scatter(out["TSNE1"], out["TSNE2"], c=out["MeOH_Conversion_%"], cmap="viridis",
                    vmin=0, vmax=100, s=20, alpha=0.78, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("MeOH conversion (%)")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
    style_axis(ax)
    save(fig, TSNE_DIR / "fig_tsne_conversion", "tsne_conversion")

    # Error map.
    fig, ax = plt.subplots(figsize=(4.6, 3.7), constrained_layout=True)
    vmax = max(1.0, float(out["Abs_Residual"].quantile(0.98)))
    sc = ax.scatter(out["TSNE1"], out["TSNE2"], c=out["Abs_Residual"], cmap="magma",
                    vmin=0, vmax=vmax, s=20, alpha=0.82, edgecolors="none")
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Absolute residual (percentage points)")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
    style_axis(ax)
    save(fig, TSNE_DIR / "fig_tsne_error", "tsne_error")

    metadata = {
        "rows": len(out), "train_rows": int(out["Set"].eq("Train").sum()),
        "test_rows": int(out["Set"].eq("Test").sum()), "features": int(matrix.shape[1]),
        "perplexity": perplexity, "random_state": SEED,
    }
    (TSNE_DIR / "tsne_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

