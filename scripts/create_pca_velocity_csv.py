#!/usr/bin/env python3
"""Create a PCA-reduced CSV from `learning_source/velocity.csv`.

The input data contains 15 motion features plus the `emotion` label.
This script standardizes the feature columns, applies PCA, and saves a new CSV
with principal component columns and the label.

Default behavior:
- input:  learning_source/velocity.csv
- output: result_csv/velocity_pca.csv
- n_components: 2

Use `--n-components` to change the number of PCA dimensions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "Body",
    "Head",
    "UpperBody",
    "LeftArm1",
    "LeftArm2",
    "RightArm1",
    "RightArm2",
    "LeftLeg1",
    "LeftLeg2",
    "RightLeg1",
    "RightLeg2",
    "LeftHand",
    "RightHand",
    "LeftFoot",
    "RightFoot",
    "Body_translational",
]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["emotion", *FEATURE_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    return df


def run_pca(df: pd.DataFrame, n_components: int):
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["emotion"].astype(str).to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, y, pca


def save_csv(out_path: Path, X_pca, y, pca: PCA) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    out_df = pd.DataFrame(X_pca, columns=columns)
    out_df.insert(0, "emotion", y)
    out_df.to_csv(out_path, index=False)

    variance_info = ", ".join(
        f"PC{i+1}={ratio:.4f}" for i, ratio in enumerate(pca.explained_variance_ratio_)
    )
    print(f"Explained variance ratio: {variance_info}")


def save_scatter_plot(out_path: Path, X_pca, y, pca: PCA) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = sorted(pd.unique(y))
    cmap = plt.get_cmap("tab10")
    color_map = {label: cmap(i % 10) for i, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for label in labels:
        mask = y == label
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=55,
            alpha=0.85,
            label=label,
            color=color_map[label],
            edgecolors="white",
            linewidths=0.5,
        )

    pc1_ratio = pca.explained_variance_ratio_[0]
    pc2_ratio = pca.explained_variance_ratio_[1]
    ax.set_xlabel("Variable 1")
    ax.set_ylabel("Variable 2")
    ax.set_title("PCA scatter plot of velocity features")
    ax.grid(True, alpha=0.3)
    ax.legend(title="emotion", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a PCA-reduced CSV from velocity.csv")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("learning_source/velocity.csv"),
        help="Path to the input CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("learning_source/velocity_pca.csv"),
        help="Path to the output PCA CSV",
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=Path("learning_source/velocity_pca_scatter.png"),
        help="Path to the output PCA scatter plot",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of principal components to keep",
    )

    args = parser.parse_args()

    if args.n_components <= 0:
        raise ValueError("--n-components must be positive")

    df = load_data(args.input)
    X_pca, y, pca = run_pca(df, n_components=args.n_components)
    save_csv(args.output, X_pca, y, pca)
    if args.n_components >= 2:
        save_scatter_plot(args.out_plot, X_pca, y, pca)
        print(f"Saved PCA scatter plot to {args.out_plot}")
    else:
        print("Skipping scatter plot because n_components < 2")

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Saved PCA CSV to {args.output}")


if __name__ == "__main__":
    main()
