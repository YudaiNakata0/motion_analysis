#!/usr/bin/env python3
"""Random-group cross validation for `learning_source/velocity.csv`.

The CSV rows are randomly assigned to 4 groups of equal size, then grouped
cross validation is performed by holding out one random group at a time.

This differs from the block-wise KFold script: the train/test split is still
4-fold, but the fold membership is randomized instead of following the source
row order.

Outputs:
- fold accuracy bar plot
- normalized confusion matrix from out-of-fold predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold


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

EXPECTED_ROWS = 80
N_GROUPS = 4


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["emotion", *FEATURE_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")
    return df


def assign_random_groups(num_rows: int, n_groups: int = N_GROUPS, random_state: int = 42) -> np.ndarray:
    if num_rows != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} data rows, got {num_rows}. "
            "This script assumes the file has 80 rows."
        )
    if num_rows % n_groups != 0:
        raise ValueError(f"Row count ({num_rows}) must be divisible by n_groups ({n_groups})")

    rng = np.random.default_rng(random_state)
    indices = np.arange(num_rows)
    rng.shuffle(indices)

    groups = np.empty(num_rows, dtype=int)
    block_size = num_rows // n_groups
    for group_id in range(n_groups):
        start = group_id * block_size
        end = start + block_size
        groups[indices[start:end]] = group_id
    return groups


def plot_fold_accuracies(fold_scores: Sequence[float], out_path: Path, show: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    folds = np.arange(1, len(fold_scores) + 1)
    bars = ax.bar(folds, fold_scores, color="tab:green", alpha=0.85)
    mean_score = float(np.mean(fold_scores))
    ax.axhline(mean_score, color="tab:red", linestyle="--", label=f"mean = {mean_score:.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(folds)
    ax.set_title("Random-group 4-fold accuracy")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for bar, score in zip(bars, fold_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    out_path: Path,
    show: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(cm_norm, cmap="Greens", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Out-of-fold confusion matrix (normalized)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-group cross validation for velocity.csv")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("learning_source/velocity.csv"),
        help="Path to the input CSV",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state used to shuffle rows into groups and for the classifier",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of trees in the random forest",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("result_plot_classification/classification_random_group_cv"),
        help="Directory where plots will be saved",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively",
    )
    args = parser.parse_args()

    df = load_data(args.input)
    X = df[FEATURE_COLUMNS]
    y = df["emotion"].astype(str)

    groups = assign_random_groups(len(df), n_groups=N_GROUPS, random_state=args.random_state)
    gkf = GroupKFold(n_splits=N_GROUPS)

    fold_scores: List[float] = []
    oof_true: List[str] = []
    oof_pred: List[str] = []

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Randomly assigned rows into {N_GROUPS} groups with random_state={args.random_state}")

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = RandomForestClassifier(
            random_state=args.random_state,
            n_estimators=args.n_estimators,
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        score = accuracy_score(y_test, pred)
        fold_scores.append(score)
        oof_true.extend(y_test.tolist())
        oof_pred.extend(pred.tolist())

        held_out_groups = sorted(set(groups[test_idx].tolist()))
        held_out_groups_str = ", ".join(str(g + 1) for g in held_out_groups)
        print(f"Fold {fold_idx}: test group {held_out_groups_str} | accuracy = {score:.3f}")

    labels = sorted(y.unique())
    overall_accuracy = accuracy_score(oof_true, oof_pred)
    print(f"Overall out-of-fold accuracy: {overall_accuracy:.3f}")
    print("Classification report:\n" + classification_report(oof_true, oof_pred, labels=labels, zero_division=0))

    plot_fold_accuracies(
        fold_scores,
        out_path=args.out_dir / "fold_accuracies.png",
        show=args.show,
    )
    plot_confusion_matrix(
        oof_true,
        oof_pred,
        labels=labels,
        out_path=args.out_dir / "confusion_matrix.png",
        show=args.show,
    )

    print(f"Saved plots under: {args.out_dir}")


if __name__ == "__main__":
    main()
