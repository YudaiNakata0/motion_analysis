#!/usr/bin/env python3
"""Compute Body translational speed magnitude from Body.dat.

Body.dat format used here:
- column 1: time
- columns 2-4: xyz position
- columns 5-13: rotation matrix (ignored in this script)

The translational speed is computed as the magnitude of the velocity vector
between consecutive samples:

    v_i = ||(p_i - p_{i-1}) / (t_i - t_{i-1})||

Outputs:
- CSV with time and translational speed magnitude
- Plot of translational speed over time
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_body_dat(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, dtype=float)
    if data.size == 0:
        raise ValueError(f"Empty file: {path}")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns in {path}, got {data.shape[1]}")
    return data


def compute_translational_speed(time: np.ndarray, position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if time.size < 2:
        raise ValueError("Need at least two samples to compute speed")

    dt = np.diff(time)
    dp = np.diff(position, axis=0)

    dt_safe = np.where(dt == 0, np.nan, dt)
    velocity = dp / dt_safe[:, None]
    speed = np.linalg.norm(velocity, axis=1)

    # If any dt is zero, speed becomes nan; convert to 0 for robustness.
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    return time[1:], speed


def filter_by_time(time: np.ndarray, position: np.ndarray, min_time: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    mask = time > min_time
    return time[mask], position[mask]


def save_csv(out_csv: Path, time: np.ndarray, speed: np.ndarray) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "translational_speed"])
        writer.writerows(zip(time, speed))


def save_plot(out_png: Path, time: np.ndarray, speed: np.ndarray, title: str, show: bool) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time, speed, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("translational speed [position units/s]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Body translational speed magnitude from Body.dat files."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=None,
        help="(positional) Directory containing Body.dat",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir_opt",
        type=Path,
        default=Path("data/NABACOE01.4"),
        help="Directory containing Body.dat (fallback)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: result_csv/<folder>_body_translational_speed.csv)",
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=None,
        help="Output plot path (default: result_plot/<folder>_body_translational_speed.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot window",
    )

    args = parser.parse_args()

    data_dir: Path = args.data_dir if args.data_dir is not None else args.data_dir_opt
    body_path = data_dir / "Body.dat"
    if not body_path.exists():
        raise FileNotFoundError(f"Missing Body.dat: {body_path}")

    data = load_body_dat(body_path)
    time = data[:, 0]
    position = data[:, 1:4]

    time, position = filter_by_time(time, position, min_time=2.0)

    speed_time, speed = compute_translational_speed(time, position)

    base_name = data_dir.name
    out_csv = args.out_csv if args.out_csv is not None else Path(f"result_csv_translational_speed/{base_name}_body_translational_speed.csv")
    out_plot = args.out_plot if args.out_plot is not None else Path(f"result_plot_translational_speed/{base_name}_body_translational_speed.png")

    save_csv(out_csv, speed_time, speed)
    save_plot(out_plot, speed_time, speed, title=f"Body translational speed: {base_name}", show=args.show)

    print(f"Processed: {body_path}")
    print(f"Saved CSV : {out_csv}")
    print(f"Saved plot: {out_plot}")


if __name__ == "__main__":
    main()
