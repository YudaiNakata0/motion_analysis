#!/usr/bin/env python3
"""Analyze .dat motion files and plot joint angles over time.

Data format per row:
- 2 columns  : time, joint_angle
- 10 columns : time, 3x3 rotation matrix (9 elements)
- 13 columns : time, xyz position (3), 3x3 rotation matrix (9 elements)

Rotation matrix is interpreted as row-major:
[r11 r12 r13 r21 r22 r23 r31 r32 r33]

Euler conversion uses ZYX order (yaw around Z, pitch around Y, roll around X):
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class JointSeries:
    time: np.ndarray
    joint_angle_deg: Optional[np.ndarray] = None
    roll_deg: Optional[np.ndarray] = None
    pitch_deg: Optional[np.ndarray] = None
    yaw_deg: Optional[np.ndarray] = None


def load_dat_file(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, dtype=float)
    if data.size == 0:
        raise ValueError(f"Empty file: {path}")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if np.isnan(data).any():
        raise ValueError(f"NaN detected while reading: {path}")

    return data


def rotmat_to_rpy_zyx_deg(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to roll/pitch/yaw in degrees.

    Args:
        R: shape (N, 3, 3)

    Returns:
        shape (N, 3) with columns [roll, pitch, yaw] in degrees.
    """
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError("R must have shape (N, 3, 3)")

    r00 = R[:, 0, 0]
    r10 = R[:, 1, 0]
    r20 = R[:, 2, 0]
    r21 = R[:, 2, 1]
    r22 = R[:, 2, 2]
    r01 = R[:, 0, 1]
    r11 = R[:, 1, 1]
    r12 = R[:, 1, 2]

    pitch = np.arcsin(np.clip(-r20, -1.0, 1.0))
    cp = np.cos(pitch)

    singular = np.abs(cp) < 1e-8

    roll = np.empty_like(pitch)
    yaw = np.empty_like(pitch)

    non_singular = ~singular
    roll[non_singular] = np.arctan2(r21[non_singular], r22[non_singular])
    yaw[non_singular] = np.arctan2(r10[non_singular], r00[non_singular])

    # Gimbal-lock handling for |pitch| ~ 90 deg
    roll[singular] = np.arctan2(-r12[singular], r11[singular])
    yaw[singular] = 0.0

    rpy = np.stack([roll, pitch, yaw], axis=1)
    return np.degrees(rpy)


def analyze_file(path: Path, scalar_unit: str) -> JointSeries:
    data = load_dat_file(path)
    n_cols = data.shape[1]
    t = data[:, 0]

    if n_cols == 2:
        joint = data[:, 1]
        if scalar_unit == "rad":
            joint = np.degrees(joint)
        return JointSeries(time=t, joint_angle_deg=joint)

    if n_cols == 10:
        R = data[:, 1:10].reshape(-1, 3, 3)
        rpy = rotmat_to_rpy_zyx_deg(R)
        return JointSeries(time=t, roll_deg=rpy[:, 0], pitch_deg=rpy[:, 1], yaw_deg=rpy[:, 2])

    if n_cols == 13:
        R = data[:, 4:13].reshape(-1, 3, 3)
        rpy = rotmat_to_rpy_zyx_deg(R)
        return JointSeries(time=t, roll_deg=rpy[:, 0], pitch_deg=rpy[:, 1], yaw_deg=rpy[:, 2])

    raise ValueError(f"Unsupported column count ({n_cols}) in {path}")


def save_long_csv(out_csv: Path, all_series: Dict[str, JointSeries]) -> None:
    rows = ["joint,time,angle_type,angle_deg"]
    for joint, s in all_series.items():
        t = s.time
        if s.joint_angle_deg is not None:
            for ti, ai in zip(t, s.joint_angle_deg):
                rows.append(f"{joint},{ti:.10g},joint_angle,{ai:.10g}")

        if s.roll_deg is not None:
            for ti, ai in zip(t, s.roll_deg):
                rows.append(f"{joint},{ti:.10g},roll,{ai:.10g}")

        if s.pitch_deg is not None:
            for ti, ai in zip(t, s.pitch_deg):
                rows.append(f"{joint},{ti:.10g},pitch,{ai:.10g}")

        if s.yaw_deg is not None:
            for ti, ai in zip(t, s.yaw_deg):
                rows.append(f"{joint},{ti:.10g},yaw,{ai:.10g}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_all(all_series: Dict[str, JointSeries], out_png: Path, title: str, show: bool) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    ax_roll, ax_pitch, ax_yaw, ax_joint = axes

    for joint, s in all_series.items():
        if s.roll_deg is not None:
            ax_roll.plot(s.time, s.roll_deg, label=joint)
        if s.pitch_deg is not None:
            ax_pitch.plot(s.time, s.pitch_deg, label=joint)
        if s.yaw_deg is not None:
            ax_yaw.plot(s.time, s.yaw_deg, label=joint)
        if s.joint_angle_deg is not None:
            ax_joint.plot(s.time, s.joint_angle_deg, label=joint)

    ax_roll.set_ylabel("roll [deg]")
    ax_pitch.set_ylabel("pitch [deg]")
    ax_yaw.set_ylabel("yaw [deg]")
    ax_joint.set_ylabel("joint angle [deg]")
    ax_joint.set_xlabel("time [s]")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend(loc="upper right", ncol=2, fontsize=8)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze .dat motion files and plot per-joint angles over time."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        default=None,
        help="(positional) Directory containing .dat files",
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir_opt",
        type=Path,
        default=Path("data/NABACOE01.4"),
        help="Directory containing .dat files (fallback)",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.dat",
        help="File pattern inside data-dir",
    )
    parser.add_argument(
        "--out-plot",
        type=Path,
        default=None,
        help="Output plot path (default: result_plot_joint_angles/<data_dir_name>_joint_angles.png)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: result_csv_joint_angles/<data_dir_name>_joint_angles_long.csv)",
    )
    parser.add_argument(
        "--scalar-unit",
        choices=["rad", "deg"],
        default="rad",
        help="Unit for 2-column joint angle files",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot window",
    )

    args = parser.parse_args()

    # Prefer positional `data_dir` when provided; otherwise use `--data-dir` fallback.
    data_dir: Path = args.data_dir if args.data_dir is not None else args.data_dir_opt
    files = sorted(data_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files found: {data_dir / args.glob}")

    all_series: Dict[str, JointSeries] = {}
    for f in files:
        joint_name = f.stem
        all_series[joint_name] = analyze_file(f, scalar_unit=args.scalar_unit)

    # Build default output filenames from input folder name when not provided
    base_name = data_dir.name
    out_plot = args.out_plot if args.out_plot is not None else Path(f"result_plot_joint_angles/{base_name}_joint_angles.png")
    out_csv = args.out_csv if args.out_csv is not None else Path(f"result_csv_joint_angles/{base_name}_joint_angles_long.csv")

    save_long_csv(out_csv, all_series)
    plot_all(
        all_series,
        out_png=out_plot,
        title=f"Joint angles over time: {data_dir.name}",
        show=args.show,
    )

    print(f"Processed {len(files)} files from {data_dir}")
    print(f"Saved plot: {out_plot}")
    print(f"Saved CSV : {out_csv}")


if __name__ == "__main__":
    main()
