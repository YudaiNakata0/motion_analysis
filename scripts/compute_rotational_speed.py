#!/usr/bin/env python3
"""Compute rotational speed magnitudes from .dat motion files.

For 2-column files (time, angle): compute absolute angular velocity = |d(angle)/dt|.
For 10/13-column files (rotation matrices): compute relative rotation
R_rel = R_prev.T @ R_curr, angle = acos((trace(R_rel)-1)/2), speed = angle/dt.

Outputs a long CSV and a combined plot (degrees/sec shown). CSV contains both deg/s and rad/s.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_dat_file(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, dtype=float)
    if data.size == 0:
        raise ValueError(f"Empty file: {path}")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def compute_speed_for_file(path: Path, scalar_unit: str):
    data = load_dat_file(path)
    n_cols = data.shape[1]
    t = data[:, 0]
    if t.size < 2:
        return None  # not enough samples

    if n_cols == 2:
        # 1-DOF angle. Convert to degrees and compute absolute derivative.
        ang = data[:, 1]
        if scalar_unit == "rad":
            ang_deg = np.degrees(ang)
        else:
            ang_deg = ang
        dt = np.diff(t)
        dang = np.diff(ang_deg)
        # avoid divide-by-zero
        dt_safe = np.where(dt == 0, 1e-12, dt)
        speed_deg_s = np.abs(dang) / dt_safe
        speed_rad_s = np.radians(speed_deg_s)
        times = t[1:]
        return times, speed_deg_s, speed_rad_s

    if n_cols == 10:
        R = data[:, 1:10].reshape(-1, 3, 3)
    elif n_cols == 13:
        R = data[:, 4:13].reshape(-1, 3, 3)
    else:
        raise ValueError(f"Unsupported column count ({n_cols}) in {path}")

    times = t[1:]
    speed_rad_s = np.empty(times.shape)
    for i in range(1, R.shape[0]):
        R_prev = R[i - 1]
        R_cur = R[i]
        R_rel = R_prev.T @ R_cur
        tr = np.trace(R_rel)
        # numerical stability
        cos_phi = (tr - 1.0) / 2.0
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        phi = np.arccos(cos_phi)
        dt = t[i] - t[i - 1]
        if dt == 0:
            speed_rad_s[i - 1] = 0.0
        else:
            speed_rad_s[i - 1] = phi / dt

    speed_deg_s = np.degrees(speed_rad_s)
    return times, speed_deg_s, speed_rad_s


def save_long_csv(out_csv: Path, all_speeds: Dict[str, tuple]):
    rows = ["joint,time,speed_deg_per_s,speed_rad_per_s"]
    for joint, v in all_speeds.items():
        if v is None:
            continue
        times, speed_deg_s, speed_rad_s = v
        for ti, sd, sr in zip(times, speed_deg_s, speed_rad_s):
            rows.append(f"{joint},{ti:.10g},{sd:.10g},{sr:.10g}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_all(all_speeds: Dict[str, tuple], out_png: Path, title: str, show: bool):
    fig, ax = plt.subplots(figsize=(14, 6))
    for joint, v in all_speeds.items():
        if v is None:
            continue
        times, speed_deg_s, _ = v
        ax.plot(times, speed_deg_s, label=joint)

    ax.set_ylabel("angular speed [deg/s]")
    ax.set_xlabel("time [s]")
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
    parser = argparse.ArgumentParser(description="Compute rotational speed magnitudes for .dat motion files")
    parser.add_argument("data_dir", nargs="?", type=Path, default=None, help="(positional) Directory containing .dat files")
    parser.add_argument("--data-dir", dest="data_dir_opt", type=Path, default=Path("data/NABACOE01.4"), help="Directory containing .dat files (fallback)")
    parser.add_argument("--glob", type=str, default="*.dat", help="File pattern inside data-dir")
    parser.add_argument("--out-plot", type=Path, default=None, help="Output plot path (default: result_plot/<data_dir_name>_rotational_speeds.png)")
    parser.add_argument("--out-csv", type=Path, default=None, help="Output CSV path (default: result_csv/<data_dir_name>_rotational_speeds_long.csv)")
    parser.add_argument("--scalar-unit", choices=["rad", "deg"], default="rad", help="Unit for 2-column joint angle files")
    parser.add_argument("--show", action="store_true", help="Show interactive plot window")
    args = parser.parse_args()

    data_dir: Path = args.data_dir if args.data_dir is not None else args.data_dir_opt
    files = sorted(data_dir.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files found: {data_dir / args.glob}")

    all_speeds: Dict[str, Optional[tuple]] = {}
    for f in files:
        joint_name = f.stem
        try:
            all_speeds[joint_name] = compute_speed_for_file(f, scalar_unit=args.scalar_unit)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            all_speeds[joint_name] = None

    base_name = data_dir.name
    out_plot = args.out_plot if args.out_plot is not None else Path(f"result_plot/{base_name}_rotational_speeds.png")
    out_csv = args.out_csv if args.out_csv is not None else Path(f"result_csv/{base_name}_rotational_speeds_long.csv")

    save_long_csv(out_csv, all_speeds)
    plot_all(all_speeds, out_plot, title=f"Rotational speeds: {data_dir.name}", show=args.show)

    print(f"Processed {len(files)} files from {data_dir}")
    print(f"Saved plot: {out_plot}")
    print(f"Saved CSV : {out_csv}")


if __name__ == "__main__":
    main()
