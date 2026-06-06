#!/usr/bin/env python3
"""Summarize per-joint RMS values from CSV files under result_csv/.

Expected input format (long form):
    joint,time,speed_deg_per_s,speed_rad_per_s

For each CSV file, this script computes the root mean square (RMS) of the
chosen speed column for each joint and writes one summary CSV with a header.

Output format:
    data_name,Body,Head,LeftArm1,...

The data_name is derived from the input CSV filename stem.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


PREFERRED_JOINT_ORDER = [
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
]


def rms(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def read_joint_speeds(csv_path: Path, value_column: str, min_time: float = 2.0) -> Dict[str, List[float]]:
    joint_values: Dict[str, List[float]] = defaultdict(list)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {csv_path}")
        if "joint" not in reader.fieldnames:
            raise ValueError(f"Missing 'joint' column in {csv_path}")
        if "time" not in reader.fieldnames:
            raise ValueError(f"Missing 'time' column in {csv_path}")
        if value_column not in reader.fieldnames:
            raise ValueError(
                f"Missing '{value_column}' column in {csv_path}. Available: {reader.fieldnames}"
            )

        for row in reader:
            raw_time = (row.get("time") or "").strip()
            if not raw_time:
                continue
            try:
                time_value = float(raw_time)
            except ValueError:
                continue
            if time_value <= min_time:
                continue

            joint = (row.get("joint") or "").strip()
            if not joint:
                continue
            raw_value = (row.get(value_column) or "").strip()
            if not raw_value:
                continue
            try:
                joint_values[joint].append(float(raw_value))
            except ValueError:
                continue

    return joint_values


def build_joint_order(all_joints: Iterable[str]) -> List[str]:
    joint_set = set(all_joints)
    ordered = [joint for joint in PREFERRED_JOINT_ORDER if joint in joint_set]
    extras = sorted(joint_set - set(ordered))
    return ordered + extras


def summarize_directory(
    input_dir: Path,
    pattern: str,
    value_column: str,
    min_time: float = 2.0,
) -> tuple[list[str], list[list[str]]]:
    csv_files = sorted(input_dir.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir} matching {pattern}")

    per_file_rms: Dict[str, Dict[str, float]] = {}
    all_joints: set[str] = set()

    for csv_path in csv_files:
        joint_values = read_joint_speeds(csv_path, value_column=value_column, min_time=min_time)
        if not joint_values:
            continue

        data_name = csv_path.stem.replace("_rotational_speeds_long", "")
        per_file_rms[data_name] = {joint: rms(values) for joint, values in joint_values.items()}
        all_joints.update(joint_values.keys())

    joint_order = build_joint_order(all_joints)

    header = ["data_name", *joint_order]
    rows: list[list[str]] = []
    for data_name in sorted(per_file_rms.keys()):
        row = [data_name]
        joint_map = per_file_rms[data_name]
        for joint in joint_order:
            value = joint_map.get(joint)
            row.append("") if value is None else row.append(f"{value:.10g}")
        rows.append(row)

    return header, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute RMS summary per joint from CSV files under result_csv/."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("result_csv"),
        help="Directory containing CSV files (default: result_csv)",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*_rotational_speeds_long.csv",
        help="File pattern inside input_dir",
    )
    parser.add_argument(
        "--value-column",
        choices=["speed_deg_per_s", "speed_rad_per_s"],
        default="speed_deg_per_s",
        help="Which speed column to use for RMS calculation",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: result_csv/rms_summary.csv)",
    )

    args = parser.parse_args()

    input_dir: Path = args.input_dir
    out_csv = args.out if args.out is not None else input_dir / "rms_summary.csv"

    header, rows = summarize_directory(
        input_dir=input_dir,
        pattern=args.glob,
        value_column=args.value_column,
        min_time=2.0,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Processed {len(rows)} files from {input_dir}")
    print(f"Saved summary: {out_csv}")


if __name__ == "__main__":
    main()
