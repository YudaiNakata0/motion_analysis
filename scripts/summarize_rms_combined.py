#!/usr/bin/env python3
"""Summarize per-joint RMS values from both rotational and translational speed CSV files.

Reads from:
- result_csv/*_rotational_speeds_long.csv
- result_csv_translational_speed/*_body_translational_speed.csv

For rotational speeds: per-joint angular speed RMS.
For translational speeds: Body translational speed RMS.

Output format:
    data_name,Body_rotational,Head_rotational,...,Body_translational
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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


def read_rotational_speeds(csv_path: Path, value_column: str, min_time: float = 2.0) -> Dict[str, List[float]]:
    joint_values: Dict[str, List[float]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        if "joint" not in reader.fieldnames or "time" not in reader.fieldnames or value_column not in reader.fieldnames:
            return {}

        for row in reader:
            try:
                time_value = float(row.get("time", ""))
            except ValueError:
                continue
            if time_value <= min_time:
                continue
            joint = (row.get("joint") or "").strip()
            if not joint:
                continue
            try:
                joint_values[joint].append(float(row.get(value_column, "")))
            except ValueError:
                continue
    return joint_values


def read_translational_speed(csv_path: Path, min_time: float = 2.0) -> Optional[float]:
    """Read translational speed CSV and return RMS of the speed column."""
    speeds: List[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        if "time" not in reader.fieldnames or "translational_speed" not in reader.fieldnames:
            return None

        for row in reader:
            try:
                time_value = float(row.get("time", ""))
            except ValueError:
                continue
            if time_value <= min_time:
                continue
            try:
                speeds.append(float(row.get("translational_speed", "")))
            except ValueError:
                continue

    return rms(speeds) if speeds else None


def build_joint_order(all_joints: Iterable[str]) -> List[str]:
    joint_set = set(all_joints)
    ordered = [joint for joint in PREFERRED_JOINT_ORDER if joint in joint_set]
    extras = sorted(joint_set - set(ordered))
    return ordered + extras


def summarize_combined(
    rot_dir: Path,
    trans_dir: Path,
    rot_value_column: str = "speed_deg_per_s",
    min_time: float = 2.0,
) -> Tuple[List[str], List[List[str]]]:
    rot_files = sorted(rot_dir.glob("*_rotational_speeds_long.csv"))
    if not rot_files:
        raise FileNotFoundError(f"No rotational CSV files found in {rot_dir}")

    all_rms: Dict[str, Dict[str, Optional[float]]] = {}
    all_joints: set[str] = set()

    for rot_path in rot_files:
        data_name = rot_path.stem.replace("_rotational_speeds_long", "")
        rot_speeds = read_rotational_speeds(rot_path, value_column=rot_value_column, min_time=min_time)
        if not rot_speeds:
            continue

        all_rms[data_name] = {}
        for joint, speeds in rot_speeds.items():
            all_rms[data_name][f"{joint}_rotational"] = rms(speeds)
            all_joints.add(joint)

        trans_path = trans_dir / f"{data_name}_body_translational_speed.csv"
        if trans_path.exists():
            trans_rms = read_translational_speed(trans_path, min_time=min_time)
            if trans_rms is not None:
                all_rms[data_name]["Body_translational"] = trans_rms

    joint_order = build_joint_order(all_joints)
    header = ["data_name"]
    for joint in joint_order:
        header.append(f"{joint}_rotational")
    header.append("Body_translational")

    rows: List[List[str]] = []
    for data_name in sorted(all_rms.keys()):
        row = [data_name]
        rms_map = all_rms[data_name]
        for joint in joint_order:
            value = rms_map.get(f"{joint}_rotational")
            row.append("") if value is None else row.append(f"{value:.10g}")
        value = rms_map.get("Body_translational")
        row.append("") if value is None else row.append(f"{value:.10g}")
        rows.append(row)

    return header, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute combined RMS summary (rotational + translational) per joint."
    )
    parser.add_argument(
        "--rot-dir",
        type=Path,
        default=Path("result_csv"),
        help="Directory with rotational speed CSVs (default: result_csv)",
    )
    parser.add_argument(
        "--trans-dir",
        type=Path,
        default=Path("result_csv_translational_speed"),
        help="Directory with translational speed CSVs (default: result_csv_translational_speed)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("result_csv/rms_summary_combined.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--rot-value-column",
        choices=["speed_deg_per_s", "speed_rad_per_s"],
        default="speed_rad_per_s",
        help="Which rotational speed column to use for RMS calculation",
    )

    args = parser.parse_args()

    header, rows = summarize_combined(
        rot_dir=args.rot_dir,
        trans_dir=args.trans_dir,
        rot_value_column=args.rot_value_column,
        min_time=2.0,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Processed {len(rows)} data samples")
    print(f"Saved combined summary: {args.out}")


if __name__ == "__main__":
    main()
