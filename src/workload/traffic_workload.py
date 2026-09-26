"""Python reference workload for same-frame NGSIM vehicle interactions."""

import argparse
import csv
import heapq
import itertools
import math
import os
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "ngsim_cleaned.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "vehicle_interactions.csv"
STATISTICS_FILE = PROJECT_ROOT / "results" / "traffic_workload_statistics.txt"
CHUNK_SIZE = 100_000
INTERACTION_RADIUS = 50.0
TTC_THRESHOLD = 3.0

INPUT_COLUMNS = [
    "Vehicle_ID",
    "Frame_ID",
    "Global_Time",
    "Lane_ID",
    "Local_X",
    "Local_Y",
    "v_Vel",
    "v_Acc",
]
OUTPUT_COLUMNS = [
    "Frame_ID",
    "Global_Time",
    "Vehicle_ID_1",
    "Vehicle_ID_2",
    "Lane_ID_1",
    "Lane_ID_2",
    "Distance",
    "Relative_Velocity",
    "TTC",
    "Critical_Interaction",
]


def distance(x1, y1, x2, y2):
    """Return the Euclidean distance between two vehicle positions."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def relative_velocity(velocity_1, velocity_2):
    """Return velocity_1 minus velocity_2, as specified for this workload."""
    return velocity_1 - velocity_2


def simplified_ttc(interaction_distance, relative_speed):
    """Calculate simplified TTC, or infinity when the pair is not closing."""
    if relative_speed < 0:
        return interaction_distance / abs(relative_speed)
    return math.inf


def same_frame(frame_1, frame_2):
    return frame_1 == frame_2


def same_or_adjacent_lane(lane_1, lane_2):
    return abs(lane_1 - lane_2) <= 1


def within_interaction_radius(interaction_distance, radius=INTERACTION_RADIUS):
    return interaction_distance <= radius


def is_critical_interaction(ttc, threshold=TTC_THRESHOLD):
    return math.isfinite(ttc) and ttc <= threshold


def analyze_frame(records):
    """Analyze one frame's records and return candidate count and interactions."""
    ordered_records = sorted(records, key=lambda record: record["Vehicle_ID"])
    candidate_pairs = 0
    interactions = []
    seen_pairs = set()

    for first, second in itertools.combinations(ordered_records, 2):
        if not same_frame(first["Frame_ID"], second["Frame_ID"]):
            continue

        vehicle_pair = (first["Vehicle_ID"], second["Vehicle_ID"])
        if vehicle_pair[0] == vehicle_pair[1] or vehicle_pair in seen_pairs:
            continue
        seen_pairs.add(vehicle_pair)
        candidate_pairs += 1

        if not same_or_adjacent_lane(first["Lane_ID"], second["Lane_ID"]):
            continue

        separation = distance(
            first["Local_X"],
            first["Local_Y"],
            second["Local_X"],
            second["Local_Y"],
        )
        if not within_interaction_radius(separation):
            continue

        relative_speed = relative_velocity(first["v_Vel"], second["v_Vel"])
        ttc = simplified_ttc(separation, relative_speed)
        interactions.append(
            {
                "Frame_ID": first["Frame_ID"],
                "Global_Time": first["Global_Time"],
                "Vehicle_ID_1": vehicle_pair[0],
                "Vehicle_ID_2": vehicle_pair[1],
                "Lane_ID_1": first["Lane_ID"],
                "Lane_ID_2": second["Lane_ID"],
                "Distance": separation,
                "Relative_Velocity": relative_speed,
                "TTC": ttc,
                "Critical_Interaction": is_critical_interaction(ttc),
            }
        )

    return candidate_pairs, interactions


def row_sort_key(row):
    """Sort temporary CSV rows by frame and vehicle, independent of text format."""
    return (float(row[1]), float(row[0]))


def update_summary(summary, chunk):
    summary["vehicles"].update(chunk["Vehicle_ID"].astype(int).unique())
    summary["lanes"].update(chunk["Lane_ID"].astype(int).unique())
    summary["velocity_sum"] += float(chunk["v_Vel"].sum())
    summary["velocity_count"] += int(chunk["v_Vel"].count())
    summary["acceleration_sum"] += float(chunk["v_Acc"].sum())
    summary["acceleration_count"] += int(chunk["v_Acc"].count())


def update_interaction_summary(summary, interactions):
    for interaction in interactions:
        interaction_distance = interaction["Distance"]
        summary["nearby_interactions"] += 1
        summary["interaction_distance_sum"] += interaction_distance
        summary["minimum_interaction_distance"] = min(
            summary["minimum_interaction_distance"], interaction_distance
        )
        ttc = interaction["TTC"]
        if math.isfinite(ttc):
            summary["valid_ttc_count"] += 1
            summary["valid_ttc_sum"] += ttc
            summary["minimum_valid_ttc"] = min(summary["minimum_valid_ttc"], ttc)
        if interaction["Critical_Interaction"]:
            summary["critical_interactions"] += 1


def average_or_na(total, count):
    return f"{total / count:.6f}" if count else "N/A"


def format_minimum(value):
    return f"{value:.6f}" if math.isfinite(value) else "N/A"


def save_statistics(summary, statistics_file):
    lines = [
        "Traffic Workload Statistics",
        "============================",
        f"Records processed: {summary['records_processed']}",
        f"Unique vehicles: {len(summary['vehicles'])}",
        f"Frames: {summary['frames']}",
        f"Lanes: {len(summary['lanes'])}",
        f"Candidate pairs: {summary['candidate_pairs']}",
        f"Nearby interactions: {summary['nearby_interactions']}",
        f"Critical interactions: {summary['critical_interactions']}",
        "Average interaction distance: "
        + average_or_na(
            summary["interaction_distance_sum"], summary["nearby_interactions"]
        ),
        "Minimum interaction distance: "
        + format_minimum(summary["minimum_interaction_distance"]),
        "Average valid TTC: "
        + average_or_na(summary["valid_ttc_sum"], summary["valid_ttc_count"]),
        "Minimum valid TTC: " + format_minimum(summary["minimum_valid_ttc"]),
        "Average velocity: "
        + average_or_na(summary["velocity_sum"], summary["velocity_count"]),
        "Average acceleration: "
        + average_or_na(summary["acceleration_sum"], summary["acceleration_count"]),
    ]
    statistics_file.parent.mkdir(parents=True, exist_ok=True)
    statistics_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def process_dataset(input_file=INPUT_FILE, output_file=OUTPUT_FILE, statistics_file=STATISTICS_FILE):
    """Process the cleaned trajectory CSV in chunks and emit sorted interactions."""
    input_file = Path(input_file)
    output_file = Path(output_file)
    statistics_file = Path(statistics_file)
    if not input_file.is_file():
        raise FileNotFoundError(f"Cleaned NGSIM CSV not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "records_processed": 0,
        "vehicles": set(),
        "lanes": set(),
        "frames": 0,
        "candidate_pairs": 0,
        "nearby_interactions": 0,
        "critical_interactions": 0,
        "interaction_distance_sum": 0.0,
        "minimum_interaction_distance": math.inf,
        "valid_ttc_sum": 0.0,
        "valid_ttc_count": 0,
        "minimum_valid_ttc": math.inf,
        "velocity_sum": 0.0,
        "velocity_count": 0,
        "acceleration_sum": 0.0,
        "acceleration_count": 0,
    }

    temporary_output = None
    try:
        with tempfile.TemporaryDirectory(prefix="traffic_sorted_") as temporary_directory:
            sorted_files = []
            for chunk_number, chunk in enumerate(
                pd.read_csv(
                    input_file,
                    usecols=INPUT_COLUMNS,
                    chunksize=CHUNK_SIZE,
                    low_memory=False,
                )
            ):
                summary["records_processed"] += len(chunk)
                for column in INPUT_COLUMNS:
                    chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
                chunk = chunk.dropna(subset=INPUT_COLUMNS)
                chunk = chunk.sort_values(
                    ["Frame_ID", "Vehicle_ID"], kind="mergesort"
                )
                update_summary(summary, chunk)

                if not chunk.empty:
                    sorted_file = Path(temporary_directory) / f"sorted_{chunk_number:06d}.csv"
                    chunk.to_csv(sorted_file, columns=INPUT_COLUMNS, index=False)
                    sorted_files.append(sorted_file)

            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                suffix=".tmp",
                dir=output_file.parent,
                delete=False,
            ) as output_handle:
                temporary_output = Path(output_handle.name)
                writer = csv.writer(output_handle)
                writer.writerow(OUTPUT_COLUMNS)

                with ExitStack() as stack:
                    readers = []
                    for sorted_file in sorted_files:
                        handle = stack.enter_context(
                            sorted_file.open("r", newline="", encoding="utf-8")
                        )
                        reader = csv.reader(handle)
                        next(reader, None)
                        readers.append(reader)

                    merged_rows = heapq.merge(*readers, key=row_sort_key)
                    frame_index = INPUT_COLUMNS.index("Frame_ID")
                    for _, frame_iterator in itertools.groupby(
                        merged_rows,
                        key=lambda row: float(row[frame_index]),
                    ):
                        summary["frames"] += 1
                        frame_records = [
                            {
                                column: (
                                    int(float(value))
                                    if column in ("Vehicle_ID", "Frame_ID", "Lane_ID")
                                    else float(value)
                                )
                                for column, value in zip(INPUT_COLUMNS, row)
                            }
                            for row in frame_iterator
                        ]
                        candidate_pairs, interactions = analyze_frame(frame_records)
                        summary["candidate_pairs"] += candidate_pairs
                        update_interaction_summary(summary, interactions)
                        for interaction in interactions:
                            writer.writerow(
                                interaction[column] for column in OUTPUT_COLUMNS
                            )

        os.replace(temporary_output, output_file)
    except (OSError, pd.errors.ParserError, ValueError, KeyError) as error:
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        raise RuntimeError(f"Could not process cleaned NGSIM data: {error}") from error

    statistics_text = save_statistics(summary, statistics_file)
    print(statistics_text)
    print(f"Interaction CSV saved to: {output_file}")
    print(f"Statistics saved to: {statistics_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--statistics", type=Path, default=STATISTICS_FILE)
    arguments = parser.parse_args()
    try:
        process_dataset(arguments.input, arguments.output, arguments.statistics)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())