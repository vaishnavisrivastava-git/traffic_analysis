
"""Prepare and validate the NGSIM vehicle trajectory dataset in chunks."""

import argparse
import csv
import heapq
import math
import os
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "export.csv"
ROOT_LEVEL_INPUT = PROJECT_ROOT / "export.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "ngsim_cleaned.csv"
STATISTICS_FILE = PROJECT_ROOT / "results" / "dataset_statistics.txt"
CHUNK_SIZE = 100_000

REQUIRED_COLUMNS = [
    "Vehicle_ID",
    "Frame_ID",
    "Total_Frames",
    "Global_Time",
    "Local_X",
    "Local_Y",
    "Global_X",
    "Global_Y",
    "v_length",
    "v_Width",
    "v_Class",
    "v_Vel",
    "v_Acc",
    "Lane_ID",
    "Preceding",
    "Following",
    "Space_Headway",
    "Time_Headway",
    "Location",
]

NUMERIC_COLUMNS = [
    column for column in REQUIRED_COLUMNS if column != "Location"
]

ESSENTIAL_COLUMNS = [
    "Vehicle_ID",
    "Frame_ID",
    "Global_Time",
    "Local_X",
    "Local_Y",
    "v_Vel",
    "v_Acc",
    "Lane_ID",
]

POSITIVE_COLUMNS = [
    "Vehicle_ID",
    "Frame_ID",
    "v_length",
    "v_Width",
    "Lane_ID",
]


def resolve_input_path(input_argument):
    """Use the requested input path, or the root-level file present in this workspace."""
    if input_argument:
        input_path = Path(input_argument)

        if not input_path.is_absolute():
            input_path = PROJECT_ROOT / input_path

        return input_path

    if DEFAULT_INPUT.exists():
        return DEFAULT_INPUT

    if ROOT_LEVEL_INPUT.exists():
        return ROOT_LEVEL_INPUT

    return DEFAULT_INPUT


def load_and_validate_header(input_file):
    """Read only the header and verify all columns required by this project."""
    if not input_file.is_file():
        raise FileNotFoundError(
            f"Input CSV does not exist: {input_file}"
        )

    try:
        header = pd.read_csv(
            input_file,
            nrows=0,
            low_memory=False
        ).columns.tolist()

    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise ValueError(
            f"Could not read the CSV header from {input_file}: {error}"
        ) from error

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in header
    ]

    if missing_columns:
        raise ValueError(
            "The input CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    return header


def clean_numeric_column(series):
    """
    Clean comma-formatted numeric values before numeric conversion.

    Example:
        '1,040.94' -> '1040.94' -> 1040.94

    Empty strings and genuinely invalid values become NaN.
    """
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
    )

    return pd.to_numeric(cleaned, errors="coerce")


def clean_chunk(chunk):
    """Convert, validate, and sort one in-memory chunk."""
    chunk = chunk.loc[:, REQUIRED_COLUMNS].copy()

    missing_values_before = int(
        chunk.isna().sum().sum()
    )

    # Convert every numeric column while handling thousands separators.
    for column in NUMERIC_COLUMNS:
        chunk[column] = clean_numeric_column(chunk[column])

    # Remove rows missing any of the 8 essential columns.
    missing_rows = chunk[ESSENTIAL_COLUMNS].isna().any(axis=1)

    missing_rows_removed = int(
        missing_rows.sum()
    )

    chunk = chunk.loc[~missing_rows].copy()

    # Physical validity checks.
    invalid_rows = chunk[POSITIVE_COLUMNS].le(0).any(axis=1)

    invalid_rows_removed = int(
        invalid_rows.sum()
    )

    chunk = chunk.loc[~invalid_rows].copy()

    # Sort by every selected column.
    # Vehicle_ID and Frame_ID remain the primary trajectory ordering
    # because they appear first in REQUIRED_COLUMNS.
    chunk = chunk.sort_values(
        by=REQUIRED_COLUMNS,
        kind="mergesort",
        na_position="last",
    )

    return (
        chunk,
        missing_values_before,
        missing_rows_removed + invalid_rows_removed,
    )


def row_sort_key(row):
    """Return a cross-chunk merge key matching the sorted CSV columns."""
    key = []

    for column, value in zip(REQUIRED_COLUMNS, row):
        if value == "":
            key.append((1, ""))

        elif column == "Location":
            key.append((0, value))

        else:
            key.append((0, float(value)))

    return tuple(key)


def empty_numeric_statistic():
    return {
        "count": 0,
        "sum": 0.0,
        "min": None,
        "max": None,
    }


def update_numeric_statistic(statistic, value):
    if math.isnan(value):
        return

    statistic["count"] += 1
    statistic["sum"] += value

    statistic["min"] = (
        value
        if statistic["min"] is None
        else min(statistic["min"], value)
    )

    statistic["max"] = (
        value
        if statistic["max"] is None
        else max(statistic["max"], value)
    )


def format_statistic(statistic):
    if statistic["count"] == 0:
        return "N/A"

    average = statistic["sum"] / statistic["count"]

    return (
        f"min={statistic['min']:.6f}, "
        f"max={statistic['max']:.6f}, "
        f"average={average:.6f}"
    )


def save_statistics(statistics, statistics_file):
    """Write the requested counts and numeric summaries to a text file."""
    lines = [
        "NGSIM Dataset Preprocessing Statistics",
        "=======================================",
        f"Input file: {statistics['input_file']}",
        f"Original number of rows processed: {statistics['original_rows']}",
        f"Final number of cleaned rows: {statistics['final_rows']}",
        f"Duplicate rows removed: {statistics['duplicate_rows_removed']}",
        f"Invalid/missing rows removed: {statistics['invalid_missing_rows_removed']}",
        f"Unique vehicles: {len(statistics['vehicles'])}",
        f"Unique frames: {len(statistics['frames'])}",
        f"Unique lanes: {len(statistics['lanes'])}",
        f"Velocity: {format_statistic(statistics['velocity'])}",
        f"Acceleration: {format_statistic(statistics['acceleration'])}",
        f"Space headway: {format_statistic(statistics['space_headway'])}",
        f"Missing values before cleaning: {statistics['missing_values_before']}",
        f"Missing values after cleaning: {statistics['missing_values_after']}",
        "Location values: "
        + (
            ", ".join(sorted(statistics["locations"]))
            or "None"
        ),
    ]

    statistics_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    statistics_file.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    return "\n".join(lines)


def merge_sorted_chunks(
    chunk_files,
    output_file,
    statistics
):
    """Merge sorted on-disk chunks, remove global duplicates, and collect stats."""
    column_indexes = {
        column: index
        for index, column in enumerate(REQUIRED_COLUMNS)
    }

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_output = output_file.with_name(
        output_file.name + ".tmp"
    )

    try:
        with ExitStack() as stack:
            readers = []

            for chunk_file in chunk_files:
                handle = stack.enter_context(
                    chunk_file.open(
                        "r",
                        newline="",
                        encoding="utf-8"
                    )
                )

                reader = csv.reader(handle)

                next(reader, None)

                readers.append(reader)

            with temporary_output.open(
                "w",
                newline="",
                encoding="utf-8"
            ) as output_handle:

                writer = csv.writer(output_handle)

                writer.writerow(REQUIRED_COLUMNS)

                merged_rows = heapq.merge(
                    *readers,
                    key=row_sort_key
                )

                previous_key = None

                for row in merged_rows:
                    identity = row_sort_key(row)

                    if identity == previous_key:
                        statistics["duplicate_rows_removed"] += 1
                        continue

                    previous_key = identity

                    writer.writerow(row)

                    statistics["final_rows"] += 1

                    statistics["missing_values_after"] += sum(
                        value == ""
                        for value in row
                    )

                    vehicle_id = float(
                        row[column_indexes["Vehicle_ID"]]
                    )

                    frame_id = float(
                        row[column_indexes["Frame_ID"]]
                    )

                    lane_id = float(
                        row[column_indexes["Lane_ID"]]
                    )

                    statistics["vehicles"].add(vehicle_id)
                    statistics["frames"].add(frame_id)
                    statistics["lanes"].add(lane_id)

                    for column, statistic_name in (
                        ("v_Vel", "velocity"),
                        ("v_Acc", "acceleration"),
                        ("Space_Headway", "space_headway"),
                    ):
                        value = row[
                            column_indexes[column]
                        ]

                        if value != "":
                            update_numeric_statistic(
                                statistics[statistic_name],
                                float(value)
                            )

                    location = row[
                        column_indexes["Location"]
                    ]

                    if location:
                        statistics["locations"].add(location)

        os.replace(
            temporary_output,
            output_file
        )

    except Exception:
        temporary_output.unlink(
            missing_ok=True
        )
        raise


def preprocess(
    input_file,
    output_file=OUTPUT_FILE,
    statistics_file=STATISTICS_FILE
):
    """Process the CSV a chunk at a time and return the calculated statistics."""
    load_and_validate_header(input_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    statistics = {
        "input_file": str(input_file),
        "original_rows": 0,
        "final_rows": 0,
        "duplicate_rows_removed": 0,
        "invalid_missing_rows_removed": 0,
        "missing_values_before": 0,
        "missing_values_after": 0,
        "vehicles": set(),
        "frames": set(),
        "lanes": set(),
        "locations": set(),
        "velocity": empty_numeric_statistic(),
        "acceleration": empty_numeric_statistic(),
        "space_headway": empty_numeric_statistic(),
    }

    try:
        with tempfile.TemporaryDirectory(
            prefix="ngsim_chunks_"
        ) as temporary_directory:

            temporary_path = Path(
                temporary_directory
            )

            chunk_files = []

            chunks = pd.read_csv(
                input_file,
                usecols=REQUIRED_COLUMNS,
                chunksize=CHUNK_SIZE,
                low_memory=False,
            )

            for chunk_number, chunk in enumerate(chunks):
                statistics["original_rows"] += len(chunk)

                cleaned, missing_before, rows_removed = clean_chunk(
                    chunk
                )

                statistics["missing_values_before"] += (
                    missing_before
                )

                statistics["invalid_missing_rows_removed"] += (
                    rows_removed
                )

                if not cleaned.empty:
                    chunk_file = (
                        temporary_path
                        / f"sorted_{chunk_number:06d}.csv"
                    )

                    cleaned.to_csv(
                        chunk_file,
                        index=False,
                        na_rep=""
                    )

                    chunk_files.append(chunk_file)

            merge_sorted_chunks(
                chunk_files,
                output_file,
                statistics
            )

    except (
        OSError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        ValueError
    ) as error:
        raise RuntimeError(
            f"The dataset could not be read or processed: {error}"
        ) from error

    statistics_text = save_statistics(
        statistics,
        statistics_file
    )

    print(statistics_text)
    print(
        f"Cleaned CSV saved to: {output_file}"
    )
    print(
        f"Statistics saved to: {statistics_file}"
    )

    return statistics


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input",
        type=Path,
        help=(
            "Input CSV path "
            "(default: data/raw/export.csv, "
            "then workspace-root export.csv)"
        ),
    )

    arguments = parser.parse_args()

    input_file = resolve_input_path(
        arguments.input
    )

    try:
        preprocess(input_file)

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError
    ) as error:

        print(
            f"Error: {error}",
            file=sys.stderr
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

