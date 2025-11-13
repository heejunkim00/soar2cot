"""
SOAR Dataset Loading and Processing

This module handles loading the SOAR dataset and converting it to Challenge objects
for use with the instruction generation pipeline.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from src.log import log
from src.models import Challenge, Example, Input, GRID


# SOAR dataset paths
SOAR_DATA_PATH = Path("/data/hjkim/soar2cot/data/soar_arc_train_5M.parquet")
SOAR_LABELED_PATH = Path("/data/hjkim/soar2cot/data/soar_arc_train_5M_labeled.parquet")


def load_soar_data(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load SOAR dataset from parquet file.

    Args:
        path: Optional path to parquet file. Defaults to SOAR_DATA_PATH.

    Returns:
        DataFrame with columns:
        - code: Python code solution
        - correct_train_input: List of bool for training correctness
        - predicted_train_output: List of predicted training outputs
        - correct_test_input: List of bool for test correctness
        - predicted_test_output: List of predicted test outputs
        - task_id: ARC task ID
        - model: Source model name
        - generation: Generation number (0-6)
    """
    if path is None:
        path = SOAR_DATA_PATH

    log.info("Loading SOAR dataset", path=str(path))

    table = pq.read_table(path)
    df = table.to_pandas()

    log.info(
        "SOAR dataset loaded",
        rows=len(df),
        unique_tasks=df["task_id"].nunique(),
        columns=list(df.columns),
    )

    return df


def filter_original_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter SOAR data to only include "original" samples.

    Original samples are those where ALL test inputs were predicted correctly:
    all(correct_test_input) == True

    Args:
        df: SOAR DataFrame

    Returns:
        Filtered DataFrame containing only original samples
    """
    original_mask = df["correct_test_input"].apply(lambda x: all(x))
    original_df = df[original_mask].copy()

    log.info(
        "Filtered original data",
        original_count=len(original_df),
        total_count=len(df),
        percentage=f"{len(original_df)/len(df)*100:.2f}%",
    )

    return original_df


def filter_hindsight_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter SOAR data to only include "hindsight" samples.

    Hindsight samples are those where at least one test input was predicted incorrectly:
    any(correct_test_input) == False

    Args:
        df: SOAR DataFrame

    Returns:
        Filtered DataFrame containing only hindsight samples
    """
    hindsight_mask = df["correct_test_input"].apply(lambda x: not all(x))
    hindsight_df = df[hindsight_mask].copy()

    log.info(
        "Filtered hindsight data",
        hindsight_count=len(hindsight_df),
        total_count=len(df),
        percentage=f"{len(hindsight_df)/len(df)*100:.2f}%",
    )

    return hindsight_df


def create_challenge_from_soar(
    task_id: str,
    predicted_train_outputs: list[GRID],
    predicted_test_outputs: list[GRID],
    arc_data_dir: Path = Path("/data/hjkim/soar2cot/data/arc-prize-2024"),
) -> Challenge:
    """
    Create a Challenge object from SOAR data.

    This function:
    1. Loads the original ARC task data to get input grids
    2. Uses SOAR predicted outputs as the "ground truth" outputs
    3. Creates a Challenge object with this hybrid data

    For hindsight data, this means treating incorrect predictions as if they were
    the correct answers for a different task.

    Args:
        task_id: ARC task ID (e.g., "007bbfb7")
        predicted_train_outputs: SOAR predicted training outputs to use as ground truth
        predicted_test_outputs: SOAR predicted test outputs to use as ground truth
        arc_data_dir: Directory containing original ARC dataset

    Returns:
        Challenge object with original inputs and SOAR predicted outputs
    """
    # Load original ARC task to get input grids
    original_challenge = Challenge.load(task_id=task_id, data_dir=arc_data_dir)

    # Validate that lengths match
    if len(original_challenge.train) != len(predicted_train_outputs):
        raise ValueError(
            f"Training size mismatch for task {task_id}: "
            f"ARC has {len(original_challenge.train)} examples, "
            f"SOAR has {len(predicted_train_outputs)} predictions"
        )

    if len(original_challenge.test) != len(predicted_test_outputs):
        raise ValueError(
            f"Test size mismatch for task {task_id}: "
            f"ARC has {len(original_challenge.test)} examples, "
            f"SOAR has {len(predicted_test_outputs)} predictions"
        )

    # Create training examples with original inputs and SOAR predicted outputs
    train_examples = [
        Example(
            input=original_challenge.train[i].input,
            output=predicted_train_outputs[i],
        )
        for i in range(len(original_challenge.train))
    ]

    # Create test inputs (outputs are stored separately in SOAR data)
    test_inputs = [
        Input(input=original_challenge.test[i].input)
        for i in range(len(original_challenge.test))
    ]

    # Create Challenge object
    challenge = Challenge(
        task_id=task_id,
        train=train_examples,
        test=test_inputs,
    )

    log.debug(
        "Created challenge from SOAR data",
        task_id=task_id,
        train_count=len(train_examples),
        test_count=len(test_inputs),
    )

    return challenge


def group_soar_by_task(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Group SOAR data by task_id.

    Args:
        df: SOAR DataFrame

    Returns:
        Dictionary mapping task_id to DataFrame of samples for that task
    """
    grouped = {}
    for task_id in df["task_id"].unique():
        task_df = df[df["task_id"] == task_id].copy()
        grouped[task_id] = task_df

    log.info(
        "Grouped SOAR data by task",
        unique_tasks=len(grouped),
        avg_samples_per_task=f"{len(df)/len(grouped):.1f}",
    )

    return grouped


def get_soar_metadata(row: pd.Series) -> dict:
    """
    Extract SOAR metadata from a DataFrame row for database storage.

    Args:
        row: Single row from SOAR DataFrame

    Returns:
        Dictionary with metadata fields:
        - soar_code: Python code
        - soar_source_model: Source model name
        - soar_generation: Generation number
        - soar_correct_train: List of training correctness
        - soar_correct_test: List of test correctness
        - is_hindsight: Whether this is hindsight data (not all test correct)
    """
    return {
        "soar_code": row["code"],
        "soar_source_model": row["model"],
        "soar_generation": int(row["generation"]),
        "soar_correct_train": row["correct_train_input"],
        "soar_correct_test": row["correct_test_input"],
        "is_hindsight": not all(row["correct_test_input"]),
    }


def load_soar_data_labeled(
    path: Optional[Path] = None, progress_tracker=None
) -> pd.DataFrame:
    """
    Load labeled SOAR dataset and filter out already processed samples.

    Args:
        path: Optional path to labeled parquet file
        progress_tracker: ProgressTracker instance to filter completed samples

    Returns:
        DataFrame with unprocessed samples only
    """
    if path is None:
        path = SOAR_LABELED_PATH

    log.info("Loading labeled SOAR data", path=str(path))

    table = pq.read_table(path)
    df = table.to_pandas()

    log.info(
        "Labeled SOAR data loaded",
        rows=len(df),
        unique_tasks=df["task_id"].nunique(),
        columns=list(df.columns),
    )

    # Filter out already processed samples if progress_tracker provided
    if progress_tracker:
        completed_set = progress_tracker.get_completed_set()
        if completed_set:
            # Filter: keep only uncompleted samples
            mask = df.apply(
                lambda row: (
                    row["task_id"],
                    row["round_index"],
                    row["data_type"],
                )
                not in completed_set,
                axis=1,
            )
            df = df[mask].copy()

            log.info(
                "Filtered out completed samples",
                remaining=len(df),
                completed=len(completed_set),
            )

    return df
