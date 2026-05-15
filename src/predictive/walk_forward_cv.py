"""
Walk-forward validation utilities.

These helpers keep chronological train/test splitting reusable outside the
XGBoost training module and make leakage checks easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    train_quarters: tuple[int, ...]
    test_quarter: int
    train_index: pd.Index
    test_index: pd.Index


def iter_walk_forward_splits(
    df: pd.DataFrame,
    time_col: str = "quarter",
    min_train_periods: int = 4,
) -> Iterator[WalkForwardSplit]:
    """Yield expanding-window splits with one future period held out."""
    if time_col not in df.columns:
        raise ValueError(f"Missing time column: {time_col}")
    if min_train_periods < 1:
        raise ValueError("min_train_periods must be >= 1")

    quarters = tuple(sorted(int(q) for q in df[time_col].dropna().unique()))
    for i, test_quarter in enumerate(quarters):
        if i < min_train_periods:
            continue

        train_quarters = quarters[:i]
        train_mask = df[time_col].isin(train_quarters)
        test_mask = df[time_col] == test_quarter
        yield WalkForwardSplit(
            train_quarters=train_quarters,
            test_quarter=test_quarter,
            train_index=df.index[train_mask],
            test_index=df.index[test_mask],
        )


def validate_no_lookahead(splits: list[WalkForwardSplit]) -> None:
    """Raise if any split trains on the test period or a future period."""
    for split in splits:
        if not split.train_quarters:
            raise ValueError("Each split must contain training quarters")
        if max(split.train_quarters) >= split.test_quarter:
            raise ValueError(
                f"Look-ahead detected: train quarters {split.train_quarters} "
                f"for test quarter {split.test_quarter}"
            )
