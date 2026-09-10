"""Shared pytest fixtures. Adds ``src`` to the path so the package imports."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from churnguard import config, data  # noqa: E402


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """The immutable raw dataset, skipping the suite if it has not been fetched."""
    if not config.RAW_DATA_PATH.exists():
        pytest.skip(f"Raw dataset missing at {config.RAW_DATA_PATH}")
    return data.load_raw_data()


@pytest.fixture(scope="session")
def prepared_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return data.prepare_dataset(raw_df)


@pytest.fixture(scope="session")
def splits(prepared_df: pd.DataFrame) -> data.DataSplits:
    return data.make_splits(prepared_df)
