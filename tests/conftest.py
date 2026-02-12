"""
Shared pytest fixtures for testing Solar CAPEX Estimator components.

These fixtures provide small, memory-efficient test data for unit tests.
"""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_data():
    """
    Create a small sample of raw TTS data for testing.

    Memory-efficient: Only 10 rows.
    """
    return pd.DataFrame(
        {
            "installation_date": pd.date_range("2020-01-01", periods=10, freq="M"),
            "customer_segment": ["COM"] * 10,
            "PV_system_size_DC": [
                100.5,
                200.0,
                50.0,
                150.0,
                300.0,
                75.0,
                125.0,
                250.0,
                90.0,
                180.0,
            ],
            "total_installed_price": [
                150000,
                300000,
                75000,
                225000,
                450000,
                112500,
                187500,
                375000,
                135000,
                270000,
            ],
            "state": ["CA", "TX", "NY", "CA", "TX", "NY", "CA", "TX", "NY", "CA"],
            "utility_service_territory": [
                "PG&E",
                "Oncor",
                "ConEd",
                "PG&E",
                "Oncor",
                "ConEd",
                "PG&E",
                "Oncor",
                "ConEd",
                "PG&E",
            ],
            "module_quantity_1": [300, 600, 150, 450, 900, 225, 375, 750, 270, 540],
            "technology_type": ["mono-Si"] * 10,
            "third_party_owned": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "self_installed": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "data_provider_1": ["Provider_A"] * 5 + ["Provider_B"] * 5,
            "system_ID_1": [f"SYS_{i:04d}" for i in range(10)],
            "zip_code": [
                "94102",
                "75201",
                "10001",
                "94103",
                "75202",
                "10002",
                "94104",
                "75203",
                "10003",
                "94105",
            ],
            "city": [
                "San Francisco",
                "Dallas",
                "New York",
                "San Francisco",
                "Dallas",
                "New York",
                "San Francisco",
                "Dallas",
                "New York",
                "San Francisco",
            ],
        }
    )


@pytest.fixture
def sample_raw_data_with_missing():
    """
    Sample data with missing values and sentinel values for testing cleaning.

    Memory-efficient: Only 10 rows.
    """
    data = pd.DataFrame(
        {
            "installation_date": pd.date_range("2020-01-01", periods=10, freq="M"),
            "customer_segment": ["COM"] * 10,
            "PV_system_size_DC": [
                100.5,
                200.0,
                50.0,
                -1,
                300.0,
                75.0,
                125.0,
                -1,
                90.0,
                180.0,
            ],
            "total_installed_price": [
                150000,
                300000,
                np.nan,
                225000,
                5,
                112500,
                187500,
                375000,
                np.nan,
                270000,
            ],
            "state": ["CA", "TX", "NY", "CA", "TX", "NY", None, "TX", "NY", "CA"],
            "utility_service_territory": [
                "PG&E",
                "Oncor",
                "ConEd",
                "PG&E",
                "Oncor",
                "ConEd",
                "PG&E",
                "Oncor",
                "ConEd",
                "PG&E",
            ],
            "module_quantity_1": [300, 600, 150, 450, 900, 225, 375, 750, 270, 540],
            "high_na_column": [np.nan] * 8 + ["value1", "value2"],
            "single_value_column": ["A"] * 10,
            "data_provider_1": ["Provider_A"] * 10,
        }
    )
    return data


@pytest.fixture
def sample_cleaned_data():
    """
    Sample cleaned data ready for feature engineering.

    Memory-efficient: Only 8 rows.
    """
    return pd.DataFrame(
        {
            "installation_date": pd.date_range("2020-01-01", periods=8, freq="M"),
            "PV_system_size_DC": [
                100.5,
                200.0,
                150.0,
                300.0,
                75.0,
                125.0,
                250.0,
                180.0,
            ],
            "total_installed_price": [
                150000,
                300000,
                225000,
                450000,
                112500,
                187500,
                375000,
                270000,
            ],
            "state": ["CA", "TX", "CA", "TX", "NY", "CA", "TX", "CA"],
            "utility_service_territory": [
                "PG&E",
                "Oncor",
                "PG&E",
                "Oncor",
                "ConEd",
                "PG&E",
                "Oncor",
                "PG&E",
            ],
            "module_quantity_1": [300, 600, 450, 900, 225, 375, 750, 540],
            "technology_type": ["mono-Si"] * 8,
        }
    )


@pytest.fixture
def sample_engineered_data():
    """
    Sample data with engineered features for testing preprocessing.

    Memory-efficient: Only 8 rows.
    """
    df = pd.DataFrame(
        {
            "installation_date": pd.date_range("2020-01-01", periods=8, freq="M"),
            "PV_system_size_DC": [
                100.5,
                200.0,
                150.0,
                300.0,
                75.0,
                125.0,
                250.0,
                180.0,
            ],
            "total_installed_price": [
                150000,
                300000,
                225000,
                450000,
                112500,
                187500,
                375000,
                270000,
            ],
            "state": ["CA", "TX", "CA", "TX", "NY", "CA", "TX", "CA"],
            "utility_service_territory": [
                "PG&E",
                "Oncor",
                "PG&E",
                "Oncor",
                "ConEd",
                "PG&E",
                "Oncor",
                "PG&E",
            ],
            "total_module_count": [300, 600, 450, 900, 225, 375, 750, 540],
            "technology_type": ["mono-Si"] * 8,
        }
    )
    # Add days_since_2000
    df["days_since_2000"] = (
        df["installation_date"] - pd.Timestamp("2000-01-01")
    ).dt.days
    return df


@pytest.fixture
def temp_data_dir():
    """
    Create a temporary directory for test data files.

    Automatically cleaned up after test.
    """
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_csv_files(temp_data_dir, sample_raw_data):
    """
    Create sample CSV files in a temporary directory for DataLoader testing.
    """
    # Create two CSV files to test multi-file loading
    file1 = temp_data_dir / "data_2020.csv"
    file2 = temp_data_dir / "data_2021.csv"

    # Split data into two files
    sample_raw_data.iloc[:5].to_csv(file1, index=False)
    sample_raw_data.iloc[5:].to_csv(file2, index=False)

    return temp_data_dir


@pytest.fixture
def config_cleaning():
    """Standard cleaning configuration for tests."""
    return {
        "min_target_value": 10,
        "high_cardinality_threshold": 0.10,
        "na_drop_thresholds": {"string_columns": 0.10, "numeric_columns": 0.50},
    }
