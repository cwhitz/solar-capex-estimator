"""
Tests for DataLoader class.

Tests data loading, filtering, and validation functionality.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from steps.data_loader import DataLoader


class TestDataLoaderInit:
    """Test DataLoader initialization."""

    def test_init_with_valid_directory(self, temp_data_dir):
        """Test initialization with valid directory."""
        loader = DataLoader(tts_data_directory=str(temp_data_dir))
        assert loader.tts_data_directory == temp_data_dir
        assert loader.df is None

    def test_init_stores_valid_customer_segments(self):
        """Test that valid customer segments are stored."""
        loader = DataLoader(tts_data_directory="./data/raw")
        assert "COM" in loader.valid_customer_segments
        assert "RES" in loader.valid_customer_segments


class TestDataLoaderLoading:
    """Test data loading functionality."""

    def test_load_training_data_single_file(self, sample_csv_files):
        """Test loading data from CSV files."""
        loader = DataLoader(tts_data_directory=str(sample_csv_files))
        loader.load_training_data()

        assert loader.df is not None
        assert len(loader.df) == 10  # 5 + 5 from two files
        assert "installation_date" in loader.df.columns

    def test_load_training_data_with_year_filter(self, sample_csv_files):
        """Test loading with year filtering."""
        loader = DataLoader(tts_data_directory=str(sample_csv_files))
        loader.load_training_data(year_min=2020, year_max=2020)

        assert loader.df is not None
        years = loader.df["installation_date"].dt.year.unique()
        assert all(year == 2020 for year in years)

    def test_load_training_data_with_customer_segment_filter(self, sample_csv_files):
        """Test loading with customer segment filtering."""
        loader = DataLoader(tts_data_directory=str(sample_csv_files))
        loader.load_training_data(customer_segments=["COM"])

        assert loader.df is not None
        assert all(loader.df["customer_segment"] == "COM")

    def test_load_training_data_no_csv_files(self, temp_data_dir):
        """Test that error is raised when no CSV files found."""
        loader = DataLoader(tts_data_directory=str(temp_data_dir))

        with pytest.raises(ValueError, match="No CSV files found"):
            loader.load_training_data()


class TestDataLoaderValidation:
    """Test validation methods."""

    def test_validate_filters_invalid_year_range(self, temp_data_dir):
        """Test validation fails with invalid year range."""
        loader = DataLoader(tts_data_directory=str(temp_data_dir))

        with pytest.raises(
            ValueError, match="year_min cannot be greater than year_max"
        ):
            loader._validate_filters(
                year_min=2022, year_max=2020, customer_segments=None
            )

    def test_validate_filters_invalid_customer_segments_type(self, temp_data_dir):
        """Test validation fails with invalid customer segments type."""
        loader = DataLoader(tts_data_directory=str(temp_data_dir))

        with pytest.raises(ValueError, match="must be a list of strings"):
            loader._validate_filters(
                year_min=None, year_max=None, customer_segments="COM"
            )

    def test_validate_filters_invalid_customer_segments_values(self, temp_data_dir):
        """Test validation fails with invalid customer segments."""
        loader = DataLoader(tts_data_directory=str(temp_data_dir))

        with pytest.raises(ValueError, match="must be a subset"):
            loader._validate_filters(
                year_min=None, year_max=None, customer_segments=["INVALID"]
            )


class TestDataLoaderFiltering:
    """Test filtering methods."""

    def test_filter_by_years_min_only(self, sample_raw_data):
        """Test filtering by minimum year only."""
        loader = DataLoader(tts_data_directory="./data/raw")
        loader.df = sample_raw_data.copy()

        filtered = loader._filter_by_years(loader.df, year_min=2020, year_max=None)

        assert all(filtered["installation_date"].dt.year >= 2020)

    def test_filter_by_years_max_only(self, sample_raw_data):
        """Test filtering by maximum year only."""
        loader = DataLoader(tts_data_directory="./data/raw")
        loader.df = sample_raw_data.copy()

        filtered = loader._filter_by_years(loader.df, year_min=None, year_max=2020)

        assert all(filtered["installation_date"].dt.year <= 2020)

    def test_filter_by_customer_segment(self, sample_raw_data):
        """Test filtering by customer segment."""
        loader = DataLoader(tts_data_directory="./data/raw")
        loader.df = sample_raw_data.copy()

        filtered = loader._filter_by_customer_segment(loader.df, segments=["COM"])

        assert all(filtered["customer_segment"] == "COM")
        assert len(filtered) == 10  # All rows are COM in sample data


class TestDataLoaderGetData:
    """Test get_data method."""

    def test_get_data_returns_dataframe(self, sample_csv_files):
        """Test get_data returns the loaded dataframe."""
        loader = DataLoader(tts_data_directory=str(sample_csv_files))
        loader.load_training_data()

        df = loader.get_data()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10

    def test_get_data_raises_error_when_not_loaded(self):
        """Test get_data raises error when data not loaded."""
        loader = DataLoader(tts_data_directory="./data/raw")

        with pytest.raises(ValueError, match="Data not loaded"):
            loader.get_data()
