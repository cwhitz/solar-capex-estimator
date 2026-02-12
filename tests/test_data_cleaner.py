"""
Tests for DataCleaner class.

Tests data cleaning, validation, and transformation functionality.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from steps.data_cleaner import DataCleaner


class TestDataCleanerInit:
    """Test DataCleaner initialization."""

    def test_init_with_default_parameters(self):
        """Test initialization with default parameters."""
        cleaner = DataCleaner()
        assert cleaner.min_target_value == 10
        assert cleaner.high_cardinality_threshold == 0.10
        assert cleaner.df is None
        assert cleaner.na_drop_thresholds == {
            "string_columns": 0.10,
            "numeric_columns": 0.50,
        }

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        cleaner = DataCleaner(
            min_target_value=100,
            high_cardinality_threshold=0.2,
            na_drop_thresholds={"string_columns": 0.3, "numeric_columns": 0.6},
        )
        assert cleaner.min_target_value == 100
        assert cleaner.high_cardinality_threshold == 0.2
        assert cleaner.na_drop_thresholds["string_columns"] == 0.3


class TestDataCleanerSentinelValues:
    """Test sentinel value replacement."""

    def test_make_true_na_replaces_minus_one(self):
        """Test that -1 values are replaced with NaN."""
        cleaner = DataCleaner()
        df = pd.DataFrame({"col1": [1, -1, 3, -1, 5]})

        result = cleaner._make_true_na(df)

        assert result["col1"].isna().sum() == 2
        assert result["col1"].tolist()[:1] == [1.0]

    def test_make_true_na_replaces_string_minus_one(self):
        """Test that '-1' string values are replaced with NaN."""
        cleaner = DataCleaner()
        df = pd.DataFrame({"col1": ["a", "-1", "c", "-1", "e"]})

        result = cleaner._make_true_na(df)

        assert result["col1"].isna().sum() == 2


class TestDataCleanerTargetCleaning:
    """Test target variable cleaning."""

    def test_clean_by_target_removes_missing_values(self, sample_raw_data_with_missing):
        """Test that rows with missing target values are removed."""
        cleaner = DataCleaner(min_target_value=10)
        cleaner.load_data(sample_raw_data_with_missing)

        result = cleaner._clean_by_target(
            cleaner.df, target_col="total_installed_price"
        )

        # Should remove 2 NaN rows and 1 row with value 5 (< 10)
        assert len(result) < len(sample_raw_data_with_missing)
        assert result["total_installed_price"].notna().all()
        assert (result["total_installed_price"] >= 10).all()

    def test_clean_by_target_removes_low_values(self):
        """Test that rows below minimum target value are removed."""
        cleaner = DataCleaner(min_target_value=1000)
        df = pd.DataFrame({"price": [500, 1500, 2000, 100, 3000]})
        cleaner.load_data(df)

        result = cleaner._clean_by_target(cleaner.df, target_col="price")

        assert len(result) == 3
        assert (result["price"] >= 1000).all()


class TestDataCleanerColumnDropping:
    """Test column dropping methods."""

    def test_drop_high_na_columns_string_threshold(self):
        """Test dropping string columns with high NA proportion."""
        cleaner = DataCleaner(
            na_drop_thresholds={"string_columns": 0.5, "numeric_columns": 0.8}
        )
        df = pd.DataFrame(
            {
                "col1": ["a", "b", "c", "d", "e"],
                "col2": ["x", np.nan, np.nan, np.nan, "y"],  # 60% NA
                "col3": [1, 2, 3, 4, 5],
            }
        )
        cleaner.load_data(df)

        result = cleaner._drop_high_na_columns(cleaner.df)

        assert "col2" not in result.columns  # Should be dropped
        assert "col1" in result.columns
        assert "col3" in result.columns

    def test_drop_high_na_columns_numeric_threshold(self):
        """Test dropping numeric columns with high NA proportion."""
        cleaner = DataCleaner(
            na_drop_thresholds={"string_columns": 0.5, "numeric_columns": 0.5}
        )
        df = pd.DataFrame(
            {
                "col1": [1.0, np.nan, np.nan, np.nan, 5.0],  # 60% NA
                "col2": [1, 2, 3, 4, 5],
            }
        )
        cleaner.load_data(df)

        result = cleaner._drop_high_na_columns(cleaner.df)

        assert "col1" not in result.columns  # Should be dropped
        assert "col2" in result.columns

    def test_drop_high_cardinality_columns(self):
        """Test dropping high cardinality string columns."""
        cleaner = DataCleaner(high_cardinality_threshold=0.5)
        df = pd.DataFrame(
            {
                "low_card": ["a", "b", "a", "b", "a"],  # 2/5 = 40% unique
                "high_card": ["a", "b", "c", "d", "e"],  # 5/5 = 100% unique
                "numeric": [
                    1,
                    2,
                    3,
                    4,
                    5,
                ],  # Should not be dropped even if high cardinality
            }
        )
        cleaner.load_data(df)

        result = cleaner._drop_high_cardinality_columns(cleaner.df)

        assert "high_card" not in result.columns
        assert "low_card" in result.columns
        assert "numeric" in result.columns

    def test_drop_single_value_columns(self):
        """Test dropping columns with single value."""
        cleaner = DataCleaner()
        df = pd.DataFrame(
            {
                "col1": ["a", "a", "a", "a", "a"],
                "col2": ["x", "y", "x", "y", "x"],
                "col3": [1, 2, 3, 4, 5],
            }
        )
        cleaner.load_data(df)

        result = cleaner._drop_single_value_columns(cleaner.df)

        assert "col1" not in result.columns
        assert "col2" in result.columns
        assert "col3" in result.columns

    def test_drop_id_and_provider_columns(self):
        """Test dropping ID and provider columns."""
        cleaner = DataCleaner()
        df = pd.DataFrame(
            {
                "data_provider_1": ["A", "B", "C"],
                "system_ID_1": ["ID1", "ID2", "ID3"],
                "keep_this": [1, 2, 3],
            }
        )
        cleaner.load_data(df)

        result = cleaner._drop_id_and_provider_columns(cleaner.df)

        assert "data_provider_1" not in result.columns
        assert "system_ID_1" not in result.columns
        assert "keep_this" in result.columns


class TestDataCleanerDatatypeCoercion:
    """Test datatype coercion."""

    def test_coerce_datatypes_date_columns(self):
        """Test that date columns are coerced to datetime."""
        cleaner = DataCleaner()
        df = pd.DataFrame(
            {"installation_date": ["2020-01-01", "2020-02-01", "2020-03-01"]}
        )

        result = cleaner._coerce_datatypes(df)

        assert pd.api.types.is_datetime64_any_dtype(result["installation_date"])

    def test_coerce_datatypes_zip_columns(self):
        """Test that zip columns are coerced to zero-padded strings."""
        cleaner = DataCleaner()
        df = pd.DataFrame({"zip_code": ["1234", "5678", "90"]})

        result = cleaner._coerce_datatypes(df)

        assert result["zip_code"].iloc[2] == "00090"

    def test_coerce_datatypes_numeric_conversion(self):
        """Test that numeric strings are converted to numbers."""
        cleaner = DataCleaner()
        df = pd.DataFrame({"value": ["123", "456", "789"]})

        result = cleaner._coerce_datatypes(df)

        assert pd.api.types.is_numeric_dtype(result["value"])


class TestDataCleanerFullPipeline:
    """Test complete cleaning pipeline."""

    def test_clean_full_pipeline(self, sample_raw_data_with_missing, config_cleaning):
        """Test complete cleaning pipeline."""
        cleaner = DataCleaner(**config_cleaning)
        cleaner.load_data(sample_raw_data_with_missing)

        result = cleaner.clean(target_col="total_installed_price")

        # Should have removed rows with invalid targets
        assert len(result) < len(sample_raw_data_with_missing)
        # Should have dropped high NA columns
        assert "high_na_column" not in result.columns
        # Should have dropped single value columns
        assert "single_value_column" not in result.columns
        # Should have dropped ID columns
        assert "data_provider_1" not in result.columns

    def test_clean_raises_error_without_data(self):
        """Test that clean raises error when data not loaded."""
        cleaner = DataCleaner()

        with pytest.raises(ValueError, match="Data not loaded"):
            cleaner.clean()


class TestDataCleanerLoadData:
    """Test load_data method."""

    def test_load_data_stores_dataframe(self, sample_raw_data):
        """Test that load_data stores the dataframe."""
        cleaner = DataCleaner()
        cleaner.load_data(sample_raw_data)

        assert cleaner.df is not None
        assert len(cleaner.df) == len(sample_raw_data)
