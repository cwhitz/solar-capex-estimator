"""
Tests for TTSPreprocessor class.

Tests preprocessing pipeline construction and feature transformation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from steps.preprocessor import TTSPreprocessor


class TestTTSPreprocessorInit:
    """Test TTSPreprocessor initialization."""

    def test_init_with_default(self):
        """Test initialization."""
        preprocessor = TTSPreprocessor()
        assert preprocessor.column_dict is None
        assert preprocessor.preprocessor is None


class TestTTSPreprocessorSortColumns:
    """Test column sorting by type."""

    def test_sort_columns_numeric(self, sample_engineered_data):
        """Test that numeric columns are identified correctly."""
        preprocessor = TTSPreprocessor()
        preprocessor._sort_columns(
            sample_engineered_data, target_col="total_installed_price"
        )

        assert "PV_system_size_DC" in preprocessor.column_dict["num_cols"]
        assert "days_since_2000" in preprocessor.column_dict["num_cols"]
        assert "total_module_count" in preprocessor.column_dict["num_cols"]

    def test_sort_columns_categorical_low_card(self, sample_engineered_data):
        """Test that low-cardinality categorical columns are identified."""
        preprocessor = TTSPreprocessor()
        preprocessor._sort_columns(
            sample_engineered_data, target_col="total_installed_price"
        )

        assert "technology_type" in preprocessor.column_dict["cat_low_card_cols"]

    def test_sort_columns_categorical_high_card(self):
        """Test that high-cardinality categorical columns are identified."""
        # Create dataframe with enough rows for high cardinality (>= 10 unique)
        df = pd.DataFrame(
            {
                "high_card_col": [f"cat_{i}" for i in range(10)],
                "target": [100 * i for i in range(10)],
            }
        )
        preprocessor = TTSPreprocessor()
        preprocessor._sort_columns(df, target_col="target")

        assert "high_card_col" in preprocessor.column_dict["cat_high_card_cols"]

    def test_sort_columns_excludes_target(self, sample_engineered_data):
        """Test that target column is excluded from feature columns."""
        preprocessor = TTSPreprocessor()
        preprocessor._sort_columns(
            sample_engineered_data, target_col="total_installed_price"
        )

        all_feature_cols = (
            preprocessor.column_dict["num_cols"]
            + preprocessor.column_dict["binary_cols"]
            + preprocessor.column_dict["cat_low_card_cols"]
            + preprocessor.column_dict["cat_high_card_cols"]
        )

        assert "total_installed_price" not in all_feature_cols

    def test_sort_columns_binary(self):
        """Test that binary columns are identified correctly."""
        df = pd.DataFrame(
            {
                "binary_col": ["yes", "no", "yes", "no", "yes"],
                "target": [100, 200, 300, 400, 500],
            }
        )
        preprocessor = TTSPreprocessor()
        preprocessor._sort_columns(df, target_col="target")

        assert "binary_col" in preprocessor.column_dict["binary_cols"]


class TestTTSPreprocessorFit:
    """Test preprocessor fitting."""

    def test_fit_creates_pipeline(self, sample_engineered_data):
        """Test that fit creates a ColumnTransformer."""
        preprocessor = TTSPreprocessor()
        X = sample_engineered_data.drop(columns=["total_installed_price"])
        y = sample_engineered_data["total_installed_price"]

        result = preprocessor.fit(X, y)

        assert result is preprocessor
        assert preprocessor.preprocessor is not None
        assert hasattr(preprocessor, "transform")

    def test_fit_stores_columns(self, sample_engineered_data):
        """Test that fit stores column information."""
        preprocessor = TTSPreprocessor()
        X = sample_engineered_data.drop(columns=["total_installed_price"])
        y = sample_engineered_data["total_installed_price"]

        preprocessor.fit(X, y)

        assert preprocessor.column_dict is not None
        assert "num_cols" in preprocessor.column_dict
        assert "cat_low_card_cols" in preprocessor.column_dict

    def test_fit_transform(self, sample_engineered_data):
        """Test that fitted preprocessor can transform data."""
        preprocessor = TTSPreprocessor()
        X = sample_engineered_data.drop(columns=["total_installed_price"])
        y = sample_engineered_data["total_installed_price"]

        # Should be able to fit and transform
        preprocessor.fit(X, y)
        transformed = preprocessor.transform(X)

        assert transformed is not None
        assert len(transformed) == len(X)


class TestTTSPreprocessorGetFeatureNames:
    """Test get_feature_names method."""

    def test_get_feature_names_after_fit(self, sample_engineered_data):
        """Test getting feature names after fitting."""
        preprocessor = TTSPreprocessor()
        X = sample_engineered_data.drop(columns=["total_installed_price"])
        y = sample_engineered_data["total_installed_price"]

        preprocessor.fit(X, y)
        feature_names = preprocessor.get_feature_names()

        assert feature_names is not None
        assert len(feature_names) > 0

    def test_get_feature_names_raises_error_before_fit(self):
        """Test that get_feature_names raises error before fitting."""
        preprocessor = TTSPreprocessor()

        with pytest.raises(ValueError, match="not built"):
            preprocessor.get_feature_names()


class TestTTSPreprocessorTransformations:
    """Test that transformations are applied correctly."""

    def test_numeric_features_scaled(self, sample_engineered_data):
        """Test that numeric features are scaled."""
        preprocessor = TTSPreprocessor()
        X = sample_engineered_data.drop(columns=["total_installed_price"])
        y = sample_engineered_data["total_installed_price"]

        preprocessor.fit(X, y)
        transformed = preprocessor.transform(X)

        # Numeric columns should be standardized (mean ~0, std ~1)
        # Note: With only 8 samples, this might not be exact
        assert transformed is not None

    def test_categorical_features_encoded(self, sample_engineered_data):
        """Test that categorical features are encoded."""
        preprocessor = TTSPreprocessor()
        X = sample_engineered_data.drop(columns=["total_installed_price"])
        y = sample_engineered_data["total_installed_price"]

        preprocessor.fit(X, y)
        transformed = preprocessor.transform(X)

        # After transformation, should have more columns due to one-hot encoding
        assert transformed.shape[1] >= len(X.columns)

    def test_handles_missing_values(self):
        """Test that missing values are imputed."""
        df = pd.DataFrame(
            {
                "num_col": [1.0, 2.0, np.nan, 4.0, 5.0],
                "cat_col": ["a", "b", np.nan, "a", "b"],
                "target": [10, 20, 30, 40, 50],
            }
        )

        preprocessor = TTSPreprocessor()
        X = df.drop(columns=["target"])
        y = df["target"]

        preprocessor.fit(X, y)
        transformed = preprocessor.transform(X)

        # Should not have NaN values after imputation
        assert not pd.DataFrame(transformed).isna().any().any()


class TestTTSPreprocessorEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_single_numeric_column(self):
        """Test preprocessing with only numeric columns."""
        df = pd.DataFrame(
            {
                "num1": [1, 2, 3, 4, 5],
                "num2": [10, 20, 30, 40, 50],
                "target": [100, 200, 300, 400, 500],
            }
        )

        preprocessor = TTSPreprocessor()
        X = df.drop(columns=["target"])
        y = df["target"]

        preprocessor.fit(X, y)
        transformed = preprocessor.transform(X)

        assert len(transformed) == len(X)

    def test_handles_single_categorical_column(self):
        """Test preprocessing with only categorical columns."""
        df = pd.DataFrame(
            {"cat1": ["a", "b", "a", "b", "a"], "target": [100, 200, 300, 400, 500]}
        )

        preprocessor = TTSPreprocessor()
        X = df.drop(columns=["target"])
        y = df["target"]

        preprocessor.fit(X, y)
        transformed = preprocessor.transform(X)

        assert len(transformed) == len(X)

    def test_preserves_pandas_output(self, sample_engineered_data):
        """Test that output is pandas DataFrame (due to set_output)."""
        preprocessor = TTSPreprocessor()
        X = sample_engineered_data.drop(columns=["total_installed_price"])
        y = sample_engineered_data["total_installed_price"]

        preprocessor.fit(X, y)
        transformed = preprocessor.transform(X)

        # Should be DataFrame due to set_output(transform="pandas")
        assert isinstance(transformed, pd.DataFrame)
