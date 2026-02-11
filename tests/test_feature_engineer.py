"""
Tests for FeatureEngineer class.

Tests feature engineering transformations.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from steps.feature_engineer import FeatureEngineer


class TestFeatureEngineerInit:
    """Test FeatureEngineer initialization."""

    def test_init(self):
        """Test initialization."""
        engineer = FeatureEngineer()
        assert engineer is not None


class TestFeatureEngineerTransform:
    """Test transform method."""

    def test_transform_returns_dataframe(self, sample_cleaned_data):
        """Test that transform returns a dataframe."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(sample_cleaned_data)

        assert result is not None
        assert len(result) == len(sample_cleaned_data)


class TestFeatureEngineerDayCount:
    """Test days_since_2000 feature creation."""

    def test_add_day_count_creates_column(self, sample_cleaned_data):
        """Test that _add_day_count creates days_since_2000 column."""
        engineer = FeatureEngineer()
        result = engineer._add_day_count(sample_cleaned_data.copy())

        assert 'days_since_2000' in result.columns
        assert pd.api.types.is_integer_dtype(result['days_since_2000'])

    def test_add_day_count_correct_calculation(self):
        """Test that days_since_2000 is calculated correctly."""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'installation_date': [pd.Timestamp('2000-01-01'), pd.Timestamp('2000-01-02')]
        })

        result = engineer._add_day_count(df)

        assert result['days_since_2000'].iloc[0] == 0
        assert result['days_since_2000'].iloc[1] == 1

    def test_add_day_count_handles_various_dates(self):
        """Test that _add_day_count works with various dates."""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'installation_date': pd.date_range('2020-01-01', periods=5, freq='M')
        })

        result = engineer._add_day_count(df)

        # All values should be positive (after 2000-01-01)
        assert (result['days_since_2000'] > 0).all()
        # Values should be in ascending order
        assert result['days_since_2000'].is_monotonic_increasing


class TestFeatureEngineerModuleCount:
    """Test total_module_count feature creation."""

    def test_combine_module_counts_single_column(self):
        """Test combining module counts with single column."""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'module_quantity_1': [100, 200, 300]
        })
        result = engineer._combine_module_counts(df)

        assert 'total_module_count' in result.columns
        assert result['total_module_count'].tolist() == [100, 200, 300]
        assert 'module_quantity_1' not in result.columns

    def test_combine_module_counts_multiple_columns(self):
        """Test combining module counts with multiple columns."""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'module_quantity_1': [100, 200, 300],
            'module_quantity_2': [50, 0, 100],
            'module_quantity_3': [25, 50, 0]
        })
        result = engineer._combine_module_counts(df)

        assert 'total_module_count' in result.columns
        assert result['total_module_count'].tolist() == [175, 250, 400]
        assert 'module_quantity_1' not in result.columns
        assert 'module_quantity_2' not in result.columns
        assert 'module_quantity_3' not in result.columns

    def test_combine_module_counts_handles_nan(self):
        """Test that NaN values are treated as 0."""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'module_quantity_1': [100, 200, 300],
            'module_quantity_2': [50, np.nan, 100]
        })
        result = engineer._combine_module_counts(df)

        assert result['total_module_count'].tolist() == [150, 200, 400]

    def test_combine_module_counts_skips_if_exists(self):
        """Test that _combine_module_counts skips if column already exists."""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'module_quantity_1': [100, 200, 300],
            'total_module_count': [999, 999, 999]  # Pre-existing
        })
        result = engineer._combine_module_counts(df)

        # Should not modify existing column
        assert result['total_module_count'].tolist() == [999, 999, 999]


class TestFeatureEngineerFullPipeline:
    """Test complete feature engineering pipeline."""

    def test_engineer_features_complete_pipeline(self, sample_cleaned_data):
        """Test complete feature engineering pipeline."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(sample_cleaned_data)

        # Should have days_since_2000
        assert 'days_since_2000' in result.columns
        # Should have total_module_count
        assert 'total_module_count' in result.columns
        # Should have dropped module_quantity columns
        assert 'module_quantity_1' not in result.columns

    def test_engineer_features_preserves_other_columns(self, sample_cleaned_data):
        """Test that other columns are preserved."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(sample_cleaned_data)

        # Original columns should still be there
        assert 'installation_date' in result.columns
        assert 'PV_system_size_DC' in result.columns
        assert 'state' in result.columns

    def test_engineer_features_maintains_row_count(self, sample_cleaned_data):
        """Test that feature engineering doesn't change row count."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(sample_cleaned_data)

        assert len(result) == len(sample_cleaned_data)


class TestFeatureEngineerEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_empty_dataframe(self):
        """Test that feature engineering handles empty dataframe."""
        engineer = FeatureEngineer()
        df = pd.DataFrame(columns=['installation_date'])
        result = engineer.fit_transform(df)

        assert len(result) == 0
        assert 'days_since_2000' in result.columns

    def test_handles_single_row(self):
        """Test that feature engineering handles single row."""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'installation_date': [pd.Timestamp('2020-01-01')],
            'module_quantity_1': [100]
        })
        result = engineer.fit_transform(df)

        assert len(result) == 1
        assert result['days_since_2000'].iloc[0] > 0
        assert result['total_module_count'].iloc[0] == 100
