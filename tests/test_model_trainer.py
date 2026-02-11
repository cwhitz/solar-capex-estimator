"""
Tests for RFRTrainer class.

Tests model training functionality with minimal data for memory efficiency.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from steps.model_trainer import RFRTrainer
from steps.preprocessor import TTSPreprocessor
from steps.feature_engineer import FeatureEngineer


@pytest.fixture
def minimal_preprocessor():
    """Create a minimal preprocessor for testing (memory-efficient)."""
    return TTSPreprocessor()


@pytest.fixture
def minimal_feature_engineer():
    """Create a minimal feature engineer for testing."""
    return FeatureEngineer()


@pytest.fixture
def minimal_param_grid():
    """Minimal parameter grid for fast testing."""
    return {
        'model__n_estimators': [10],  # Small number for speed
        'model__max_depth': [3],
        'model__min_samples_split': [2]
    }


@pytest.fixture
def minimal_training_data():
    """Create minimal training data (memory-efficient: 10 rows for 5-fold CV)."""
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']
    })
    y = pd.Series([100, 200, 150, 250, 120, 220, 140, 240, 160, 260])

    return X, y


class TestRFRTrainerInit:
    """Test RFRTrainer initialization."""

    def test_init_with_parameters(self, minimal_preprocessor, minimal_feature_engineer, minimal_param_grid):
        """Test initialization with preprocessor and param_grid."""
        trainer = RFRTrainer(
            preprocessor=minimal_preprocessor,
            feature_engineer=minimal_feature_engineer,
            param_grid=minimal_param_grid
        )

        assert trainer.param_grid == minimal_param_grid
        assert trainer.model_pipeline is not None

    def test_init_creates_pipeline(self, minimal_preprocessor, minimal_feature_engineer, minimal_param_grid):
        """Test that initialization creates a pipeline with preprocessor and model."""
        trainer = RFRTrainer(
            preprocessor=minimal_preprocessor,
            feature_engineer=minimal_feature_engineer,
            param_grid=minimal_param_grid
        )

        assert 'preprocessor' in trainer.model_pipeline.named_steps
        assert 'feature_engineering' in trainer.model_pipeline.named_steps
        assert 'model' in trainer.model_pipeline.named_steps


class TestRFRTrainerTraining:
    """Test model training functionality."""

    def test_train_new_model_returns_pipeline(
        self, minimal_preprocessor, minimal_feature_engineer, minimal_param_grid, minimal_training_data
    ):
        """Test that train_new_model returns a trained pipeline."""
        trainer = RFRTrainer(
            preprocessor=minimal_preprocessor,
            feature_engineer=minimal_feature_engineer,
            param_grid=minimal_param_grid
        )

        X, y = minimal_training_data
        model = trainer.train_new_model(X, y)

        assert model is not None
        assert hasattr(model, 'predict')

    def test_train_new_model_can_predict(
        self, minimal_preprocessor, minimal_feature_engineer, minimal_param_grid, minimal_training_data
    ):
        """Test that trained model can make predictions."""
        trainer = RFRTrainer(
            preprocessor=minimal_preprocessor,
            feature_engineer=minimal_feature_engineer,
            param_grid=minimal_param_grid
        )

        X, y = minimal_training_data
        model = trainer.train_new_model(X, y)

        # Should be able to predict on same data
        predictions = model.predict(X)

        assert predictions is not None
        assert len(predictions) == len(X)

    def test_train_new_model_performs_grid_search(
        self, minimal_preprocessor, minimal_feature_engineer, minimal_training_data
    ):
        """Test that training performs grid search."""
        # Use multiple parameter values to test grid search
        param_grid = {
            'model__n_estimators': [5, 10],
            'model__max_depth': [2, 3]
        }

        trainer = RFRTrainer(
            preprocessor=minimal_preprocessor,
            feature_engineer=minimal_feature_engineer,
            param_grid=param_grid
        )

        X, y = minimal_training_data
        model = trainer.train_new_model(X, y)

        # Model should be trained with one of the parameter combinations
        assert model.named_steps['model'].n_estimators in [5, 10]
        assert model.named_steps['model'].max_depth in [2, 3]

    def test_trained_model_has_correct_structure(
        self, minimal_preprocessor, minimal_feature_engineer, minimal_param_grid, minimal_training_data
    ):
        """Test that trained model has expected structure."""
        trainer = RFRTrainer(
            preprocessor=minimal_preprocessor,
            feature_engineer=minimal_feature_engineer,
            param_grid=minimal_param_grid
        )

        X, y = minimal_training_data
        model = trainer.train_new_model(X, y)

        # Should have preprocessor and model steps
        assert 'preprocessor' in model.named_steps
        assert 'model' in model.named_steps

        # Model should be a Random Forest
        assert hasattr(model.named_steps['model'], 'estimators_')


class TestRFRTrainerPredictions:
    """Test prediction quality (basic sanity checks)."""

    def test_predictions_are_reasonable(
        self, minimal_preprocessor, minimal_feature_engineer, minimal_param_grid, minimal_training_data
    ):
        """Test that predictions are in reasonable range."""
        trainer = RFRTrainer(
            preprocessor=minimal_preprocessor,
            feature_engineer=minimal_feature_engineer,
            param_grid=minimal_param_grid
        )

        X, y = minimal_training_data
        model = trainer.train_new_model(X, y)

        predictions = model.predict(X)

        # Predictions should be in similar range to training data
        assert predictions.min() >= 0
        assert predictions.min() < y.max()
        assert predictions.max() > y.min()

    def test_model_learns_from_data(
        self, minimal_preprocessor, minimal_feature_engineer, minimal_param_grid, minimal_training_data
    ):
        """Test that model captures general pattern in data."""
        trainer = RFRTrainer(
            preprocessor=minimal_preprocessor,
            feature_engineer=minimal_feature_engineer,
            param_grid=minimal_param_grid
        )

        X, y = minimal_training_data
        model = trainer.train_new_model(X, y)

        predictions = model.predict(X)

        # R² score should be positive (better than predicting mean)
        from sklearn.metrics import r2_score
        r2 = r2_score(y, predictions)
        assert r2 > 0


class TestRFRTrainerEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_minimal_data(self, minimal_preprocessor, minimal_feature_engineer):
        """Test training with very minimal data."""
        # 10 samples for 5-fold CV compatibility
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']
        })
        y = pd.Series([100, 200, 150, 250, 120, 220, 180, 280, 160, 260])

        param_grid = {
            'model__n_estimators': [5],
            'model__max_depth': [2]
        }

        trainer = RFRTrainer(
            preprocessor=minimal_preprocessor,
            feature_engineer=minimal_feature_engineer,
            param_grid=param_grid
        )

        # Should still train without error
        model = trainer.train_new_model(X, y)
        assert model is not None

    def test_predictions_maintain_shape(
        self, minimal_preprocessor, minimal_feature_engineer, minimal_param_grid, minimal_training_data
    ):
        """Test that predictions maintain input shape."""
        trainer = RFRTrainer(
            preprocessor=minimal_preprocessor,
            feature_engineer=minimal_feature_engineer,
            param_grid=minimal_param_grid
        )

        X, y = minimal_training_data
        model = trainer.train_new_model(X, y)

        # Test with different sized inputs
        predictions = model.predict(X[:3])
        assert len(predictions) == 3

        predictions = model.predict(X)
        assert len(predictions) == len(X)
