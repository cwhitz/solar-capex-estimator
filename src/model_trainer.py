"""Model training module for solar CAPEX estimation."""

import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


class RFRTrainer:
    """
    Random Forest Regressor trainer with hyperparameter tuning.

    Parameters
    ----------
    param_grid : dict
        Dictionary of hyperparameters to search.
    preprocessor : sklearn ColumnTransformer
        Preprocessing pipeline.

    Attributes
    ----------
    param_grid : dict
        Hyperparameter grid for search.
    model_pipeline : Pipeline
        The full model pipeline including preprocessing and model.

    """

    def __init__(self, param_grid: dict, preprocessor):
        """Initialize the RFRTrainer."""
        self.param_grid = param_grid
        self.model_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(random_state=42, n_estimators=100, max_depth=9))
            ])

    def train_new_model(self, X, y, filepath=Path('../models/best_model.pkl')):
        """
        Train a new model using GridSearchCV and save the best model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data.
        y : pd.Series
            Target data.
        filepath : Path, optional
            Path to save the best model. Default is '../models/best_model.pkl'.

        Returns
        -------
        Pipeline
            The trained best model.
        """
        grid = GridSearchCV(
            self.model_pipeline,
            param_grid=self.param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )

        print("Training new model with GridSearchCV...")
        grid.fit(X, y)

        print(f"Best parameters discovered: {grid.best_params_}")
        print(f"Best CV RMSE: {np.sqrt(-grid.best_score_):.4f}")
        print("----------------------------------")
        print("Retraining best model on full dataset...")

        best_params = grid.best_params_

        best_model = Pipeline([
                ("preprocessor", self.model_pipeline.named_steps["preprocessor"]),
                ("model", RandomForestRegressor(random_state=42, **{k.replace('model__', ''): v for k, v in best_params.items()}))
            ])

        best_model.fit(X, y)


        joblib.dump(best_model, filepath)

        print(f"Best model saved to {filepath}")

        return best_model
