from typing import Optional

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from .feature_engineer import FeatureEngineer
from .feature_reducer import FeatureReducer
from .preprocessor import TTSPreprocessor


class RFRTrainer:
    def __init__(
        self,
        param_grid: dict,
        preprocessor: TTSPreprocessor,
        feature_engineer: FeatureEngineer,
        feature_reducer: Optional[FeatureReducer] = None,
    ):
        self.param_grid = param_grid
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.feature_reducer = feature_reducer

        # Build pipeline steps based on whether feature_reducer is provided
        pipeline_steps = [("feature_engineering", feature_engineer)]
        if feature_reducer is not None:
            pipeline_steps.append(("feature_reducer", feature_reducer))
        pipeline_steps.extend(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        random_state=42, n_estimators=100, max_depth=9
                    ),
                ),
            ]
        )

        self.model_pipeline = Pipeline(pipeline_steps)

    def train_new_model(self, X, y):
        grid = GridSearchCV(
            self.model_pipeline,
            param_grid=self.param_grid,
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Training new model with GridSearchCV...")
        grid.fit(X_train, y_train)

        print(f"Best parameters discovered: {grid.best_params_}")
        print(f"Best CV MAE: {-grid.best_score_:.4f}")
        print("----------------------------------")
        print("Evaluating best model on test set...")

        # Build train pipeline with same structure as model_pipeline
        train_steps = [("feature_engineering", self.feature_engineer)]
        if self.feature_reducer is not None:
            train_steps.append(("feature_reducer", self.feature_reducer))
        train_steps.extend(
            [
                ("preprocessor", self.preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        random_state=42,
                        **{
                            k.replace("model__", ""): v
                            for k, v in grid.best_params_.items()
                        },
                    ),
                ),
            ]
        )
        train_pipeline = Pipeline(train_steps)
        train_pipeline.fit(X_train, y_train)
        test_mae = mean_absolute_error(y_test, train_pipeline.predict(X_test))

        print(f"Test MAE with best parameters: {test_mae:.4f}")
        print("--------------------------------")
        print("Retraining on full dataset...")

        best_params = grid.best_params_

        # Build best model pipeline with same structure as model_pipeline
        best_steps = [("feature_engineering", self.feature_engineer)]
        if self.feature_reducer is not None:
            best_steps.append(("feature_reducer", self.feature_reducer))
        best_steps.extend(
            [
                ("preprocessor", self.preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        random_state=42,
                        **{k.replace("model__", ""): v for k, v in best_params.items()},
                    ),
                ),
            ]
        )
        best_model = Pipeline(best_steps)

        best_model.fit(X, y)

        return best_model
