from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

class RFRlTrainer:
    def __init__(self, param_grid: dict, preprocessor):
        self.param_grid = param_grid
        self.model_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(random_state=42, n_estimators=100, max_depth=9))
            ])
    
    def train_new_model(self, X, y, filepath=Path('../models/best_model.pkl')):
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