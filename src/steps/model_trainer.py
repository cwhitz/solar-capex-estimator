from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib

from .preprocessor import TTSPreprocessor
from .feature_engineer import FeatureEngineer
from .feature_reducer import FeatureReducer

class RFRTrainer:
    def __init__(self, param_grid: dict, preprocessor: TTSPreprocessor, feature_engineer: FeatureEngineer, feature_reducer: FeatureReducer):
        self.param_grid = param_grid
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.feature_reducer = feature_reducer
        
        self.model_pipeline = Pipeline([
                ("feature_engineering", feature_engineer),
                ("feature_reducer", feature_reducer),
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor(random_state=42, n_estimators=100, max_depth=9))
            ])
    
    def train_new_model(self, X, y):
        grid = GridSearchCV(
            self.model_pipeline,
            param_grid=self.param_grid,
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=1
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("Training new model with GridSearchCV...")
        grid.fit(X_train, y_train)

        print(f"Best parameters discovered: {grid.best_params_}")
        print(f"Best CV MAE: {-grid.best_score_:.4f}")
        print("----------------------------------")
        print("Evaluating best model on test set...")
        
        train_pipeline = Pipeline([
                ("feature_engineering", self.feature_engineer),
                ("feature_reducer", self.feature_reducer),
                ("preprocessor", self.preprocessor),
                ("model", RandomForestRegressor(random_state=42, **{k.replace('model__', ''): v for k, v in grid.best_params_.items()}))
            ])
        train_pipeline.fit(X_train, y_train)
        test_mae = mean_absolute_error(y_test, train_pipeline.predict(X_test))
        
        print(f"Test MAE with best parameters: {test_mae:.4f}")
        print('--------------------------------')
        print("Retraining on full dataset...")

        best_params = grid.best_params_

        best_model = Pipeline([
                ("feature_engineering", self.feature_engineer),
                ("feature_reducer", self.feature_reducer),
                ("preprocessor", self.preprocessor),
                ("model", RandomForestRegressor(random_state=42, **{k.replace('model__', ''): v for k, v in best_params.items()}))
            ])
        
        best_model.fit(X, y)

        return best_model