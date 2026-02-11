import json

import joblib
import numpy as np
import pandas as pd

from src.config import config
from src.steps.data_cleaner import DataCleaner
from src.steps.data_loader import DataLoader
from src.steps.feature_engineer import FeatureEngineer
from src.steps.model_trainer import RFRTrainer
from src.steps.preprocessor import Preprocessor
from src.utils.input_validation import validate_prediction_requests


class SolarCapexEstimator:
    """
    Class for estimating solar capital expenditures (CAPEX) using a machine learning pipeline.

    This class has four public methods:
    - train_model: Trains a Random Forest Regressor model using any data in the specified directory.
    - load_model: Loads a previously trained model from disk.
    - predict: Uses the trained model to make one or more predictions on new data.
    - predict_from_csv: Loads new data from a CSV file and makes predictions using the trained model.
    """

    def __init__(self, data_directory="./data/raw"):
        self.data_directory = data_directory
        self.model = None

    def train_model(self):
        # Load data
        tts_dataloader = DataLoader(tts_data_directory=self.data_directory)
        tts_dataloader.load_training_data(**config["loading"])
        data = tts_dataloader.get_data()

        # Clean data
        cleaner = DataCleaner(**config["cleaning"])
        cleaner.load_data(data)
        cleaned_data = cleaner.clean(config["model_features"]["target"])

        # Engineer features
        engineer = FeatureEngineer()
        engineer.load_data(cleaned_data)
        engineered_data = engineer.engineer_features()

        # minimize engineered data to just features used in model
        # ensure engineered features like 'days_since_2000' are retained
        engineered_extra_features = ["days_since_2000"]
        engineered_data = engineered_data.filter(
            items=config["model_features"]["features"]
            + engineered_extra_features
            + [config["model_features"]["target"]]
        )

        # Build preprocessor
        preprocessor = Preprocessor(target_col=config["model_features"]["target"])
        built_preprocessor = preprocessor.build_preprocessor(engineered_data)

        # Train model
        trainer = RFRTrainer(
            preprocessor=built_preprocessor,
            param_grid=config["hyperparameter_search"]["param_grid"],
        )
        self.model = trainer.train_new_model(
            engineered_data.drop(columns=[config["model_features"]["target"]]),
            engineered_data[config["model_features"]["target"]],
        )

        return self.model

    def save_model(self, pipeline, filename=None):
        """
        Save the trained model pipeline to disk.

        Parameters
        ----------
        pipeline : sklearn.pipeline.Pipeline
            The trained model pipeline to save.
        filename : str, optional
            Optional custom filename for the saved model. If not provided, a default name with timestamp will be used.

        Returns
        -------
        str
            The filepath where the model was saved.

        Raises
        ------
        Exception
            If the model cannot be saved.
        """
        try:
            filename = (
                filename
                or f"solar_capex_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pkl"
            )
            filepath = f"./models/{filename}"
            joblib.dump(pipeline, filepath)
            return filepath

        except Exception as e:
            raise Exception(f"Failed to save model: {str(e)}") from e

    def load_model(self, filepath):
        """
        Load a previously trained model from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model file (.pkl).

        Returns
        -------
        sklearn.pipeline.Pipeline
            The loaded model pipeline.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        Exception
            If the model cannot be loaded.
        """
        try:
            self.model = joblib.load(filepath)
            return self.model
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found: {str(e)}") from e
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}") from e

    def predict(self, prediction_requests):
        """
        Make predictions on new data using the trained model.

        Parameters
        ----------
        prediction_requests : list of dict
            List of prediction request dictionaries. Each dict must contain:
            - PV_system_size_DC (float): System size in kW
            - state (str): Two-letter state code
            - utility_service_territory (str): Utility name
            - total_module_count (int): Number of modules
            - installation_date (str or datetime): Installation date

        Returns
        -------
        list of dict
            List of prediction results, each containing:
            - prediction (float): Predicted total installed price in dollars
            - uncertainty (float): Prediction uncertainty (std dev across trees)
            - confidence (float): Confidence score (0-1)

        Raises
        ------
        ValueError
            If model has not been loaded or if input validation fails.
        Exception
            If prediction fails.
        """
        self._validate_model_loaded()
        validated_requests = self._validate_inputs(prediction_requests)
        X = self._requests_to_dataframe(validated_requests)

        return self._predict_with_uncertainty(X)

    def _validate_model_loaded(self):
        """Ensure model is loaded before prediction."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

    def _validate_inputs(self, prediction_requests):
        """
        Validate prediction requests using Pydantic.

        Parameters
        ----------
        prediction_requests : list of dict
            Raw prediction requests.

        Returns
        -------
        list of PredictionRequest
            Validated prediction requests.

        Raises
        ------
        ValueError
            If validation fails.
        """
        try:
            return validate_prediction_requests(prediction_requests)
        except Exception as e:
            raise ValueError(f"Input validation failed: {str(e)}") from e

    def _requests_to_dataframe(self, validated_requests):
        """
        Convert validated requests to DataFrame.

        Parameters
        ----------
        validated_requests : list of PredictionRequest
            Validated prediction requests.

        Returns
        -------
        pd.DataFrame
            DataFrame with prediction features.
        """

        df = pd.DataFrame(
            [
                request.model_dump() if hasattr(request, "model_dump") else request.dict()
                for request in validated_requests
            ]
        )

        df['days_since_2000'] = (pd.to_datetime(df['installation_date']) - pd.Timestamp("2000-01-01")).dt.days

        return df

    def _predict_with_uncertainty(self, X):
        """
        Make predictions with uncertainty estimates for each row.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        list of dict
            Prediction results with uncertainty metrics.

        Raises
        ------
        Exception
            If prediction fails.
        """
        try:
            return [self._predict_single_row(X.iloc[[i]]) for i in range(len(X))]
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}") from e

    def _predict_single_row(self, X_row):
        """
        Make prediction with uncertainty for a single row.

        Parameters
        ----------
        X_row : pd.DataFrame
            Single row DataFrame.

        Returns
        -------
        dict
            Prediction result with uncertainty metrics.
        """
        prediction = self.model.predict(X_row)[0]
        tree_predictions = self._get_tree_predictions(X_row)
        uncertainty = self._calculate_uncertainty(tree_predictions)
        confidence = self._calculate_confidence(prediction, uncertainty)

        return {
            "prediction": round(float(prediction), 2),
            "uncertainty": round(float(uncertainty), 2),
            "confidence": round(float(confidence), 2),
        }

    def _get_tree_predictions(self, X_row):
        """
        Get predictions from all trees in the Random Forest.

        Parameters
        ----------
        X_row : pd.DataFrame
            Single row DataFrame.

        Returns
        -------
        np.ndarray
            Array of predictions from each tree.
        """
        X_transformed = self.model.named_steps["preprocessor"].transform(X_row)
        return np.array(
            [
                tree.predict(X_transformed)[0]
                for tree in self.model.named_steps["model"].estimators_
            ]
        )

    def _calculate_uncertainty(self, tree_predictions):
        """
        Calculate prediction uncertainty as standard deviation across trees.

        Parameters
        ----------
        tree_predictions : np.ndarray
            Predictions from all trees.

        Returns
        -------
        float
            Uncertainty (standard deviation).
        """
        return tree_predictions.std()

    def _calculate_confidence(self, prediction, uncertainty):
        """
        Calculate confidence score based on relative uncertainty.

        Parameters
        ----------
        prediction : float
            Main prediction value.
        uncertainty : float
            Prediction uncertainty.

        Returns
        -------
        float
            Confidence score (0-1).
        """
        relative_uncertainty = uncertainty / (np.abs(prediction) + 1e-8)
        return 1 / (1 + relative_uncertainty)


if __name__ == "__main__":
    estimator = SolarCapexEstimator()

    print("Training model...")
    trained_model = estimator.train_model()

    trained_model_fp = estimator.save_model(trained_model)
    print(f"Model trained and saved to {trained_model_fp}")

    # trained_model_fp = "./models/solar_capex_model_20260210_2231.pkl" 

    print("Loading model...")
    estimator.load_model(trained_model_fp)

    print("Model loaded successfully.")

    print("Making predictions on new data...")
    prediction_requests = json.load(open("./data/sample/prediction_requests.json", "r"))
    predictions = estimator.predict(prediction_requests)

    print("Predictions:")
    print(predictions)
