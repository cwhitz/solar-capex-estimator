import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from typing import Optional


class TTSPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for our Capex solar cost estimation model.
    sklearn-compatible transformer.
    """

    def __init__(self):
        self.column_dict = None
        self.preprocessor = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Determine column types and fit the preprocessing pipeline.
        """
        df = X.copy()

        self._sort_columns(df=X, target_col=y.name if y is not None else None)
        self._build_preprocessor()

        # Fit underlying column transformer
        if self.preprocessor is not None:
            if y is not None:
                self.preprocessor.fit(df, y)
            else:
                self.preprocessor.fit(df)

        return self

    def transform(self, X: pd.DataFrame):
        """
        Apply preprocessing.
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        return self.preprocessor.transform(X)

    def get_feature_names(self):
        """Get feature names after preprocessing."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not built. Must be fit to data first.")
        return self.preprocessor.get_feature_names_out()

    def _sort_columns(self, df: pd.DataFrame, target_col: Optional[str]):
        """Sort columns into types.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to analyze column types from.
        target_col : str, optional
            Name of the target column. If None, no column will be treated as target.
        """
        self.column_dict = dict(
            target_col=target_col,
            num_cols=[],
            binary_cols=[],
            cat_low_card_cols=[],
            cat_high_card_cols=[],
        )

        for col in df.columns:
            if col == self.column_dict["target_col"]:
                continue
            elif df[col].dtype in ["int64", "float64"]:
                self.column_dict["num_cols"].append(col)
            elif (
                df[col].dtype == "bool"
                or (df[col].dtype == "object" and df[col].nunique() == 2)
            ):
                self.column_dict["binary_cols"].append(col)
            elif df[col].dtype in ["str", "object", "category"]:
                if df[col].nunique() < 10:
                    self.column_dict["cat_low_card_cols"].append(col)
                else:
                    self.column_dict["cat_high_card_cols"].append(col)

    def _build_preprocessor(self):
        """Build the column transformer in four parts: low-cardinality categorical, high-cardinality categorical, binary, and numeric."""

        low_card_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        ).set_output(transform="pandas")

        high_card_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", TargetEncoder()),
            ]
        ).set_output(transform="pandas")

        binary_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        drop="if_binary",
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        ).set_output(transform="pandas")

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        ).set_output(transform="pandas")

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat_low_card", low_card_pipeline, self.column_dict["cat_low_card_cols"]),
                ("cat_high_card", high_card_pipeline, self.column_dict["cat_high_card_cols"]),
                ("binary", binary_pipeline, self.column_dict["binary_cols"]),
                ("num", num_pipeline, self.column_dict["num_cols"]),
            ],
            remainder="drop",
        ).set_output(transform="pandas")
