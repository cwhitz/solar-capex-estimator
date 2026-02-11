"""Preprocessing module for solar CAPEX estimation model."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder


class Preprocessor:
    """
    Preprocessor for our Capex solar cost estimation model.
    """

    def __init__(self, target_col="total_installed_price"):
        """Initialize the Preprocessor with configuration parameters."""
        self.target_col = target_col
        self.columns = None

        self.preprocessor = None

    def _sort_columns(self, df: pd.DataFrame):
        """Sort columns into numerical, binary, low-cardinality categorical, and high-cardinality categorical based on datatypes and unique value counts.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to use for determining column types.
        """
        self.columns = dict(
            target_col=self.target_col,
            num_cols=[],
            binary_cols=[],
            cat_low_card_cols=[],
            cat_high_card_cols=[],
        )

        for col in df.columns:
            if col == self.columns["target_col"]:
                continue
            elif df[col].dtype in ["int64", "float64"]:
                self.columns["num_cols"].append(col)
            elif (df[col].dtype == "bool") or (df[col].dtype == "object" and df[col].nunique() == 2):
                self.columns["binary_cols"].append(col)
            elif df[col].dtype in ["str", "object", "category"]:
                if df[col].nunique() < 10:
                    self.columns["cat_low_card_cols"].append(col)
                else:
                    self.columns["cat_high_card_cols"].append(col)

    def get_feature_names(self):
        """Get feature names after preprocessing"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not built. Must be fit to data first.")
        return self.preprocessor.get_feature_names_out()

    def build_preprocessor(self, df):
        """
        Build the preprocessing pipeline based on the dataframe's columns and datatypes.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to use for determining column types and building the preprocessor.

        Returns
        -------
        ColumnTransformer
            The built preprocessing pipeline.
        """
        self._sort_columns(df)

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
            steps=[("imputer", SimpleImputer(strategy="most_frequent"))]
        ).set_output(transform="pandas")

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        ).set_output(transform="pandas")

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat_low_card", low_card_pipeline, self.columns["cat_low_card_cols"]),
                (
                    "cat_high_card",
                    high_card_pipeline,
                    self.columns["cat_high_card_cols"],
                ),
                ("binary", binary_pipeline, self.columns["binary_cols"]),
                ("num", num_pipeline, self.columns["num_cols"]),
            ],
            remainder="drop",
        ).set_output(transform="pandas")

        return self.preprocessor
