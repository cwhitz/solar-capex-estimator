import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineer for LBNL Tracking the Sun dataset.

    Scikit-learn compatible transformer.
    """

    def __init__(self):
        # no parameters yet, but required for sklearn compatibility
        pass

    def fit(self, X, y=None):
        """
        Fit does nothing because this transformer is stateless.
        Required for sklearn pipelines.
        """
        return self

    def transform(self, X):
        """
        Apply feature engineering steps.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        df = X.copy()

        df = self._add_day_count(df)
        df = self._combine_module_counts(df)

        return df

    def _add_day_count(self, df):
        if 'installation_date' in df.columns:
            if len(df) > 0:
                df['days_since_2000'] = (
                    df['installation_date'] - pd.Timestamp('2000-01-01')
                ).dt.days
            else:
                # Create empty column for empty dataframe
                df['days_since_2000'] = pd.Series(dtype='float64')
        return df

    def _combine_module_counts(self, df):
        if 'total_module_count' in df.columns:
            return df

        module_cols = [
            col for col in df.columns
            if 'module_quantity' in col.lower()
        ]

        if module_cols:
            df['total_module_count'] = (
                df[module_cols].fillna(0).sum(axis=1)
            )
            df = df.drop(columns=module_cols)

        return df