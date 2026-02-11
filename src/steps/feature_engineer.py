import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Feature engineer for LBNL Tracking the Sun dataset.

    This class handles TTS-specific feature engineering operations independent of the modeling pipeline.

    Parameters
    ----------
    None

    Attributes
    ----------
    None

    """

    def __init__(self):
        # Initialize dataframe attribute; it will be set by load_data().
        self.df = None

    def load_data(self, df):
        """
        Load data into the feature engineer.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to engineer.

        Returns
        -------
        None
        """
        self.df = df

    def _add_day_count(self, df):
        """
        Add a feature for the number of days since installation.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to engineer.

        Returns
        -------
        pd.DataFrame
            Dataframe with new 'days_since_2000' feature.
        """
        df["days_since_2000"] = (
            df["installation_date"] - pd.Timestamp("2000-01-01")
        ).dt.days

        return df

    def _combine_module_counts(self, df):
        """
        Combine module count features into a single feature.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to engineer.

        Returns
        -------
        pd.DataFrame
            Dataframe with new 'total_module_count' feature.
        """
        if "total_module_count" in df.columns:
            print(
                "Warning: 'total_module_count' already exists in dataframe. Skipping module count combination."
            )
            return df
        module_cols = [col for col in df.columns if "module_quantity" in col.lower()]
        df["total_module_count"] = df[module_cols].replace(np.nan, 0).sum(axis=1)

        if module_cols:
            df = df.drop(columns=module_cols)

        return df

    def engineer_features(self):
        """
        Perform all feature engineering steps on the loaded dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        pd.DataFrame
            Dataframe with engineered features.

        Raises
        ------
        ValueError
            If dataframe has not been loaded.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self._add_day_count(self.df)
        df = self._combine_module_counts(df)

        return df
