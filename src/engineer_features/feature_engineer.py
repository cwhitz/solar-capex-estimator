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
        pass

    def add_day_count(self, df):
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
        df['days_since_2000'] = (df['installation_date'] - pd.Timestamp('2000-01-01')).dt.days
        
        return df
    
    def combine_module_counts(self, df):
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
        module_cols = [col for col in df.columns if 'module_quantity' in col.lower()]
        df['total_module_count'] = df[module_cols].replace(np.nan, 0).sum(axis=1)

        df = df.drop(columns=module_cols)
        
        return df