import pandas as pd
import numpy as np

class DataCleaner:
    """
    Data cleaner for LBNL Tracking the Sun dataset.

    This class handles TTS-specific data cleaning operations independent of the modeling pipeline.


    """

    def __init__(self, config_min_target_value=10, config_high_cardinality_threshold=0.05):
        """
        Initialize the DataCleaner with configuration parameters.

        Parameters
        ----------
        config_min_target_value : float, optional
            Minimum valid target value. Rows with target values below this will be removed. Default is 10.
        config_high_cardinality_threshold : float, optional
            Proportion of unique values above which a column will be dropped. Default is 0.05 (5%).
        """

        self.df = None

        self.config_min_target_value = config_min_target_value
        self.config_high_cardinality_threshold = config_high_cardinality_threshold

    def load_data(self, df):
        """
        Load data into the cleaner.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to clean.

        Returns
        -------
        None
        """
        self.df = df

    def _make_true_na(self, df):
        """
        Convert common placeholder values for missing data to true NaN.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to clean.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe with true NaN values.
        """
        df = df.replace([-1, "-1"], np.nan)
        
        return df
    
    def _coerce_datatypes(self, df):
        """
        Coerce datatypes of columns to appropriate types.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to clean.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe with coerced datatypes.
        """
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')

            elif "zip" in col.lower() or "postal" in col.lower():
                df[col] = df[col].astype(str).str.zfill(5)

            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    df[col] = df[col].astype('category')
        return df

    def _clean_by_target(self, df, target_col):
        """
        Clean the target variable by removing rows with missing or invalid values.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to clean.
        target_col : str
            Name of the target column to clean.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe.
        """
        before_count = len(df)
        df = df.dropna(subset=[target_col])
        df = df[df[target_col] >= self.config_min_target_value]
        after_count = len(df)
        print(f"> Removed {before_count - after_count} rows with missing or invalid target values.")
        return df
    
    def _drop_high_cardinality_columns(self, df):
        """
        Drop columns that have a high proportion of unique values.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to clean.
            
        Returns
        -------
        pd.DataFrame 
            Cleaned dataframe with high-cardinality columns dropped.
        """
        if len(df) == 0:
            print("Warning: Dataframe is empty. Skipping high-cardinality column drop.")
            return df
        
        unique_proportions = df.nunique() / len(df)

        cols_to_drop = unique_proportions[unique_proportions > self.config_high_cardinality_threshold].index
        cols_to_drop = [col for col in cols_to_drop if df[col].dtype in ['object', 'str']]
        print(f"> Dropping high-cardinality columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

        return df
    
    def clean(self, target_col='total_installed_price'):
        """
        Perform all cleaning steps on the loaded dataframe.

        Parameters
        ----------
        target_col : str, optional
            Target column to clean. Default is 'total_installed_price'.
        min_target_value : float, optional
            Minimum valid target value. Default is 10.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe.

        Raises
        ------
        ValueError
            If dataframe has not been loaded.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.df = self._make_true_na(self.df)
        self.df = self._clean_by_target(self.df, target_col)
        self.df = self._coerce_datatypes(self.df)
        self.df = self._drop_high_cardinality_columns(self.df)
        return self.df