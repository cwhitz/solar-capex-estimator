import numpy as np
import pandas as pd


class DataCleaner:
    """
    Data cleaner for LBNL Tracking the Sun dataset.

    This class handles TTS-specific data cleaning operations independent of the modeling pipeline.


    """

    def __init__(
        self,
        min_target_value=10,
        high_cardinality_threshold=0.10,
        na_drop_thresholds={"string_columns": 0.10, "numeric_columns": 0.50},
    ):
        """
        Initialize the DataCleaner with configuration parameters.

        Parameters
        ----------
        min_target_value : float, optional
            Minimum valid target value. Rows with target values below this will be removed. Default is 10.
        high_cardinality_threshold : float, optional
            Proportion of unique values above which a column will be dropped. Default is 0.10 (10%).
        na_drop_thresholds : dict, optional
            Thresholds for dropping columns with high NA values. Default is {'string_columns': 0.10, 'numeric_columns': 0.50}.
        """

        self.df = None

        self.min_target_value = min_target_value
        self.high_cardinality_threshold = high_cardinality_threshold
        self.na_drop_thresholds = na_drop_thresholds

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
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")

            elif "zip" in col.lower() or "postal" in col.lower():
                df[col] = df[col].astype(str).str.zfill(5)
            elif df[col].dtype in ["bool"]:
                df[col] = df[col].astype(int)
            elif df[col].dtype in ["object", "str"]:
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    df[col] = df[col].astype("str")
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
        df = df[df[target_col] >= self.min_target_value]
        after_count = len(df)
        print(
            f"> Removed {before_count - after_count} rows with missing or invalid target values."
        )
        return df

    def _drop_high_na_columns(self, df):
        """
        Drop columns that have a majority of missing values based on configured thresholds.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to clean.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe with high-NA columns dropped.
        """

        if len(df) == 0:
            print("Warning: Dataframe is empty. Skipping high-NA column drop.")
            return df

        string_cols = df.select_dtypes(include=["object"]).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        string_na_proportions = df[string_cols].isna().mean()
        numeric_na_proportions = df[numeric_cols].isna().mean()

        cols_to_drop_string = string_na_proportions[
            string_na_proportions > self.na_drop_thresholds["string_columns"]
        ].index
        cols_to_drop_numeric = numeric_na_proportions[
            numeric_na_proportions > self.na_drop_thresholds["numeric_columns"]
        ].index

        cols_to_drop = list(cols_to_drop_string) + list(cols_to_drop_numeric)
        print(f"> Dropping columns with majority NA values: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

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

        cols_to_drop = unique_proportions[
            unique_proportions > self.high_cardinality_threshold
        ].index
        cols_to_drop = [
            col for col in cols_to_drop if df[col].dtype in ["object", "str"]
        ]
        print(f"> Dropping high-cardinality columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

        return df

    def _drop_id_and_provider_columns(self, df):
        """
        Drop columns that are likely to be identifiers or provider-specific information.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to clean.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe with ID and provider columns dropped.
        """
        id_cols = ["data_provider_1", "data_provider_2", "system_ID_1", "system_ID_2"]
        print(f"> Dropping ID and provider columns: {id_cols}")
        df = df.drop(columns=id_cols, errors="ignore")

        return df

    def _drop_single_value_columns(self, df):
        """
        Drop columns that have only a single unique value.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to clean.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe with single-value columns dropped.
        """
        single_value_cols = df.columns[df.nunique() <= 1]
        print(f"> Dropping single-value columns: {list(single_value_cols)}")
        df = df.drop(columns=single_value_cols)

        return df

    def clean(self, target_col="total_installed_price"):
        """
        Perform all cleaning steps on the loaded dataframe.

        Parameters
        ----------
        target_col : str, optional
            Target column to clean. Default is 'total_installed_price'.

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
        self.df = self._drop_id_and_provider_columns(self.df)
        self.df = self._drop_high_na_columns(self.df)
        self.df = self._drop_single_value_columns(self.df)
        self.df = self._drop_high_cardinality_columns(self.df)
        self.df = self._coerce_datatypes(self.df)
        return self.df
