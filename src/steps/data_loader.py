from pathlib import Path
from typing import List, Optional

import pandas as pd


class DataLoader:
    """
    Data loader for LBNL Tracking the Sun dataset.

    This class handles TTS-specific data loading, cleaning, and filtering
    operations independent of the modeling pipeline.

    Parameters
    ----------
    tts_data_directory : str
        Path to the directory containing the raw TTS data files.

    Attributes
    ----------
    tts_data_directory : Path
        Directory containing the raw TTS data files.
    df : pd.DataFrame
        Loaded dataframe.
    valid_customer_segments : list
        List of valid customer segments.

    """

    def __init__(self, tts_data_directory: str):
        """Initialize the DataLoader."""
        self.tts_data_directory = Path(tts_data_directory)
        self.df = None

        self.valid_customer_segments = [
            "COM",
            "RES_MF",
            "RES_SF",
            "RES",
            "AGRICULTURAL",
            "OTHER TAX-EXEMPT",
            "GOV",
            "SCHOOL",
            "NON-RES",
            "NON-PROFIT",
        ]

    def _filter_by_years(self, df, year_min=None, year_max=None):
        """
        Filter data to specific installation years.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to filter.
        year_min : int, optional
            Minimum installation year to include. If None, includes all years.
        year_max : int, optional
            Maximum installation year to include. If None, includes all years.

        Returns
        -------
        pd.DataFrame
            Filtered dataframe.

        Raises
        ------
        ValueError
            If dataframe has not been loaded.
        """
        if df is None:
            raise ValueError("Data not loaded. Call load_raw() first.")

        if year_min is not None:
            df = df[df.installation_date.dt.year >= year_min]
        if year_max is not None:
            df = df[df.installation_date.dt.year <= year_max]

        return df

    def _filter_by_customer_segment(self, df, segments):
        """
        Filter data to specific customer segments.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to filter.
        segments : list of str
            Customer segments to filter to. If None, includes all segments.

        Returns
        -------
        pd.DataFrame
            Filtered dataframe.

        Raises
        ------
        ValueError
            If dataframe has not been loaded.
        """
        if df is None:
            raise ValueError("Data not loaded. Call load_raw() first.")

        df = df[df["customer_segment"].isin(segments)]

        return df

    def _validate_filters(self, year_min, year_max, customer_segments):
        """
        Validate filter parameters.

        Parameters
        ----------
        year_min : int, optional
            Minimum installation year to include. If None, includes all years.
        year_max : int, optional
            Maximum installation year to include. If None, includes all years.
        customer_segments : list of str, optional
            Customer segments to include (e.g., ['COM', 'NON-RES']).

        Raises
        ------
        ValueError
            If year_min is greater than year_max or if customer_segments is not a list of strings.
        """
        if year_min is not None and year_max is not None and year_min > year_max:
            raise ValueError("year_min cannot be greater than year_max.")

        if customer_segments is not None:
            if not isinstance(customer_segments, list) or not all(
                isinstance(seg, str) for seg in customer_segments
            ):
                raise ValueError("customer_segments must be a list of strings.")
            if not set(customer_segments).issubset(set(self.valid_customer_segments)):
                raise ValueError(
                    f"customer_segments must be a subset of {self.valid_customer_segments}."
                )

    def load_training_data(
        self,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        customer_segments: Optional[List[str]] = None,
    ):
        """
        Load and filter TTS data with common preprocessing steps.

        Parameters
        ----------
        year_min : int, optional
            Minimum year to filter to. If None, includes all years.
        year_max : int, optional
            Maximum year to filter to. If None, includes all years.
        customer_segments : list of str, optional
            Customer segments to filter to. If None, includes all segments.

        Returns
        -------
        pd.DataFrame
            Filtered and cleaned dataframe.
        """

        csvs = list(self.tts_data_directory.glob("*.csv"))

        self._validate_filters(year_min, year_max, customer_segments)

        if csvs:
            self.df = pd.DataFrame()
            for csv in csvs:
                csv_df = pd.read_csv(
                    csv, parse_dates=["installation_date"], low_memory=False
                )

                if year_min is not None or year_max is not None:
                    csv_df = self._filter_by_years(csv_df, year_min, year_max)

                if customer_segments is not None:
                    csv_df = self._filter_by_customer_segment(csv_df, customer_segments)

                self.df = pd.concat([self.df, csv_df], ignore_index=True)
                print(f"Loaded {len(csv_df)} rows from {csv.name}")

        else:
            raise ValueError(
                f"No CSV files found in directory {self.tts_data_directory}"
            )

    def get_data(self):
        """
        Get the current dataframe.

        Returns
        -------
        pd.DataFrame
            Current dataframe.

        Raises
        ------
        ValueError
            If dataframe has not been loaded.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_training_data() first.")

        return self.df
