"""
Input validation for Solar CAPEX Estimator using Pydantic.

This module provides validation models for prediction requests to ensure
data quality and type safety before making predictions.
"""

from datetime import datetime
from typing import List, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """
    Validation model for a single prediction request.

    Attributes
    ----------
    PV_system_size_DC : float
        PV system size in kilowatts (DC). Must be positive.
    state : str
        US state code (e.g., 'CA', 'TX'). 2-letter uppercase code.
    utility_service_territory : str
        Name of the utility service territory.
    total_module_count : int
        Total number of solar modules. Must be positive integer.
    installation_date : Union[str, datetime]
        Installation date. Can be string (YYYY-MM-DD) or datetime object.

    Examples
    --------
    >>> request = PredictionRequest(
    ...     PV_system_size_DC=100.5,
    ...     state='CA',
    ...     utility_service_territory='Pacific Gas & Electric Co',
    ...     total_module_count=300,
    ...     installation_date='2023-06-15'
    ... )
    """

    PV_system_size_DC: float = Field(
        gt=0, description="PV system size in kilowatts (DC). Must be positive."
    )

    state: str = Field(
        min_length=2,
        max_length=2,
        description="US state code (2-letter uppercase, e.g., 'CA', 'TX')",
    )

    utility_service_territory: str = Field(
        min_length=1, description="Name of the utility service territory"
    )

    total_module_count: int = Field(
        gt=0, description="Total number of solar modules. Must be positive integer."
    )

    installation_date: Union[str, datetime] = Field(
        description="Installation date (YYYY-MM-DD format or datetime object)"
    )

    @field_validator("state")
    @classmethod
    def validate_state_uppercase(cls, v: str) -> str:
        """Ensure state code is uppercase."""
        return v.upper()

    @field_validator("installation_date")
    @classmethod
    def validate_installation_date(cls, v: Union[str, datetime]) -> datetime:
        """Convert string dates to datetime objects."""
        if isinstance(v, str):
            try:
                return pd.to_datetime(v)
            except Exception as e:
                raise ValueError(
                    f"Invalid date format. Expected YYYY-MM-DD, got: {v}"
                ) from e
        return v

    @field_validator("PV_system_size_DC")
    @classmethod
    def validate_reasonable_system_size(cls, v: float) -> float:
        """Warn if system size is unreasonably large."""
        if v > 10000:  # 10 MW - extremely large for commercial
            raise ValueError(f"System size {v} kW is unusually large. Please verify.")
        return v

    @field_validator("total_module_count")
    @classmethod
    def validate_reasonable_module_count(cls, v: int) -> int:
        """Warn if module count is unreasonably large."""
        if v > 50000:  # Very large installation
            raise ValueError(f"Module count {v} is unusually large. Please verify.")
        return v


def validate_prediction_requests(requests: List[dict]) -> List[PredictionRequest]:
    """
    Validate a list of prediction requests.

    Parameters
    ----------
    requests : List[dict]
        List of prediction request dictionaries.

    Returns
    -------
    List[PredictionRequest]
        List of validated PredictionRequest objects.

    Raises
    ------
    ValueError
        If any request fails validation with details about which request failed.

    Examples
    --------
    >>> requests = [
    ...     {
    ...         'PV_system_size_DC': 100.5,
    ...         'state': 'CA',
    ...         'utility_service_territory': 'PG&E',
    ...         'total_module_count': 300,
    ...         'installation_date': '2023-06-15'
    ...     }
    ... ]
    >>> validated = validate_prediction_requests(requests)
    """
    validated_requests = []

    for i, request in enumerate(requests):
        try:
            validated = PredictionRequest(**request)
            validated_requests.append(validated)
        except Exception as e:
            raise ValueError(f"Validation failed for request {i}: {str(e)}") from e

    return validated_requests
