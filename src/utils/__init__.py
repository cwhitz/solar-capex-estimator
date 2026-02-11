"""
Utility functions for Solar CAPEX Estimator.

This package contains helper functions for model persistence and input validation.
"""

from .input_validation import PredictionRequest, validate_prediction_requests

__all__ = [
    "PredictionRequest",
    "validate_prediction_requests",
]
