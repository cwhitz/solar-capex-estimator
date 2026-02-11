from sklearn.base import BaseEstimator, TransformerMixin


class FeatureReducer(BaseEstimator, TransformerMixin):
    """
    Transformer that selects a subset of columns.
    """

    def __init__(self, features_to_keep=None):
        self.features_to_keep = features_to_keep

    def fit(self, X, y=None):
        if self.features_to_keep is None:
            raise ValueError("features_to_keep must be provided.")

        missing = [col for col in self.features_to_keep if col not in X.columns]
        if missing:
            raise ValueError(f"Columns not found in input data: {missing}")

        return self

    def transform(self, X):
        return X[self.features_to_keep].copy()
