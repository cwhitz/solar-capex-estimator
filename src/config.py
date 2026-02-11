config = {
    "loading": {"year_min": 2019, "year_max": 2022, "customer_segments": ["COM"]},
    "cleaning": {
        "min_target_value": 10,
        "high_cardinality_threshold": 0.1,
        "na_drop_thresholds": {"string_columns": 0.1, "numeric_columns": 0.5},
    },
    "hyperparameter_search": {
        "param_grid": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [9, 12],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        }
    },
    "validation_loading": {
        "year_min": 2023,
        "year_max": 2023,
        "customer_segments": ["COM"],
    },
    "model_features": {
        "features": [
            "PV_system_size_DC",
            "state",
            "utility_service_territory",
            "total_module_count",
            "installation_date",
        ],
        "target": "total_installed_price",
    },
}
