# Solar CAPEX Estimator - Codebase Architecture

## Overview

This codebase implements a production-ready machine learning system for estimating commercial solar installation costs (CAPEX) using the LBNL Tracking the Sun dataset. The design emphasizes **composition over inheritance** and **configuration-driven development** to create a modular, testable pipeline where each component has a single, well-defined responsibility.

## Design Philosophy: Composition & Configuration

### Composable Pipeline Components

Rather than building a monolithic estimator class, we decompose the ML pipeline into five independent, reusable components:

1. **`DataLoader`** - Handles TTS-specific CSV loading, date parsing, and filtering.
2. **`DataCleaner`** - Manages data quality: sentinel values, missing data, cardinality reduction.
3. **`FeatureEngineer`** - BaseEstimator subclass that creates new features (date-based, aggregations) and selects the final feature set to be put into the pipeline.
4. **`Preprocessor`** - BaseEstimator subclass that encodes categoricals, and applies transformations to prepare data for modeling.
5. **`RFRTrainer`** - Builds a pipeline with the previous two components and orchestrates hyperparameter search and model training for a Random Forest Regressor.

This composition approach brings several benefits:
- **Modularity**: Each component can be developed, tested, and improved independently.
- **Reusability**: Components can be mixed and matched for different modeling tasks.
- **Testability**: Unit tests focus on single responsibilities (74 tests across 5 modules).
- **Debuggability**: Pipeline failures localize to specific components.
- **Extensibility**: Add new steps (e.g., outlier detection) without refactoring existing code.
- **Flexibility**: Swap implementations (e.g., different feature engineering strategies) without affecting other steps.

## Modeling Decisions

### Feature Selection: From 29 to 5 Features

Post-training feature importance analysis revealed that the size of the PV system (in DC kW) overwhelmingly dominates cost predictions, accounting for 86% of the model's predictive power. This insight led to a strategic decision to minimize the feature set for production deployment, retaining only the most impactful features while dropping those with minimal importance or high missingness.

**Retained Features:**
1. `PV_system_size_DC` (86% importance) - Dominant cost driver
2. `state` - Labor costs, permitting, incentives
3. `utility_service_territory` - Rates, grid infrastructure
4. `total_module_count` - Complexity beyond size
5. `installation_date` - Temporal pricing trends

**Dropped Features:**
- Equipment details (module/inverter models): 80%+ missing values
- High-cardinality IDs (zip_code, installer_name): Overfitting risk

This minimization brings production benefits:
- **Simpler data collection**: 5 fields vs 29
- **Faster inference**: Fewer transformations
- **Reduced surface area**: Fewer validation/encoding failures
- **Maintained performance**: Validation MAE within 2% of full model

### Algorithm Choice: Random Forest

Compared three algorithms via 5-fold cross-validation:
- **Random Forest**: $176k MAE (selected)
- **Linear Regression**: $186k MAE (surprisingly competitive)
- **K-Nearest Neighbors**: $231k MAE (poor performance)

Random Forest won due to:
- Best accuracy with acceptable interpretability (feature importances)
- Robust to feature scale, missing values, outliers
- Natural handling of nonlinear relationships and interactions
- No assumptions about data distribution

### Validation Strategy: Temporal Hold-Out

To ensure the model generalizes to future installations:
1. **Training**: 2019-2022 data (8,296 installations)
2. **Validation**: Held-out 2023 data (2,109 installations, never seen during training or tuning) with the caveat that 2023 data may have distribution shifts (e.g., supply chain issues, policy changes) that could affect performance.
3. **No leakage**: Strict temporal separation prevents optimistic metrics

This simulates real production usage: train on historical data, predict on new installations.

## Implementation: `SolarCapexEstimator`

The production estimator class (`src/main.py`) orchestrates these components while exposing a clean API:

### Input Validation with Pydantic

The `EstimationRequest` dataclass provides type-safe, validated inputs:
- **Type coercion**: Strings → floats, dates, enums
- **Range validation**: `0 < system_size_kw < 10_000`, `module_count < 50_000`
- **Required fields**: No silent failures from missing data
- **Clear errors**: "system_size_kw must be positive" vs generic exceptions

This prevents the common production failure mode where invalid inputs silently propagate through the pipeline.

### Prediction Outputs with Uncertainty

Each prediction returns structured data:
```python
{
    'request': {...},  # Echo inputs for traceability
    'prediction': 425000.0,  # Point estimate (dollars)
    'uncertainty': 52000.0,  # Std dev across Random Forest trees
    'confidence': 0.89  # 1 / (1 + relative_uncertainty)
}
```

The uncertainty estimate (tree variance) provides a proxy for epistemic uncertainty, allowing downstream systems to flag low-confidence predictions for human review.

## Key Limitations & Production Roadmap

### Current Limitations

1. **Commercial-only scope**: Model trained exclusively on `customer_segment='COM'`; unsuitable for residential. Other categories (industrial, non-profit) may also differ significantly in amount of data which may change the model performance and feature importance.
2. **Outlier performance**: Struggles with installations >$2M (heavy residual tails in validation).
3. **No geographic granularity**: State-level only; zip-code dropped due to high cardinality and likelihood of seeing new zip codes in production.
4. **Static model**: Requires manual retraining as market conditions change

## Production Improvements

**Short-term (engineering)**:
- **API layer**: FastAPI wrapper with `/predict` and `/health` endpoints
- **Logging**: Structured logs for predictions, validation failures, model versions
- **Monitoring**: Track prediction distributions, error rates, feature drift
- **Docker**: Containerized deployment with pinned dependencies (`requirements.txt`)

**Medium-term (modeling)**:
- **Automated retraining**: Monthly pipeline triggered by new TTS data releases
- **Prediction intervals**: Bootstrap-based confidence intervals for better uncertainty quantification
- **Cost breakdown**: Multi-output model predicting equipment/labor/permitting separately
- **Explainability**: SHAP values for individual predictions ("system size drove 82% of estimate")

**Long-term (architecture)**:
- **Ensemble models**: Combine Random Forest + Gradient Boosting for improved accuracy
- **Online learning**: Incremental updates as new installations arrive
- **Geographic expansion**: Expand beyond commercial to residential/industrial segments
- **Feature store**: Centralize feature engineering for consistency across models

---