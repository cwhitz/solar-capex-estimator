# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Solar capital expenditure estimator - a machine learning project for estimating solar project costs.

## Code Style

- **Docstring format**: NumPy style
- Main class: `SolarCapexEstimator` in `main.py`

## Project Structure

```
solar-capex-estimator/
├── main.py           # Main module with SolarCapexEstimator class
├── src/              # Source code for utilities and components
├── notebooks/        # Jupyter notebooks for model development and experimentation
├── data/             # Dataset storage
└── models/           # Trained model storage
```

## Development Workflow

This project is set up for:
- Model development and experimentation in notebooks
- Lightweight model deployment via the main SolarCapexEstimator class
- Separation of research (notebooks) from production code (src/)
