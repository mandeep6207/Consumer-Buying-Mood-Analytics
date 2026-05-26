# ShopShift AI

Consumer Buying Mood Analytics System

> A lightweight consumer intelligence project that synthesizes shopping behavior, engineers behavioral signals, trains mood classifiers, and exports professional analytics artifacts.

## What it does

- Generates a realistic consumer shopping behavior dataset when no source data is available.
- Engineers behavioral features such as impulse score, shopping intensity, browsing pressure, and emotional purchase index.
- Compares Logistic Regression, Random Forest, and XGBoost for multi-class buying mood prediction.
- Produces EDA visuals, evaluation reports, model artifacts, and a notebook-ready analysis flow.

## Project Structure

- `data/` synthetic dataset and intermediate data files
- `notebooks/` analysis notebook
- `models/` trained model artifact
- `visuals/` generated charts
- `reports/` markdown and JSON reports
- `metrics/` text classification report
- `src/` reusable pipeline code

## Workflow

1. Data generation and cleaning
2. Missing value handling
3. Feature engineering
4. Exploratory analysis
5. Visualization export
6. Model training and comparison
7. Cross validation and evaluation
8. Report generation
9. Notebook generation

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the pipeline

```bash
python run_pipeline.py
```

The pipeline will create the dataset, train all models, export the best model, and write outputs to the artifact folders.

## Outputs

- `models/buying_mood_predictor.pkl`
- `reports/model_metrics.json`
- `reports/project_report.md`
- `metrics/classification_report.txt`
- `notebooks/shopshift_analysis.ipynb`
- `visuals/*.png`

## Behavioral Analytics Angle

ShopShift AI treats buying mood as a behavioral state inferred from shopping motion, attention, price sensitivity, and time-of-day habits. The system is designed to surface patterns such as impulsive bursts, luxury-oriented spend, discount dependence, and window-shopping behavior.

## Model Comparison

The project evaluates three supervised classifiers on the same leakage-safe preprocessing pipeline:

- Logistic Regression for a compact linear baseline
- Random Forest for non-linear interaction capture
- XGBoost for boosted decision boundary learning

## Results

The synthetic dataset is intentionally noisy but structured so that realistic models land in a strong high-accuracy band without perfect scores.

## Future Scope

- Integrate a real transactional dataset
- Add time-series session features
- Deploy as a dashboard or API
- Add SHAP-based explanations
- Monitor drift across customer segments
