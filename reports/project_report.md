# ShopShift AI Project Report

## Objective
Predict consumer buying mood from shopping behavior and surface behavioral signals tied to impulsive, luxury, budget-conscious, window-shopping, and emotional patterns.

## Best Model
- **XGBoost**
- Test Accuracy: 0.9299
- Weighted F1: 0.9299

## Model Comparison
- XGBoost: accuracy=0.9299, precision=0.9307, recall=0.9299, f1=0.9299, cv=0.9177, gap=0.0000
- Logistic Regression: accuracy=0.9280, precision=0.9290, recall=0.9280, f1=0.9280, cv=0.9156, gap=0.0019
- Random Forest: accuracy=0.9242, precision=0.9255, recall=0.9242, f1=0.9242, cv=0.9156, gap=0.0057

## Behavioral Insights
- Late night activity and impulsive clicks separate emotional and impulsive buyers.
- Income and spending are the strongest luxury-oriented signals.
- Discount usage and lower purchase frequency identify budget-conscious behavior.
- Browsing pressure and low cart conversion shape the window shopper class.

## Outputs
- Visuals exported to `visuals/`
- Model artifact exported to `models/`
- Reports exported to `reports/` and `metrics/`