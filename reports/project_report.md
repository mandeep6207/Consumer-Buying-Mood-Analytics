# ShopShift AI Project Report

## Objective
Predict consumer buying mood from shopping behavior and surface behavioral signals tied to impulsive, luxury, budget-conscious, window-shopping, and emotional patterns.

## Best Model
- **XGBoost**
- Test Accuracy: 0.9333
- Weighted F1: 0.9333

## Model Comparison
- Logistic Regression: accuracy=0.9271, precision=0.9281, recall=0.9271, f1=0.9270, cv=0.9219
- Random Forest: accuracy=0.9250, precision=0.9260, recall=0.9250, f1=0.9249, cv=0.9167
- XGBoost: accuracy=0.9333, precision=0.9349, recall=0.9333, f1=0.9333, cv=0.9146

## Behavioral Insights
- Late night activity and impulsive clicks separate emotional and impulsive buyers.
- Income and spending are the strongest luxury-oriented signals.
- Discount usage and lower purchase frequency identify budget-conscious behavior.
- Browsing pressure and low cart conversion shape the window shopper class.

## Outputs
- Visuals exported to `visuals/`
- Model artifact exported to `models/`
- Reports exported to `reports/` and `metrics/`