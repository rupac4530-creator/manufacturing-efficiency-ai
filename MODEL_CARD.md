# Model Card — Manufacturing Efficiency Classifier

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | Random Forest Classifier |
| **Version** | 1.0 |
| **Type** | Multi-class Classification |
| **Framework** | Scikit-learn 1.8.0 |
| **Author** | Bedanta — Unified Mentor Internship |
| **Date** | April 2026 |

## Intended Use

- **Primary Use:** Classify manufacturing processes as High, Medium, or Low efficiency in real-time
- **Target Users:** Factory managers, production engineers, quality control teams
- **Deployment:** Streamlit web dashboard with interactive predictions

## Training Data

| Attribute | Value |
|-----------|-------|
| **Dataset** | Thales Group Manufacturing Sensor Data |
| **Records** | 100,000 |
| **Machines** | 50 industrial machines |
| **Period** | January — March 2025 |
| **Features** | 14 raw + 8 engineered = 22 total |
| **Target** | Efficiency_Status (High / Medium / Low) |
| **Class Distribution** | High: 3.0%, Medium: 19.2%, Low: 77.8% |
| **Split** | 80% train / 20% test (stratified) |

## Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.99% |
| **F1 Score (Weighted)** | 99.99% |
| **Precision (Weighted)** | 99.99% |
| **Recall (Weighted)** | 99.99% |
| **Cross-Validation (5-fold)** | 0.9999 ± 0.0000 |

### Why Is Accuracy So High?

This is **not data leakage**. Verification shows:
- A simple Decision Tree (depth=5) on raw features alone achieves 100% F1
- The target classes are **genuinely well-separated** by Error_Rate:
  - High efficiency: Error_Rate ≈ 1.01% (std: 0.58)
  - Medium efficiency: Error_Rate ≈ 2.73% (std: 1.40)
  - Low efficiency: Error_Rate ≈ 8.93% (std: 3.79)
- The dataset was synthetically designed with clear decision boundaries

## Feature Importance (Top 5)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Error_Rate_% | 32.7% |
| 2 | Error_Output_Ratio | 26.2% |
| 3 | Production_Speed_units_per_hr | 17.8% |
| 4 | Quality_Control_Defect_Rate_% | 8.4% |
| 5 | Energy_Efficiency_Ratio | 5.1% |

## Limitations

1. **Synthetic data:** Trained on structured simulated data — real factory data may have more noise
2. **50 machines only:** May not generalize to factories with very different machine types
3. **Static model:** Does not adapt to concept drift over time without retraining
4. **Class imbalance:** Only 3% of records are High efficiency — model may be biased toward Low
5. **No time-series modeling:** Treats each record independently, ignoring temporal dependencies

## Ethical Considerations

- Model decisions should **supplement**, not replace, human judgment
- Should not be used for employee performance evaluation
- Predictions should be reviewed before triggering automated maintenance actions
- Regular retraining recommended as factory conditions change

## How to Use

```python
import joblib
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
X_scaled = scaler.transform(new_data)
predictions = model.predict(X_scaled)
```
