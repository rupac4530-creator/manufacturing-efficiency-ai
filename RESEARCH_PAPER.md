# Research Paper: AI-Based Manufacturing Efficiency Classification Using Sensor, Production, and 6G Network Data

**Author:** Bedanta  
**Organization:** Unified Mentor — Data Science Internship  
**Domain:** Thales Group — Smart Manufacturing & Industrial IoT  
**Date:** 2025

---

## Abstract

This research presents an AI-driven system for real-time classification of manufacturing efficiency into three categories — High, Medium, and Low — using sensor readings, production metrics, and 6G network data from industrial IoT environments. The study analyzes 100,000 operational records from 50 machines across multiple operation modes. Through comprehensive feature engineering and multi-model comparison, we demonstrate that a Random Forest classifier achieves 99.99% accuracy with robust cross-validation performance, enabling automated and interpretable efficiency assessment for smart factories.

---

## 1. Introduction

### 1.1 Background

Modern smart factories rely on Industrial IoT sensors and high-speed 6G connectivity to monitor production in real time. However, traditional dashboards only show historical data — they cannot proactively classify the current efficiency state. Manufacturing teams face:

- **Delayed detection** of efficiency degradation
- **Manual interpretation** of dozens of metrics simultaneously
- **Lack of automated, interpretable** efficiency assessment

### 1.2 Objective

This project develops an AI-based classification system that:
1. Automatically classifies manufacturing efficiency as High, Medium, or Low
2. Identifies the key drivers of efficiency through feature importance analysis
3. Provides real-time predictions with confidence scores
4. Delivers insights through an interactive dashboard

---

## 2. Dataset Description

### 2.1 Overview

| Attribute | Value |
|-----------|-------|
| Total Records | 100,000 |
| Features | 14 (including target) |
| Time Period | January 1 – March 10, 2025 |
| Machines | 50 unique industrial machines |
| Missing Values | 0 |
| Duplicates | 0 |

### 2.2 Feature Categories

**Machine Data:** Machine_ID, Operation_Mode (Active/Idle/Maintenance)

**Sensor Data:** Temperature (30-90°C), Vibration (0-5 Hz), Power Consumption (1-10 kW)

**6G Network Data:** Network Latency (1-50 ms), Packet Loss (0-5%)

**Quality Metrics:** Defect Rate (0-10%), Error Rate (0-15%), Predictive Maintenance Score (0-1)

**Production:** Production Speed (50-500 units/hr)

**Target:** Efficiency_Status — High (2.99%), Medium (19.19%), Low (77.82%)

### 2.3 Key Observation: Class Imbalance

The target variable shows significant imbalance with Low efficiency dominating at 77.82%. This was addressed using balanced class weights during model training to ensure the minority classes (High and Medium) are properly learned.

---

## 3. Methodology

### 3.1 Data Preprocessing

1. **Datetime Processing** — Combined Date and Timestamp columns; extracted Hour, DayOfWeek, and IsWeekend features
2. **Categorical Encoding** — Label encoded Operation_Mode (Active=0, Idle=1, Maintenance=2)
3. **Feature Scaling** — StandardScaler applied to all numeric features
4. **Data Splitting** — 80/20 stratified train-test split

### 3.2 Feature Engineering

Eight domain-specific features were engineered:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| Energy Efficiency Ratio | Production Speed / Power Consumption | Output per unit energy |
| Error-Output Ratio | Error Rate / Production Speed | Error intensity relative to output |
| Network Reliability | 1 / (1 + Latency/50 + PacketLoss/5) | Combined network quality score |
| Sensor Stability | Normalized temp × vibration deviation | Sensor reading consistency |
| Quality-Production Score | Speed × (1 - DefectRate/100) | Defect-adjusted output |
| Maintenance-Error Score | Maintenance × (1 - ErrorRate/15) | Maintenance vs error interaction |
| Power-Vibration Ratio | Power / Vibration | Energy per vibration unit |
| Machine Health Score | Weighted composite of 5 metrics | Overall machine condition |

### 3.3 Model Selection

Four models were trained and compared:

1. **Logistic Regression** — Baseline linear model with balanced class weights
2. **Random Forest** — 200 estimators, max_depth=20, balanced weights
3. **XGBoost** — 200 estimators, max_depth=8, learning_rate=0.1
4. **Gradient Boosting** — 150 estimators, max_depth=6, learning_rate=0.1

---

## 4. Results

### 4.1 Model Performance Comparison

| Model | Accuracy | F1 Score (Weighted) |
|-------|----------|-------------------|
| Logistic Regression | 88.67% | 89.30% |
| Random Forest | **99.99%** | **99.99%** |
| XGBoost | 99.82% | 99.83% |
| Gradient Boosting | 99.98% | 99.98% |

### 4.2 Best Model: Random Forest

The Random Forest classifier achieved near-perfect classification:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| High | 1.00 | 1.00 | 1.00 | 597 |
| Low | 1.00 | 1.00 | 1.00 | 15,565 |
| Medium | 1.00 | 1.00 | 1.00 | 3,838 |

**5-Fold Cross-Validation:** Mean F1 = 0.9999 ± 0.0000

### 4.3 Feature Importance Analysis

The top 5 features driving efficiency classification:

1. **Error_Rate_%** — 32.73% importance
2. **Error_Output_Ratio** — 26.22% importance
3. **Production_Speed_units_per_hr** — 17.84% importance
4. **Quality_Production_Score** — 13.30% importance
5. **Energy_Efficiency_Ratio** — 3.65% importance

**Key Insight:** Error-related metrics and production speed are the dominant factors. Network metrics (latency, packet loss) have minimal direct impact on efficiency classification, suggesting the 6G infrastructure provides consistent connectivity.

---

## 5. Exploratory Data Analysis Findings

### 5.1 Operation Mode Impact
- All three modes (Active, Idle, Maintenance) show similar efficiency distributions
- No single mode disproportionately causes low efficiency

### 5.2 Temporal Patterns
- Efficiency distribution remains consistent across hours and days
- No significant weekend vs weekday differences observed

### 5.3 Machine Variability
- All 50 machines show similar efficiency profiles
- No single machine is a consistent outlier

---

## 6. Dashboard Implementation

A Streamlit web application was built with five interactive modules:

1. **Overview Dashboard** — Real-time KPI metrics, efficiency pie charts, hourly trends
2. **Prediction Engine** — Slider-based input for real-time classification with confidence scores
3. **Machine Insights** — Per-machine efficiency profiles, scatter performance maps
4. **Explainability Panel** — Feature importance charts, model comparison, confusion matrix
5. **Network & Sensor Analysis** — Latency/production correlations, sensor heatmaps

---

## 7. Conclusions

1. **High Classification Accuracy** — The Random Forest model achieves 99.99% accuracy, making it suitable for production deployment
2. **Error Rate is the Key Driver** — Error_Rate_% alone accounts for 32.7% of the classification decision
3. **Engineered Features Add Value** — Error_Output_Ratio (26.2%) proves more discriminative than many raw features
4. **Network Impact is Minimal** — 6G connectivity metrics have low feature importance, indicating reliable infrastructure
5. **Class Imbalance Handled** — Despite 77.8% Low class dominance, balanced class weights ensure all classes are accurately predicted

---

## 8. Recommendations

1. **Deploy the Random Forest model** for real-time efficiency monitoring
2. **Prioritize error rate reduction** as the primary lever for improving efficiency
3. **Monitor production speed** alongside error rates for early warning detection
4. **Implement automated alerts** when predicted efficiency drops to Low
5. **Conduct root cause analysis** on the Error_Output_Ratio metric for process optimization

---

## 9. Future Work

- Implement time-series forecasting for proactive efficiency prediction
- Add SHAP-based individual prediction explanations
- Integrate with real factory SCADA systems for live monitoring
- Develop anomaly detection for sensor drift identification

---

## References

1. Thales Group — Smart Manufacturing Division
2. Unified Mentor — Data Science Internship Program
3. scikit-learn Documentation — https://scikit-learn.org
4. XGBoost Documentation — https://xgboost.readthedocs.io
5. Streamlit Documentation — https://docs.streamlit.io
