# 🏭 AI-Based Manufacturing Efficiency Classification

## Using Sensor, Production, and 6G Network Data

**Unified Mentor Internship — Project 2**

An end-to-end machine learning system that classifies manufacturing efficiency as **High**, **Medium**, or **Low** using sensor readings, production metrics, and 6G network data from smart factories.

---

## 📊 Project Overview

| Item | Details |
|------|---------|
| **Dataset** | 100,000 records, 14 features |
| **Task** | Multi-class classification (High / Medium / Low) |
| **Best Model** | Random Forest — 99.99% accuracy |
| **Features** | 21 (9 original + 8 engineered + 4 temporal) |
| **Dashboard** | Streamlit with 5 interactive tabs |

---

## 🔧 Tech Stack

- **Python** — pandas, NumPy, scikit-learn, XGBoost
- **Visualization** — matplotlib, seaborn, Plotly
- **Dashboard** — Streamlit
- **ML Models** — Logistic Regression, Random Forest, XGBoost, Gradient Boosting

---

## 📁 Project Structure

```
├── analysis.py                  # Full ML pipeline (EDA → Training → Evaluation)
├── app.py                       # Streamlit dashboard
├── requirements.txt             # Dependencies
├── Thales_Group_Manufacturing.csv  # Dataset
├── charts/                      # Generated EDA charts
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── ...
├── models/                      # Trained models & preprocessors
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── ...
├── RESEARCH_PAPER.md
└── README.md
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis pipeline
python analysis.py

# Launch dashboard
streamlit run app.py
```

---

## 📈 Feature Engineering

| Feature | Description |
|---------|-------------|
| Energy_Efficiency_Ratio | Production output per unit of power consumed |
| Error_Output_Ratio | Errors relative to production speed |
| Network_Reliability | Combined latency and packet loss score |
| Sensor_Stability | Temperature and vibration deviation index |
| Quality_Production_Score | Output adjusted for defect rate |
| Maintenance_Error_Score | Maintenance readiness vs error interaction |
| Machine_Health_Score | Composite health indicator |

---

## 🏆 Model Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 88.67% | 89.30% |
| Random Forest | **99.99%** | **99.99%** |
| XGBoost | 99.82% | 99.83% |
| Gradient Boosting | 99.98% | 99.98% |

---

## 🔍 Top Feature Drivers

1. **Error_Rate_%** (32.7%)
2. **Error_Output_Ratio** (26.2%)
3. **Production_Speed** (17.8%)
4. **Quality_Production_Score** (13.3%)
5. **Energy_Efficiency_Ratio** (3.7%)

---

## 📊 Dashboard Features

- **Overview** — KPI metrics, efficiency distribution, hourly patterns
- **Predictions** — Real-time efficiency classification with confidence scores
- **Machine Insights** — Per-machine efficiency profiles and performance maps
- **Explainability** — Feature importance, model comparison, confusion matrix
- **Network & Sensors** — Latency/packet loss analysis, sensor correlations

---

## 👤 Author

**Bedanta** — Unified Mentor Data Science Intern

---

*Built with ❤️ using Python, scikit-learn, and Streamlit*
