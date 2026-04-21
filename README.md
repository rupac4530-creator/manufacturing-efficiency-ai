# 🏭 AI-Based Manufacturing Efficiency Classification

### Using Sensor, Production, and 6G Network Data

**Unified Mentor Internship — Project 2** | Author: Bedanta

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![Gemini](https://img.shields.io/badge/Google_Gemini-AI_Insights-4285F4?logo=google)](https://ai.google.dev)

---

## 🎯 Overview

An end-to-end AI system that classifies manufacturing efficiency as **High**, **Medium**, or **Low** using real-time sensor, production, and 6G network data from 50 industrial machines.

> **Best Model:** Random Forest — **99.99% Accuracy** | **99.99% F1 Score**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                                 │
│                                                                  │
│  📊 Raw Data          🔧 Preprocessing        🧠 ML Models      │
│  ┌──────────┐        ┌──────────────┐        ┌──────────────┐   │
│  │ 100K     │───────▶│ Cleaning     │───────▶│ Logistic Reg │   │
│  │ Records  │        │ Encoding     │        │ Random Forest│   │
│  │ 50 Mach. │        │ Scaling      │        │ XGBoost      │   │
│  │ 14 Feat. │        │ 8 Engineered │        │ Gradient GB  │   │
│  └──────────┘        └──────────────┘        └──────┬───────┘   │
│                                                      │           │
│  ┌───────────────────────────────────────────────────▼────────┐  │
│  │                 STREAMLIT DASHBOARD                         │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │  │
│  │  │Overview  │ │Predict   │ │Machine   │ │Explainability│  │  │
│  │  │Health    │ │Real-time │ │Insights  │ │SHAP/Feature  │  │  │
│  │  │Anomaly   │ │Classify  │ │Per-Mach. │ │Importance    │  │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │  │
│  │  ┌──────────┐ ┌──────────────────────────────────────────┐ │  │
│  │  │Network & │ │ ✦ AI Insights (Google Gemini)            │ │  │
│  │  │Sensors   │ │   Executive Summary | Recommendations    │ │  │
│  │  └──────────┘ └──────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

```bash
# Clone
git clone https://github.com/rupac4530-creator/manufacturing-efficiency-ai.git
cd manufacturing-efficiency-ai

# Option A: One command
chmod +x run.sh && ./run.sh

# Option B: Manual
pip install -r requirements.txt
streamlit run app.py --server.port 8502
```

Open **http://localhost:8502** in your browser.

---

## 📊 Results

| Model | Accuracy | F1 Score | Training Time |
|-------|----------|----------|---------------|
| Logistic Regression | 88.67% | 89.30% | ~2s |
| **Random Forest** 🏆 | **99.99%** | **99.99%** | ~15s |
| XGBoost | 99.82% | 99.83% | ~8s |
| Gradient Boosting | 99.98% | 99.98% | ~45s |

**Cross-Validation:** 5-fold CV = 0.9999 ± 0.0000

### Top Feature Importance

| Feature | Importance | Interpretation |
|---------|------------|----------------|
| Error_Rate_% | 32.7% | Primary driver of efficiency |
| Error_Output_Ratio | 26.2% | Engineered ratio — errors vs speed |
| Production_Speed | 17.8% | Higher speed → higher efficiency |
| Quality_Defect_Rate | 8.4% | Defects directly reduce efficiency |
| Energy_Efficiency | 5.1% | Power consumption per unit output |

---

## 🏭 Dashboard Features (6 Tabs)

| Tab | Features |
|-----|----------|
| **◈ Overview** | KPI cards, Factory Health Score (gauge), Anomaly Detection, Business Impact, Download Reports |
| **◉ Predictions** | Real-time efficiency classification with confidence scores |
| **⬡ Machine Insights** | Per-machine analysis, performance scatter, drill-down |
| **◎ Explainability** | Feature importance, model comparison, confusion matrix |
| **◇ Network & Sensors** | Latency analysis, sensor correlations, heatmaps |
| **✦ AI Insights** | Gemini-powered executive summaries, recommendations, Q&A |

---

## 📁 Project Structure

```
manufacturing-efficiency-ai/
├── analysis.py              # Full ML pipeline (EDA → Training → Evaluation)
├── app.py                   # Streamlit dashboard (6 tabs, Gemini AI)
├── run.sh                   # One-command setup & launch
├── requirements.txt         # Python dependencies
├── .env                     # Gemini API key (not in repo)
├── .gitignore               # Excludes .env, venv, cache
├── README.md                # This file
├── RESEARCH_PAPER.md         # Academic research paper
├── EXECUTIVE_SUMMARY.md      # Stakeholder executive summary
├── MODEL_CARD.md             # Model documentation & limitations
├── Thales_Group_Manufacturing.csv  # Dataset (100K rows)
├── charts/                   # 11 EDA & analysis visualizations
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── ... (6 more)
└── models/                   # 13 trained model files
    ├── best_model.pkl        # Random Forest (winner)
    ├── scaler.pkl
    ├── label_encoder.pkl
    ├── feature_importance.csv
    └── ... (9 more)
```

---

## 🔑 Key Findings

1. **Error Rate is the #1 efficiency driver** (32.7% importance) — reducing error rates directly improves efficiency
2. **Network quality (6G) has minimal impact** — infrastructure is already reliable
3. **Random Forest dominates** — near-perfect classification with robust generalization
4. **Class separation is genuine** — verified via independent Decision Tree analysis
5. **Anomaly detection identifies risky machines** — enables proactive maintenance

---

## 🛡️ Security

- API keys loaded from `.env` file (excluded from Git)
- No secrets hardcoded in source code
- Gemini integration is optional — core ML works independently

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| ML | Scikit-learn, XGBoost |
| Dashboard | Streamlit, Plotly |
| AI Assistant | Google Gemini 2.0 Flash |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |

---

## 📝 License

This project was developed as part of the Unified Mentor Data Science Internship.

**Author:** Bedanta | **Domain:** Thales Group — Smart Manufacturing & Industrial IoT
