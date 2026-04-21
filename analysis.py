"""
AI-Based Manufacturing Efficiency Classification
Complete Analysis Pipeline: EDA, Preprocessing, Feature Engineering, Model Training & Evaluation
Author: Bedanta | Unified Mentor Internship Project 2
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             f1_score, precision_score, recall_score)
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Thales_Group_Manufacturing.csv')
CHARTS_DIR = os.path.join(BASE_DIR, 'charts')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {'High': '#00C853', 'Medium': '#FFD600', 'Low': '#FF1744'}
PALETTE = ['#FF1744', '#FFD600', '#00C853']

print("=" * 70)
print("  AI-BASED MANUFACTURING EFFICIENCY CLASSIFICATION")
print("  Sensor, Production & 6G Network Data Analysis")
print("=" * 70)

# ============================================================
# STEP 1: DATA LOADING & INSPECTION
# ============================================================
print("\n📊 STEP 1: Loading & Inspecting Dataset...")
df = pd.read_csv(DATA_PATH)

print(f"   ✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   ✅ Missing values: {df.isnull().sum().sum()}")
print(f"   ✅ Duplicates: {df.duplicated().sum()}")
print(f"   ✅ Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
print(f"   ✅ Machines: {df['Machine_ID'].nunique()} unique")
print(f"   ✅ Operation Modes: {df['Operation_Mode'].unique().tolist()}")

print("\n   Target Distribution (Efficiency_Status):")
for status, count in df['Efficiency_Status'].value_counts().items():
    pct = count / len(df) * 100
    print(f"      {status}: {count:,} ({pct:.1f}%)")

# ============================================================
# STEP 2: DATETIME PROCESSING
# ============================================================
print("\n⏰ STEP 2: Processing Date & Time...")
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], format='%d-%m-%Y %H:%M:%S')
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
df = df.sort_values('Datetime').reset_index(drop=True)
print("   ✅ Datetime combined, Hour/DayOfWeek/IsWeekend extracted")

# ============================================================
# STEP 3: EDA CHARTS
# ============================================================
print("\n📈 STEP 3: Generating EDA Charts...")

# Chart 1: Target Distribution
fig, ax = plt.subplots(figsize=(8, 5))
counts = df['Efficiency_Status'].value_counts()
bars = ax.bar(counts.index, counts.values, color=[COLORS[x] for x in counts.index], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontweight='bold', fontsize=11)
ax.set_title('Efficiency Status Distribution', fontsize=14, fontweight='bold')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'target_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ target_distribution.png")

# Chart 2: Correlation Heatmap
fig, ax = plt.subplots(figsize=(12, 9))
numeric_cols = ['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW',
                'Network_Latency_ms', 'Packet_Loss_%', 'Quality_Control_Defect_Rate_%',
                'Production_Speed_units_per_hr', 'Predictive_Maintenance_Score', 'Error_Rate_%']
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ correlation_heatmap.png")

# Chart 3: Feature Distributions by Efficiency Status
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
for idx, col in enumerate(numeric_cols):
    ax = axes[idx // 3, idx % 3]
    for status in ['Low', 'Medium', 'High']:
        subset = df[df['Efficiency_Status'] == status][col]
        ax.hist(subset, bins=40, alpha=0.5, label=status, color=COLORS[status])
    ax.set_title(col.replace('_', ' '), fontsize=10, fontweight='bold')
    ax.legend(fontsize=7)
fig.suptitle('Feature Distributions by Efficiency Status', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ feature_distributions.png")

# Chart 4: Operation Mode vs Efficiency
fig, ax = plt.subplots(figsize=(10, 6))
ct = pd.crosstab(df['Operation_Mode'], df['Efficiency_Status'], normalize='index') * 100
ct[['Low', 'Medium', 'High']].plot(kind='bar', stacked=True, ax=ax,
                                     color=[COLORS['Low'], COLORS['Medium'], COLORS['High']],
                                     edgecolor='white')
ax.set_title('Efficiency Distribution by Operation Mode', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentage (%)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Efficiency')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'operation_mode_efficiency.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ operation_mode_efficiency.png")

# Chart 5: Hourly Efficiency Pattern
fig, ax = plt.subplots(figsize=(12, 5))
hourly = pd.crosstab(df['Hour'], df['Efficiency_Status'], normalize='index') * 100
hourly[['High', 'Medium']].plot(ax=ax, marker='o', linewidth=2)
ax.set_title('Hourly Efficiency Trend (High & Medium %)', fontsize=14, fontweight='bold')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Percentage (%)')
ax.set_xticks(range(0, 24))
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'hourly_efficiency.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ hourly_efficiency.png")

# Chart 6: Machine-wise Efficiency
fig, ax = plt.subplots(figsize=(16, 6))
machine_eff = pd.crosstab(df['Machine_ID'], df['Efficiency_Status'], normalize='index') * 100
machine_eff = machine_eff.sort_values('High', ascending=False)
machine_eff[['Low', 'Medium', 'High']].plot(kind='bar', stacked=True, ax=ax,
    color=[COLORS['Low'], COLORS['Medium'], COLORS['High']], edgecolor='white', width=0.8)
ax.set_title('Efficiency Profile by Machine', fontsize=14, fontweight='bold')
ax.set_ylabel('Percentage (%)')
ax.set_xlabel('Machine ID')
ax.legend(title='Efficiency')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'machine_efficiency.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ machine_efficiency.png")

# Chart 7: Box plots for key features
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
box_features = ['Error_Rate_%', 'Production_Speed_units_per_hr', 'Predictive_Maintenance_Score',
                'Network_Latency_ms', 'Packet_Loss_%', 'Quality_Control_Defect_Rate_%']
for idx, col in enumerate(box_features):
    ax = axes[idx // 3, idx % 3]
    data_to_plot = [df[df['Efficiency_Status'] == s][col].values for s in ['Low', 'Medium', 'High']]
    bp = ax.boxplot(data_to_plot, labels=['Low', 'Medium', 'High'], patch_artist=True)
    for patch, color in zip(bp['boxes'], [COLORS['Low'], COLORS['Medium'], COLORS['High']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(col.replace('_', ' '), fontsize=10, fontweight='bold')
fig.suptitle('Feature Distributions by Efficiency Class', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'boxplots_by_efficiency.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ boxplots_by_efficiency.png")

# ============================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================
print("\n🔧 STEP 4: Feature Engineering...")

# Energy Efficiency Ratio: Production output per unit of power consumed
df['Energy_Efficiency_Ratio'] = df['Production_Speed_units_per_hr'] / (df['Power_Consumption_kW'] + 0.001)

# Error-to-Output Ratio: Errors relative to production speed
df['Error_Output_Ratio'] = df['Error_Rate_%'] / (df['Production_Speed_units_per_hr'] + 0.001)

# Network Reliability Score: Inverse of latency and packet loss combined
df['Network_Reliability'] = 1 / (1 + df['Network_Latency_ms'] / 50 + df['Packet_Loss_%'] / 5)

# Sensor Stability Index: Combined normalized sensor readings
df['Sensor_Stability'] = (
    (1 - abs(df['Temperature_C'] - df['Temperature_C'].median()) / df['Temperature_C'].std()) *
    (1 - abs(df['Vibration_Hz'] - df['Vibration_Hz'].median()) / df['Vibration_Hz'].std())
)

# Quality-Production Score
df['Quality_Production_Score'] = df['Production_Speed_units_per_hr'] * (1 - df['Quality_Control_Defect_Rate_%'] / 100)

# Maintenance-Error Interaction
df['Maintenance_Error_Score'] = df['Predictive_Maintenance_Score'] * (1 - df['Error_Rate_%'] / 15)

# Power-Vibration Ratio
df['Power_Vibration_Ratio'] = df['Power_Consumption_kW'] / (df['Vibration_Hz'] + 0.001)

# Overall Health Score
df['Machine_Health_Score'] = (
    df['Predictive_Maintenance_Score'] * 0.3 +
    (1 - df['Error_Rate_%'] / 15) * 0.25 +
    (1 - df['Quality_Control_Defect_Rate_%'] / 10) * 0.2 +
    df['Network_Reliability'] * 0.15 +
    (df['Production_Speed_units_per_hr'] / 500) * 0.1
)

engineered_features = ['Energy_Efficiency_Ratio', 'Error_Output_Ratio', 'Network_Reliability',
                       'Sensor_Stability', 'Quality_Production_Score', 'Maintenance_Error_Score',
                       'Power_Vibration_Ratio', 'Machine_Health_Score']
print(f"   ✅ Created {len(engineered_features)} engineered features:")
for f in engineered_features:
    print(f"      • {f}")

# ============================================================
# STEP 5: PREPROCESSING
# ============================================================
print("\n⚙️ STEP 5: Preprocessing...")

# Encode Operation_Mode
le_mode = LabelEncoder()
df['Operation_Mode_Encoded'] = le_mode.fit_transform(df['Operation_Mode'])

# Encode Target
le_target = LabelEncoder()
df['Efficiency_Encoded'] = le_target.fit_transform(df['Efficiency_Status'])
class_names = le_target.classes_
print(f"   ✅ Classes: {list(class_names)}")
print(f"   ✅ Encoding: {dict(zip(class_names, le_target.transform(class_names)))}")

# Feature columns
feature_cols = numeric_cols + engineered_features + ['Operation_Mode_Encoded', 'Hour', 'DayOfWeek', 'IsWeekend']
X = df[feature_cols].copy()
y = df['Efficiency_Encoded'].copy()

# Handle any infinities
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(X.median(), inplace=True)

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

# Train-Test Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✅ Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f"   ✅ Features: {len(feature_cols)}")

# Save preprocessors
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
joblib.dump(le_target, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
joblib.dump(le_mode, os.path.join(MODELS_DIR, 'mode_encoder.pkl'))
joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_columns.pkl'))
print("   ✅ Saved: scaler.pkl, label_encoder.pkl, mode_encoder.pkl, feature_columns.pkl")

# ============================================================
# STEP 6: MODEL TRAINING
# ============================================================
print("\n🤖 STEP 6: Training Models...")

# Compute class weights for imbalance
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print(f"   Class weights: {class_weight_dict}")

results = {}

# Model 1: Logistic Regression (Baseline)
print("\n   🔹 Training Logistic Regression (Baseline)...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred, average='weighted')
results['Logistic Regression'] = {'accuracy': lr_acc, 'f1': lr_f1, 'model': lr, 'preds': lr_pred}
print(f"      Accuracy: {lr_acc:.4f} | F1: {lr_f1:.4f}")

# Model 2: Random Forest
print("\n   🔹 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced',
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
results['Random Forest'] = {'accuracy': rf_acc, 'f1': rf_f1, 'model': rf, 'preds': rf_pred}
print(f"      Accuracy: {rf_acc:.4f} | F1: {rf_f1:.4f}")

# Model 3: XGBoost
print("\n   🔹 Training XGBoost...")
# Calculate sample weights for XGBoost
sample_weights = np.array([class_weight_dict[c] for c in y_train])
xgb = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                     use_label_encoder=False, eval_metric='mlogloss',
                     random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train, sample_weight=sample_weights)
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
results['XGBoost'] = {'accuracy': xgb_acc, 'f1': xgb_f1, 'model': xgb, 'preds': xgb_pred}
print(f"      Accuracy: {xgb_acc:.4f} | F1: {xgb_f1:.4f}")

# Model 4: Gradient Boosting
print("\n   🔹 Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred, average='weighted')
results['Gradient Boosting'] = {'accuracy': gb_acc, 'f1': gb_f1, 'model': gb, 'preds': gb_pred}
print(f"      Accuracy: {gb_acc:.4f} | F1: {gb_f1:.4f}")

# ============================================================
# STEP 7: MODEL COMPARISON
# ============================================================
print("\n📊 STEP 7: Model Comparison...")

# Find best model
best_name = max(results, key=lambda k: results[k]['f1'])
best_model = results[best_name]['model']
best_preds = results[best_name]['preds']
print(f"\n   🏆 Best Model: {best_name}")
print(f"      Accuracy: {results[best_name]['accuracy']:.4f}")
print(f"      F1 Score: {results[best_name]['f1']:.4f}")

# Save best model
joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
joblib.dump(best_name, os.path.join(MODELS_DIR, 'best_model_name.pkl'))
print(f"   ✅ Best model saved: best_model.pkl")

# Save all models
for name, data in results.items():
    fname = name.lower().replace(' ', '_') + '.pkl'
    joblib.dump(data['model'], os.path.join(MODELS_DIR, fname))

# Chart: Model Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
f1_scores = [results[m]['f1'] for m in model_names]

colors_bar = ['#5C6BC0', '#26A69A', '#FF7043', '#AB47BC']
axes[0].barh(model_names, accuracies, color=colors_bar, edgecolor='white', height=0.5)
axes[0].set_xlim(0, 1)
axes[0].set_title('Model Accuracy', fontsize=13, fontweight='bold')
for i, v in enumerate(accuracies):
    axes[0].text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

axes[1].barh(model_names, f1_scores, color=colors_bar, edgecolor='white', height=0.5)
axes[1].set_xlim(0, 1)
axes[1].set_title('Model F1 Score (Weighted)', fontsize=13, fontweight='bold')
for i, v in enumerate(f1_scores):
    axes[1].text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

plt.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ model_comparison.png")

# Chart: Confusion Matrix for best model
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, best_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
            yticklabels=class_names, ax=ax, linewidths=0.5)
ax.set_title(f'Confusion Matrix — {best_name}', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ confusion_matrix.png")

# Classification Report
print(f"\n   Classification Report ({best_name}):")
print(classification_report(y_test, best_preds, target_names=class_names))

# ============================================================
# STEP 8: FEATURE IMPORTANCE & EXPLAINABILITY
# ============================================================
print("\n🔍 STEP 8: Feature Importance & Explainability...")

# Get feature importances from best tree-based model
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
else:
    # Use Random Forest importances as fallback
    importances = rf.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=True)

# Save feature importance data
importance_df.to_csv(os.path.join(MODELS_DIR, 'feature_importance.csv'), index=False)

# Chart: Feature Importance
fig, ax = plt.subplots(figsize=(10, 8))
colors_imp = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importance_df)))
ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors_imp, edgecolor='white')
ax.set_title(f'Feature Importance — {best_name}', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ feature_importance.png")

# Top 10 features
print("\n   Top 10 Most Important Features:")
for _, row in importance_df.tail(10).iloc[::-1].iterrows():
    print(f"      • {row['Feature']}: {row['Importance']:.4f}")

# Chart: Top 10 Feature Importance
fig, ax = plt.subplots(figsize=(10, 6))
top10 = importance_df.tail(10)
ax.barh(top10['Feature'], top10['Importance'],
        color=plt.cm.viridis(np.linspace(0.3, 0.9, 10)), edgecolor='white')
ax.set_title('Top 10 Features Driving Efficiency Classification', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, 'top10_features.png'), dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ top10_features.png")

# ============================================================
# STEP 9: CROSS-VALIDATION
# ============================================================
print("\n🔄 STEP 9: Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_scaled, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
print(f"   5-Fold CV F1 Scores: {cv_scores.round(4)}")
print(f"   Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Save CV results
cv_results = {'cv_scores': cv_scores.tolist(), 'mean': cv_scores.mean(), 'std': cv_scores.std()}
joblib.dump(cv_results, os.path.join(MODELS_DIR, 'cv_results.pkl'))

# ============================================================
# STEP 10: SAVE RESULTS SUMMARY
# ============================================================
print("\n💾 STEP 10: Saving Results...")

model_summary = {
    'models': {name: {'accuracy': data['accuracy'], 'f1': data['f1']} for name, data in results.items()},
    'best_model': best_name,
    'best_accuracy': results[best_name]['accuracy'],
    'best_f1': results[best_name]['f1'],
    'cv_mean_f1': cv_scores.mean(),
    'n_features': len(feature_cols),
    'n_samples': len(df),
    'class_distribution': df['Efficiency_Status'].value_counts().to_dict()
}
joblib.dump(model_summary, os.path.join(MODELS_DIR, 'model_summary.pkl'))

print("\n" + "=" * 70)
print("  ✅ ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\n  📁 Charts saved to: {CHARTS_DIR}")
print(f"  📁 Models saved to: {MODELS_DIR}")
print(f"\n  🏆 Best Model: {best_name}")
print(f"     Accuracy: {results[best_name]['accuracy']:.4f}")
print(f"     F1 Score: {results[best_name]['f1']:.4f}")
print(f"     CV Mean F1: {cv_scores.mean():.4f}")
print(f"\n  📊 Charts generated: {len(os.listdir(CHARTS_DIR))}")
print(f"  🤖 Models saved: {len(os.listdir(MODELS_DIR))}")
print("\n  Run the dashboard: streamlit run app.py")
print("=" * 70)
