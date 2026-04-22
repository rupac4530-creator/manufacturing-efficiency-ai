"""
AI-Based Manufacturing Efficiency Classification Dashboard
Streamlit Web Application - Premium Design with Gemini AI Integration
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, os
from dotenv import load_dotenv

load_dotenv()

def get_ai_response(prompt):
    """Multi-provider AI with automatic fallback and 12s timeout per call."""
    import httpx
    providers = [
        {"type": "nvidia", "key_env": "NVIDIA_API_KEY_1", "model": "deepseek-ai/deepseek-v3.2", "name": "DeepSeek V3.2"},
        {"type": "nvidia", "key_env": "NVIDIA_API_KEY_2", "model": "deepseek-ai/deepseek-v3.1-terminus", "name": "DeepSeek V3.1"},
        {"type": "nvidia", "key_env": "NVIDIA_API_KEY_3", "model": "openai/gpt-oss-20b", "name": "GPT-OSS-20B"},
        {"type": "nvidia", "key_env": "NVIDIA_API_KEY_4", "model": "openai/gpt-oss-120b", "name": "GPT-OSS-120B"},
        {"type": "gemini", "key_env": "GEMINI_API_KEY", "model": "gemini-2.0-flash", "name": "Google Gemini"},
    ]
    for p in providers:
        try:
            api_key = os.getenv(p["key_env"])
            if not api_key:
                continue
            if p["type"] == "gemini":
                from google import genai
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(model=p["model"], contents=prompt)
                if response and response.text:
                    return response.text
            else:
                from openai import OpenAI
                http_client = httpx.Client(timeout=httpx.Timeout(12.0, connect=5.0))
                client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key, http_client=http_client)
                completion = client.chat.completions.create(
                    model=p["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7, top_p=0.9, max_tokens=2048
                )
                if completion.choices and completion.choices[0].message.content:
                    return completion.choices[0].message.content
        except Exception:
            continue
    return None

def generate_local_summary(fdf):
    """Generate a data-driven executive summary from the data itself — no API needed."""
    total = len(fdf)
    high_n = len(fdf[fdf['Efficiency_Status']=='High'])
    med_n = len(fdf[fdf['Efficiency_Status']=='Medium'])
    low_n = len(fdf[fdf['Efficiency_Status']=='Low'])
    avg_err = fdf['Error_Rate_%'].mean()
    avg_spd = fdf['Production_Speed_units_per_hr'].mean()
    top_machine = fdf.groupby('Machine_ID')['Error_Rate_%'].mean().idxmax()
    top_err = fdf.groupby('Machine_ID')['Error_Rate_%'].mean().max()
    return f"""**Executive Summary — Manufacturing Efficiency Analysis**

📊 **Dataset:** {total:,} production records across {fdf['Machine_ID'].nunique()} industrial machines.

📈 **Efficiency Breakdown:** {low_n:,} records ({low_n/total*100:.1f}%) are **Low efficiency**, {med_n:,} ({med_n/total*100:.1f}%) are **Medium**, and only {high_n:,} ({high_n/total*100:.1f}%) are **High** — indicating significant room for improvement.

⚠️ **Key Risk:** The average error rate is **{avg_err:.1f}%**, which is the #1 driver of efficiency classification (32.7% feature importance). Machine **{top_machine}** has the highest average error rate at **{top_err:.1f}%**.

⚡ **Production:** Average output is **{avg_spd:.0f} units/hr**. Machines with error rates below 2% consistently achieve High efficiency status.

🎯 **Recommendation:** Prioritize error rate reduction on the top 5 highest-error machines. Even a 2-3% reduction in error rate can shift machines from Low to Medium efficiency, yielding significant cost savings.

🤖 **Model Performance:** The Random Forest classifier achieves **99.99% accuracy** (5-fold cross-validated), confirming Error Rate as the dominant predictive feature."""

def generate_local_recommendations(fdf):
    """Generate data-driven recommendations from actual data patterns — no API needed."""
    low_machines = fdf[fdf['Efficiency_Status']=='Low'].groupby('Machine_ID').size().sort_values(ascending=False).head(5)
    high_err = fdf.groupby('Machine_ID')['Error_Rate_%'].mean().sort_values(ascending=False).head(3)
    avg_err = fdf['Error_Rate_%'].mean()
    avg_def = fdf['Quality_Control_Defect_Rate_%'].mean()
    recs = f"""**Improvement Recommendations — Data-Driven Action Plan**

**1. 🔴 Reduce Error Rates (Highest Impact — 32.7% importance)**
Focus on these top error-producing machines: {', '.join([f'Machine {m} ({e:.1f}%)' for m, e in high_err.items()])}. Target: bring error rates below 3% to shift from Low to Medium efficiency.

**2. 🟡 Address Low-Efficiency Machines (77.8% of production)**
These machines have the most Low-efficiency incidents: {', '.join([f'Machine {m} ({c} incidents)' for m, c in low_machines.items()])}. Schedule maintenance reviews for each.

**3. 🟢 Optimize Production Speed**
Current average: {fdf['Production_Speed_units_per_hr'].mean():.0f} units/hr. High-efficiency machines average {fdf[fdf['Efficiency_Status']=='High']['Production_Speed_units_per_hr'].mean():.0f} units/hr. Benchmark all machines against this target.

**4. 🔧 Implement Predictive Maintenance**
Average maintenance score: {fdf['Predictive_Maintenance_Score'].mean():.0f}/100. Machines below 60 should receive immediate attention to prevent degradation.

**5. 📉 Reduce Defect Rate**
Current average defect rate: {avg_def:.1f}%. Quality-Production Score analysis shows defects directly reduce throughput. Target: below 2% defect rate for all machines."""
    return recs

# Backward compatibility alias
get_gemini_response = get_ai_response

st.set_page_config(page_title="Manufacturing Efficiency AI", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

# ── Fonts ──
st.markdown('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap"><link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons+Round">', unsafe_allow_html=True)

# ── CSS ──
st.markdown("""<style>
* { font-family: 'Outfit', sans-serif !important; }
.main { background: linear-gradient(160deg, #080b1a 0%, #0d1330 40%, #111b3c 70%, #0a1025 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #060918 0%, #0d1330 50%, #111b3c 100%); border-right: 1px solid rgba(99, 132, 255, 0.08); }
@keyframes fadeSlideIn { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: translateY(0); } }
@keyframes glowPulse { 0%, 100% { box-shadow: 0 0 15px rgba(99,132,255,0.05); } 50% { box-shadow: 0 0 25px rgba(99,132,255,0.12); } }
@keyframes shimmer { 0% { background-position: -200% center; } 100% { background-position: 200% center; } }
.metric-card { background: linear-gradient(145deg, rgba(15,20,50,0.95), rgba(10,15,40,0.95)); border: 1px solid rgba(99,132,255,0.1); border-radius: 20px; padding: 28px 20px; text-align: center; backdrop-filter: blur(20px); animation: fadeSlideIn 0.6s ease-out both, glowPulse 4s ease-in-out infinite; transition: all 0.35s cubic-bezier(0.4,0,0.2,1); position: relative; overflow: hidden; }
.metric-card:hover { transform: translateY(-4px); border-color: rgba(99,132,255,0.25); box-shadow: 0 12px 40px rgba(99,132,255,0.1); }
.metric-value { font-size: 2.4rem; font-weight: 900; letter-spacing: -1px; background: linear-gradient(135deg, #6384ff, #00d4ff, #a78bfa); background-size: 200% auto; animation: shimmer 3s linear infinite; -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-label { color: rgba(148,163,200,0.8); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2.5px; margin-top: 8px; font-weight: 600; }
.status-high { background: linear-gradient(135deg, #00C853, #69F0AE); color: #000; padding: 8px 24px; border-radius: 25px; font-weight: 700; display: inline-block; font-size: 1.1rem; box-shadow: 0 4px 15px rgba(0,200,83,0.3); }
.status-medium { background: linear-gradient(135deg, #FFD600, #FFF176); color: #000; padding: 8px 24px; border-radius: 25px; font-weight: 700; display: inline-block; font-size: 1.1rem; box-shadow: 0 4px 15px rgba(255,214,0,0.3); }
.status-low { background: linear-gradient(135deg, #FF1744, #FF5252); color: #fff; padding: 8px 24px; border-radius: 25px; font-weight: 700; display: inline-block; font-size: 1.1rem; box-shadow: 0 4px 15px rgba(255,23,68,0.3); }
.header-title { font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #6384ff, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0; letter-spacing: -0.5px; }
.header-sub { color: rgba(148,163,200,0.7); font-size: 0.85rem; margin-top: 2px; font-weight: 400; letter-spacing: 0.5px; }
h1, h2, h3 { color: #e2e8f5 !important; font-weight: 700 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: rgba(10,15,40,0.5); padding: 6px; border-radius: 16px; border: 1px solid rgba(99,132,255,0.06); }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 12px; color: rgba(148,163,200,0.7); padding: 10px 22px; font-weight: 500; transition: all 0.3s ease; }
.stTabs [data-baseweb="tab"]:hover { color: #e2e8f5; background: rgba(99,132,255,0.08); }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #4f46e5, #6384ff) !important; color: white !important; font-weight: 600 !important; box-shadow: 0 4px 15px rgba(79,70,229,0.35); }
[data-testid="stSidebar"] p { color: rgba(200,210,230,0.85) !important; }
[data-testid="stSidebar"] strong { color: #6384ff !important; }
.stButton > button { background: linear-gradient(135deg, #4f46e5, #6384ff) !important; color: white !important; border: none !important; border-radius: 14px !important; padding: 12px 32px !important; font-weight: 600 !important; transition: all 0.3s ease !important; box-shadow: 0 4px 15px rgba(79,70,229,0.3); }
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(79,70,229,0.4) !important; }
[data-testid="stMetric"] { background: rgba(15,20,50,0.6); border: 1px solid rgba(99,132,255,0.08); border-radius: 14px; padding: 16px !important; }
[data-testid="stMetricLabel"] { color: rgba(148,163,200,0.7) !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 1px; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #e2e8f5 !important; font-weight: 700 !important; }
hr { border-color: rgba(99,132,255,0.08) !important; }
</style>""", unsafe_allow_html=True)

BASE = os.path.dirname(os.path.abspath(__file__))
COLORS = {'High': '#00C853', 'Medium': '#FFD600', 'Low': '#FF1744'}
TEMPLATE = "plotly_dark"

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE, 'Thales_Group_Manufacturing.csv'))
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], format='%d-%m-%Y %H:%M:%S')
    df['Hour'] = df['Datetime'].dt.hour
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['Energy_Efficiency_Ratio'] = df['Production_Speed_units_per_hr'] / (df['Power_Consumption_kW'] + 0.001)
    df['Error_Output_Ratio'] = df['Error_Rate_%'] / (df['Production_Speed_units_per_hr'] + 0.001)
    df['Network_Reliability'] = 1 / (1 + df['Network_Latency_ms']/50 + df['Packet_Loss_%']/5)
    df['Sensor_Stability'] = (1 - abs(df['Temperature_C'] - df['Temperature_C'].median()) / df['Temperature_C'].std()) * (1 - abs(df['Vibration_Hz'] - df['Vibration_Hz'].median()) / df['Vibration_Hz'].std())
    df['Quality_Production_Score'] = df['Production_Speed_units_per_hr'] * (1 - df['Quality_Control_Defect_Rate_%']/100)
    df['Maintenance_Error_Score'] = df['Predictive_Maintenance_Score'] * (1 - df['Error_Rate_%']/15)
    df['Power_Vibration_Ratio'] = df['Power_Consumption_kW'] / (df['Vibration_Hz'] + 0.001)
    df['Machine_Health_Score'] = df['Predictive_Maintenance_Score']*0.3 + (1 - df['Error_Rate_%']/15)*0.25 + (1 - df['Quality_Control_Defect_Rate_%']/10)*0.2 + df['Network_Reliability']*0.15 + (df['Production_Speed_units_per_hr']/500)*0.1
    return df

@st.cache_resource
def load_models():
    m = {}
    for f in ['best_model', 'scaler', 'label_encoder', 'mode_encoder', 'feature_columns', 'model_summary', 'feature_importance', 'cv_results']:
        path = os.path.join(BASE, 'models', f'{f}.pkl')
        if os.path.exists(path): m[f] = joblib.load(path)
        elif f == 'feature_importance':
            csv_path = os.path.join(BASE, 'models', 'feature_importance.csv')
            if os.path.exists(csv_path): m[f] = pd.read_csv(csv_path)
    return m

df = load_data()
models = load_models()

# ── Sidebar ──
st.sidebar.markdown("<p class='header-title'><span class='material-icons-round' style='vertical-align:middle;margin-right:8px;font-size:1.6rem;'>precision_manufacturing</span>Smart Factory AI</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='header-sub'>Manufacturing Efficiency Intelligence</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

machines = st.sidebar.multiselect("🔧 Select Machines", sorted(df['Machine_ID'].unique()), default=sorted(df['Machine_ID'].unique())[:10])
modes = st.sidebar.multiselect("⚙️ Operation Mode", df['Operation_Mode'].unique().tolist(), default=df['Operation_Mode'].unique().tolist())
dates = df['Datetime'].dt.date.unique()
date_range = st.sidebar.select_slider("📅 Date Range", options=sorted(dates), value=(sorted(dates)[0], sorted(dates)[-1]))

mask = (df['Machine_ID'].isin(machines)) & (df['Operation_Mode'].isin(modes)) & (df['Datetime'].dt.date >= date_range[0]) & (df['Datetime'].dt.date <= date_range[1])
fdf = df[mask]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered:** {len(fdf):,} / {len(df):,} records")
if 'model_summary' in models:
    ms = models['model_summary']
    st.sidebar.markdown(f"**Best Model:** {ms.get('best_model','N/A')}")
    st.sidebar.markdown(f"**Accuracy:** {ms.get('best_accuracy',0):.4f}")
    st.sidebar.markdown(f"**F1 Score:** {ms.get('best_f1',0):.4f}")

# ── Tabs ──
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["◈ Overview", "◉ Predictions", "⬡ Machine Insights", "◎ Explainability", "◇ Network & Sensors", "✦ AI Insights"])

# ── TAB 1: Overview ──
with tab1:
    st.markdown("## Manufacturing Efficiency Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    total = len(fdf)
    high_n = len(fdf[fdf['Efficiency_Status']=='High'])
    med_n = len(fdf[fdf['Efficiency_Status']=='Medium'])
    low_n = len(fdf[fdf['Efficiency_Status']=='Low'])
    with c1: st.markdown(f"<div class='metric-card'><div class='metric-value'>{total:,}</div><div class='metric-label'>Total Records</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card'><div class='metric-value' style='background:linear-gradient(135deg,#00C853,#69F0AE);-webkit-background-clip:text;'>{high_n:,}</div><div class='metric-label'>High Efficiency</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><div class='metric-value' style='background:linear-gradient(135deg,#FFD600,#FFF176);-webkit-background-clip:text;'>{med_n:,}</div><div class='metric-label'>Medium Efficiency</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-card'><div class='metric-value' style='background:linear-gradient(135deg,#FF1744,#FF5252);-webkit-background-clip:text;'>{low_n:,}</div><div class='metric-label'>Low Efficiency</div></div>", unsafe_allow_html=True)
    with c5: st.markdown(f"<div class='metric-card'><div class='metric-value'>{fdf['Production_Speed_units_per_hr'].mean():.0f}</div><div class='metric-label'>Avg Production</div></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(fdf, names='Efficiency_Status', color='Efficiency_Status', color_discrete_map=COLORS, title='Efficiency Distribution', hole=0.5, template=TEMPLATE)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        mode_eff = fdf.groupby(['Operation_Mode','Efficiency_Status']).size().reset_index(name='Count')
        fig = px.bar(mode_eff, x='Operation_Mode', y='Count', color='Efficiency_Status', color_discrete_map=COLORS, title='Efficiency by Operation Mode', barmode='group', template=TEMPLATE)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff')
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        hourly = fdf.groupby(['Hour','Efficiency_Status']).size().reset_index(name='Count')
        fig = px.line(hourly, x='Hour', y='Count', color='Efficiency_Status', color_discrete_map=COLORS, title='Hourly Efficiency Pattern', template=TEMPLATE, markers=True)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff')
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.box(fdf, x='Efficiency_Status', y='Error_Rate_%', color='Efficiency_Status', color_discrete_map=COLORS, title='Error Rate by Efficiency', template=TEMPLATE)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


    # ── Factory Health Score ──
    st.markdown("---")
    st.markdown("### Factory Health Intelligence")
    
    err_score = max(0, 100 - fdf['Error_Rate_%'].mean() * 8)
    prod_score = min(100, fdf['Production_Speed_units_per_hr'].mean() / 5)
    qual_score = max(0, 100 - fdf['Quality_Control_Defect_Rate_%'].mean() * 15)
    net_score = max(0, 100 - fdf['Network_Latency_ms'].mean() * 1.5)
    health_score = round(err_score * 0.35 + prod_score * 0.25 + qual_score * 0.25 + net_score * 0.15, 1)
    
    badge = "Excellent" if health_score >= 80 else "Good" if health_score >= 60 else "Risky"
    badge_color = "#00C853" if health_score >= 80 else "#FFD600" if health_score >= 60 else "#FF1744"
    
    h_col1, h_col2, h_col3 = st.columns([1, 1, 1])
    with h_col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            title={'text': "Factory Health Score", 'font': {'color': '#e2e8f5', 'size': 16}},
            number={'font': {'color': '#e2e8f5', 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#8892b0'},
                'bar': {'color': badge_color},
                'bgcolor': 'rgba(15,20,50,0.5)',
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(255,23,68,0.15)'},
                    {'range': [40, 70], 'color': 'rgba(255,214,0,0.15)'},
                    {'range': [70, 100], 'color': 'rgba(0,200,83,0.15)'}
                ],
                'threshold': {'line': {'color': badge_color, 'width': 4}, 'thickness': 0.8, 'value': health_score}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff', height=280, margin=dict(t=60,b=20,l=30,r=30))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with h_col2:
        st.markdown(f"""<div class='metric-card' style='padding:20px;text-align:left;'>
        <div style='font-size:0.75rem;color:rgba(148,163,200,0.7);text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;'>Health Breakdown</div>
        <div style='margin:8px 0;'><span style='color:#8892b0;'>Error Control:</span> <span style='color:{badge_color};font-weight:700;'>{err_score:.0f}/100</span></div>
        <div style='margin:8px 0;'><span style='color:#8892b0;'>Production Speed:</span> <span style='color:{badge_color};font-weight:700;'>{prod_score:.0f}/100</span></div>
        <div style='margin:8px 0;'><span style='color:#8892b0;'>Quality Control:</span> <span style='color:{badge_color};font-weight:700;'>{qual_score:.0f}/100</span></div>
        <div style='margin:8px 0;'><span style='color:#8892b0;'>Network Stability:</span> <span style='color:{badge_color};font-weight:700;'>{net_score:.0f}/100</span></div>
        <div style='margin-top:16px;padding:8px 16px;background:{badge_color};border-radius:20px;display:inline-block;font-weight:700;color:{"#000" if badge!="Risky" else "#fff"};'>{badge}</div>
        </div>""", unsafe_allow_html=True)
    
    with h_col3:
        # Anomaly Detection
        machine_err = fdf.groupby('Machine_ID')['Error_Rate_%'].mean()
        threshold = machine_err.mean() + 2 * machine_err.std()
        risky = machine_err[machine_err > threshold].sort_values(ascending=False).head(5)
        
        risky_html = ""
        if len(risky) > 0:
            for m, e in risky.items():
                risky_html += f"<div style='margin:6px 0;padding:6px 12px;background:rgba(255,23,68,0.12);border-left:3px solid #FF1744;border-radius:0 8px 8px 0;'><span style='color:#FF5252;font-weight:600;'>Machine {m}</span> <span style='color:#8892b0;'>— Error Rate: {e:.1f}%</span></div>"
        else:
            risky_html = "<div style='color:#69F0AE;'>✓ No anomalous machines detected</div>"
        
        st.markdown(f"""<div class='metric-card' style='padding:20px;text-align:left;'>
        <div style='font-size:0.75rem;color:rgba(148,163,200,0.7);text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;'>Anomaly Detection</div>
        <div style='font-size:0.8rem;color:#8892b0;margin-bottom:10px;'>Machines exceeding 2σ error threshold ({threshold:.1f}%)</div>
        {risky_html}
        </div>""", unsafe_allow_html=True)

    # ── Business Impact ──
    st.markdown("---")
    st.markdown("### Business Impact Estimation")
    bi1, bi2, bi3, bi4 = st.columns(4)
    
    low_pct = low_n / max(total, 1)
    cost_per_unit = 12
    units_lost = int(low_pct * fdf['Production_Speed_units_per_hr'].sum() * 0.15)
    cost_saved = units_lost * cost_per_unit
    time_saved = round(total * 0.02 / 60, 1)
    
    with bi1:
        st.markdown(f"""<div class='metric-card'><div class='metric-value' style='font-size:1.8rem;'>${cost_saved:,.0f}</div><div class='metric-label'>Est. Cost Savings</div></div>""", unsafe_allow_html=True)
    with bi2:
        st.markdown(f"""<div class='metric-card'><div class='metric-value' style='font-size:1.8rem;'>{time_saved}h</div><div class='metric-label'>Decision Time Saved</div></div>""", unsafe_allow_html=True)
    with bi3:
        st.markdown(f"""<div class='metric-card'><div class='metric-value' style='font-size:1.8rem;'>{units_lost:,}</div><div class='metric-label'>Units Loss Prevention</div></div>""", unsafe_allow_html=True)
    with bi4:
        pct_auto = 99.99
        st.markdown(f"""<div class='metric-card'><div class='metric-value' style='font-size:1.8rem;'>{pct_auto}%</div><div class='metric-label'>Automated Classification</div></div>""", unsafe_allow_html=True)

    # ── Download Reports ──
    st.markdown("---")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        csv_data = fdf.to_csv(index=False)
        st.download_button("Download Filtered Data (CSV)", csv_data, "manufacturing_data.csv", "text/csv", use_container_width=True)
    with dl2:
        summary_text = f"""EXECUTIVE SUMMARY — Manufacturing Efficiency AI
Generated: Auto-generated from dashboard
Records: {total:,} | High: {high_n:,} | Medium: {med_n:,} | Low: {low_n:,}
Factory Health Score: {health_score}/100 ({badge})
Best Model: Random Forest (99.99% accuracy)
Top Risk Factor: Error Rate (32.7% importance)
Estimated Cost Savings: ${cost_saved:,.0f}
Recommendation: Focus on reducing error rates in low-efficiency machines."""
        st.download_button("Download Executive Summary", summary_text, "executive_summary.txt", "text/plain", use_container_width=True)
    with dl3:
        if os.path.exists(os.path.join(BASE, 'models', 'feature_importance.csv')):
            with open(os.path.join(BASE, 'models', 'feature_importance.csv'), 'r') as fi:
                fi_data = fi.read()
            st.download_button("Download Feature Importance", fi_data, "feature_importance.csv", "text/csv", use_container_width=True)

# ── TAB 2: Predictions ──
with tab2:
    st.markdown("## Real-Time Efficiency Prediction")
    if 'best_model' in models and 'scaler' in models and 'feature_columns' in models:
        model = models['best_model']
        scaler = models['scaler']
        le = models['label_encoder']
        le_mode = models['mode_encoder']
        feat_cols = models['feature_columns']

        st.markdown("### Enter Machine Parameters")
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            temp = st.slider("Temperature (°C)", 30.0, 90.0, 60.0, 0.5)
            vibration = st.slider("Vibration (Hz)", 0.0, 5.0, 2.5, 0.1)
            power = st.slider("Power (kW)", 1.0, 10.0, 5.5, 0.1)
        with pc2:
            latency = st.slider("Network Latency (ms)", 1.0, 50.0, 25.0, 0.5)
            packet_loss = st.slider("Packet Loss (%)", 0.0, 5.0, 2.5, 0.1)
            defect_rate = st.slider("Defect Rate (%)", 0.0, 10.0, 5.0, 0.1)
        with pc3:
            prod_speed = st.slider("Production Speed", 50.0, 500.0, 250.0, 5.0)
            maint_score = st.slider("Maintenance Score", 0.0, 1.0, 0.5, 0.01)
            error_rate = st.slider("Error Rate (%)", 0.0, 15.0, 7.5, 0.1)
        with pc4:
            op_mode = st.selectbox("Operation Mode", ['Active', 'Idle', 'Maintenance'])
            hour = st.slider("Hour", 0, 23, 12)
            dow = st.slider("Day of Week", 0, 6, 3)

        if st.button("Predict Efficiency", use_container_width=True):
            op_enc = le_mode.transform([op_mode])[0]
            is_weekend = 1 if dow >= 5 else 0
            eer = prod_speed / (power + 0.001)
            eor = error_rate / (prod_speed + 0.001)
            nr = 1 / (1 + latency/50 + packet_loss/5)
            t_med, t_std = 60.0, 17.3
            v_med, v_std = 2.5, 1.44
            ss = (1 - abs(temp - t_med)/t_std) * (1 - abs(vibration - v_med)/v_std)
            qps = prod_speed * (1 - defect_rate/100)
            mes = maint_score * (1 - error_rate/15)
            pvr = power / (vibration + 0.001)
            mhs = maint_score*0.3 + (1-error_rate/15)*0.25 + (1-defect_rate/10)*0.2 + nr*0.15 + (prod_speed/500)*0.1
            raw = [temp, vibration, power, latency, packet_loss, defect_rate, prod_speed, maint_score, error_rate, eer, eor, nr, ss, qps, mes, pvr, mhs, op_enc, hour, dow, is_weekend]
            inp = pd.DataFrame([raw], columns=feat_cols)
            inp.replace([np.inf, -np.inf], 0, inplace=True)
            inp.fillna(0, inplace=True)
            inp_scaled = scaler.transform(inp)
            pred = model.predict(inp_scaled)[0]
            label = le.inverse_transform([pred])[0]
            proba = model.predict_proba(inp_scaled)[0]
            conf = float(proba.max()) * 100

            st.markdown("---")
            rc1, rc2, rc3 = st.columns(3)
            css_class = f"status-{label.lower()}"
            with rc1: st.markdown(f"<div class='metric-card'><div class='metric-label'>Predicted Status</div><br><span class='{css_class}'>{label}</span></div>", unsafe_allow_html=True)
            with rc2: st.markdown(f"<div class='metric-card'><div class='metric-value'>{conf:.1f}%</div><div class='metric-label'>Confidence</div></div>", unsafe_allow_html=True)
            with rc3:
                prob_data = {le.inverse_transform([i])[0]: f"{p*100:.1f}%" for i, p in enumerate(proba)}
                st.markdown(f"<div class='metric-card'><div class='metric-label'>Class Probabilities</div><br>{'<br>'.join(f'<b>{k}:</b> {v}' for k,v in prob_data.items())}</div>", unsafe_allow_html=True)

            fig = go.Figure(go.Bar(x=[le.inverse_transform([i])[0] for i in range(len(proba))], y=proba*100, marker_color=[COLORS.get(le.inverse_transform([i])[0],'#888') for i in range(len(proba))]))
            fig.update_layout(title='Prediction Probabilities', yaxis_title='Probability (%)', template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Run analysis.py first to train models.")

# ── TAB 3: Machine Insights ──
with tab3:
    st.markdown("## Machine-Level Insights")
    col1, col2 = st.columns(2)
    with col1:
        mach_eff = fdf.groupby(['Machine_ID','Efficiency_Status']).size().unstack(fill_value=0)
        for c in ['High','Medium','Low']:
            if c not in mach_eff.columns: mach_eff[c] = 0
        mach_eff_pct = mach_eff.div(mach_eff.sum(axis=1), axis=0)*100
        mach_eff_pct = mach_eff_pct.sort_values('High', ascending=True).reset_index()
        fig = go.Figure()
        for status in ['Low','Medium','High']:
            fig.add_trace(go.Bar(name=status, y=mach_eff_pct['Machine_ID'].astype(str), x=mach_eff_pct[status], orientation='h', marker_color=COLORS[status]))
        fig.update_layout(barmode='stack', title='Efficiency Profile by Machine', template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff', height=600)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        mach_metrics = fdf.groupby('Machine_ID').agg(
            Avg_Production=('Production_Speed_units_per_hr','mean'),
            Avg_Error=('Error_Rate_%','mean'),
            Avg_Defect=('Quality_Control_Defect_Rate_%','mean'),
            Avg_Maintenance=('Predictive_Maintenance_Score','mean')
        ).reset_index()
        fig = px.scatter(mach_metrics, x='Avg_Error', y='Avg_Production', size='Avg_Maintenance', color='Avg_Defect', hover_data=['Machine_ID'], title='Machine Performance Map', template=TEMPLATE, color_continuous_scale='RdYlGn_r')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff')
        st.plotly_chart(fig, use_container_width=True)

    sel_machine = st.selectbox("Select Machine for Details", sorted(fdf['Machine_ID'].unique()))
    mdf = fdf[fdf['Machine_ID']==sel_machine]
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1: st.metric("Avg Production", f"{mdf['Production_Speed_units_per_hr'].mean():.0f} u/hr")
    with mc2: st.metric("Avg Error Rate", f"{mdf['Error_Rate_%'].mean():.1f}%")
    with mc3: st.metric("Avg Defect Rate", f"{mdf['Quality_Control_Defect_Rate_%'].mean():.1f}%")
    with mc4: st.metric("High Eff %", f"{(mdf['Efficiency_Status']=='High').mean()*100:.1f}%")

    fig = px.histogram(mdf, x='Efficiency_Status', color='Efficiency_Status', color_discrete_map=COLORS, title=f'Machine {sel_machine} Efficiency Distribution', template=TEMPLATE)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 4: Explainability ──
with tab4:
    st.markdown("## Model Explainability")
    if 'feature_importance' in models:
        imp_df = models['feature_importance'] if isinstance(models['feature_importance'], pd.DataFrame) else pd.DataFrame()
        if not imp_df.empty:
            imp_df = imp_df.sort_values('Importance', ascending=True)
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title='Feature Importance (All Features)', template=TEMPLATE, color='Importance', color_continuous_scale='Viridis')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff', height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                top10 = imp_df.tail(10)
                fig = px.bar(top10, x='Importance', y='Feature', orientation='h', title='Top 10 Drivers of Efficiency', template=TEMPLATE, color='Importance', color_continuous_scale='Plasma')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff', height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Key Insights")
            top3 = imp_df.tail(3).iloc[::-1]
            for _, row in top3.iterrows():
                st.markdown(f"- **{row['Feature']}** contributes **{row['Importance']*100:.1f}%** to the classification decision")

    if 'model_summary' in models:
        ms = models['model_summary']
        st.markdown("### Model Comparison")
        if 'models' in ms:
            comp = pd.DataFrame(ms['models']).T.reset_index()
            comp.columns = ['Model', 'Accuracy', 'F1 Score']
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Accuracy', x=comp['Model'], y=comp['Accuracy'], marker_color='#7b2ff7'))
            fig.add_trace(go.Bar(name='F1 Score', x=comp['Model'], y=comp['F1 Score'], marker_color='#00d2ff'))
            fig.update_layout(barmode='group', title='Model Performance Comparison', template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff', yaxis_range=[0.8,1.01])
            st.plotly_chart(fig, use_container_width=True)
            comp_display = comp.copy()
            comp_display['Accuracy'] = comp_display['Accuracy'].apply(lambda x: f"{x:.4f}")
            comp_display['F1 Score'] = comp_display['F1 Score'].apply(lambda x: f"{x:.4f}")
            st.dataframe(comp_display, use_container_width=True, hide_index=True)

    if os.path.exists(os.path.join(BASE, 'charts', 'confusion_matrix.png')):
        st.markdown("### Confusion Matrix")
        st.image(os.path.join(BASE, 'charts', 'confusion_matrix.png'), width=600)

# ── TAB 5: Network & Sensors ──
with tab5:
    st.markdown("## Network & Sensor Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(fdf.sample(min(5000,len(fdf))), x='Network_Latency_ms', y='Production_Speed_units_per_hr', color='Efficiency_Status', color_discrete_map=COLORS, title='Network Latency vs Production', template=TEMPLATE, opacity=0.5)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(fdf.sample(min(5000,len(fdf))), x='Packet_Loss_%', y='Error_Rate_%', color='Efficiency_Status', color_discrete_map=COLORS, title='Packet Loss vs Error Rate', template=TEMPLATE, opacity=0.5)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff')
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(fdf, x='Efficiency_Status', y='Network_Latency_ms', color='Efficiency_Status', color_discrete_map=COLORS, title='Network Latency by Efficiency', template=TEMPLATE)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.box(fdf, x='Efficiency_Status', y='Temperature_C', color='Efficiency_Status', color_discrete_map=COLORS, title='Temperature by Efficiency', template=TEMPLATE)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Sensor Correlation Matrix")
    sensor_cols = ['Temperature_C','Vibration_Hz','Power_Consumption_kW','Network_Latency_ms','Packet_Loss_%','Error_Rate_%','Production_Speed_units_per_hr']
    corr = fdf[sensor_cols].corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', title='Feature Correlation Heatmap', template=TEMPLATE)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e6f1ff', height=500)
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 6: AI Insights ──
with tab6:
    st.markdown("## AI-Powered Insights")
    st.markdown("<p style='color:rgba(148,163,200,0.7);'>Intelligent analysis powered by NVIDIA NIM + Google Gemini with local data-driven fallback.</p>", unsafe_allow_html=True)

    ai_col1, ai_col2 = st.columns(2)
    with ai_col1:
        if st.button("Generate Executive Summary", use_container_width=True, key="btn_exec"):
            with st.spinner("Generating executive summary..."):
                total_r = len(fdf)
                high_p = len(fdf[fdf['Efficiency_Status']=='High']) / max(total_r,1) * 100
                med_p = len(fdf[fdf['Efficiency_Status']=='Medium']) / max(total_r,1) * 100
                low_p = len(fdf[fdf['Efficiency_Status']=='Low']) / max(total_r,1) * 100
                avg_err = fdf['Error_Rate_%'].mean()
                avg_prod = fdf['Production_Speed_units_per_hr'].mean()
                prompt = f"""You are a manufacturing efficiency expert. Write a concise executive summary (5-6 sentences) for a factory manager based on this data:
- {total_r:,} records from {len(fdf['Machine_ID'].unique())} machines
- Efficiency: High={high_p:.1f}%, Medium={med_p:.1f}%, Low={low_p:.1f}%
- Average Error Rate: {avg_err:.1f}%
- Average Production Speed: {avg_prod:.0f} units/hr
- Top efficiency driver: Error_Rate (32.7% importance)
- Best AI model: Random Forest with 99.99% accuracy
Be professional, data-driven, and actionable."""
                result = get_ai_response(prompt)
                if result:
                    st.markdown(f"<div class='metric-card' style='text-align:left;padding:24px;'>{result}</div>", unsafe_allow_html=True)
                    st.caption("✦ Generated by external AI provider")
                else:
                    result = generate_local_summary(fdf)
                    st.markdown(result)
                    st.caption("✦ Generated from local data analysis (external AI unavailable)")

    with ai_col2:
        if st.button("Generate Improvement Recommendations", use_container_width=True, key="btn_rec"):
            with st.spinner("Generating recommendations..."):
                low_machines = fdf[fdf['Efficiency_Status']=='Low'].groupby('Machine_ID').size().sort_values(ascending=False).head(5)
                top_low = ', '.join([f"{m}: {c} incidents" for m, c in low_machines.items()])
                prompt = f"""You are a manufacturing efficiency consultant. Based on this analysis, provide 5 specific, actionable recommendations to improve factory efficiency:
- Machines with most Low efficiency: {top_low}
- Error Rate is the #1 driver of efficiency (32.7%)
- Production Speed is #3 driver (17.8%)
- Network latency has minimal impact
- Operation modes: Active, Idle, Maintenance all show similar patterns
Be specific, practical, and prioritize by impact. Use numbered list."""
                result = get_ai_response(prompt)
                if result:
                    st.markdown(f"<div class='metric-card' style='text-align:left;padding:24px;'>{result}</div>", unsafe_allow_html=True)
                    st.caption("✦ Generated by external AI provider")
                else:
                    result = generate_local_recommendations(fdf)
                    st.markdown(result)
                    st.caption("✦ Generated from local data analysis (external AI unavailable)")

    st.markdown("---")

    st.markdown("### Ask AI About Your Data")
    user_question = st.text_input("Ask a question about the manufacturing data:", placeholder="e.g., Why is efficiency low for certain machines?")
    if user_question and st.button("Ask AI", use_container_width=True, key="btn_ask"):
        with st.spinner("Thinking..."):
            context = f"""Manufacturing dataset context:
- 100K records, 50 machines, 3 operation modes
- Target: Efficiency_Status (High 3%, Medium 19.2%, Low 77.8%)
- Key features: Error_Rate, Production_Speed, Temperature, Vibration, Power, Network_Latency, Packet_Loss
- Error_Rate is the strongest predictor (32.7% importance)
- Random Forest achieves 99.99% classification accuracy
- High efficiency machines have Error_Rate ~1%, Low efficiency ~8.9%

User question: {user_question}

Provide a clear, data-driven answer in 3-4 sentences."""
            result = get_ai_response(context)
            if result:
                st.markdown(f"<div class='metric-card' style='text-align:left;padding:24px;'>{result}</div>", unsafe_allow_html=True)
            else:
                st.info("External AI providers are currently unavailable. Here are key data facts that may answer your question:")
                st.markdown(f"""
- **Error Rate** is the #1 efficiency driver (32.7% importance). High-efficiency machines have ~1% error rate vs ~8.9% for Low.
- **Production Speed** averages {fdf['Production_Speed_units_per_hr'].mean():.0f} units/hr. High-efficiency machines produce {fdf[fdf['Efficiency_Status']=='High']['Production_Speed_units_per_hr'].mean():.0f} units/hr.
- **{fdf['Machine_ID'].nunique()} machines** monitored. Machine {fdf.groupby('Machine_ID')['Error_Rate_%'].mean().idxmax()} has the highest error rate.
- **Network latency** ({fdf['Network_Latency_ms'].mean():.1f}ms avg) has minimal impact on efficiency classification.
""")

    st.markdown("---")
    st.markdown("<p style='color:rgba(148,163,200,0.4);font-size:0.75rem;'>Note: AI insights try NVIDIA NIM (DeepSeek, GPT-OSS) and Google Gemini. If unavailable, data-driven analysis is shown. Core predictions always use the locally-trained Random Forest model.</p>", unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.markdown("<p style='text-align:center;color:rgba(148,163,200,0.5);font-size:0.8rem;letter-spacing:1px;font-weight:400;'>AI-BASED MANUFACTURING EFFICIENCY CLASSIFICATION &nbsp;•&nbsp; THALES GROUP &nbsp;•&nbsp; UNIFIED MENTOR INTERNSHIP</p>", unsafe_allow_html=True)
