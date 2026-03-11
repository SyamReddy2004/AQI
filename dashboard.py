import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AQI India Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark premium theme */
    .stApp { background-color: #0a0f1e; color: #e0f2fe; }
    .stSidebar { background-color: #0d1629; }
    .stMetric { background: linear-gradient(135deg, #0d1a2e, #112240);
                border: 1px solid rgba(0,212,255,0.2);
                border-radius: 12px; padding: 16px; }
    .stMetric label { color: #94a3b8 !important; font-size: 13px !important; }
    .stMetric [data-testid="metric-container"] { color: white; }
    h1, h2, h3 { color: #00d4ff !important; }
    .stSelectbox label, .stSlider label { color: #94a3b8 !important; }
    div[data-testid="stSidebarNav"] { padding-top: 20px; }
    .prediction-box {
        background: linear-gradient(135deg, #0d2137, #112d4e);
        border: 2px solid #00d4ff;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .aqi-value { font-size: 80px; font-weight: 800; }
    .aqi-label { font-size: 22px; font-weight: 600; }
    .good    { color: #00ff88; }
    .moderate{ color: #FFD700; }
    .poor    { color: #FF8C00; }
    .vpoor   { color: #FF4500; }
    .severe  { color: #8B0000; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD & PROCESS DATA (cached)
# ─────────────────────────────────────────────
DATA_PATH = "city_day.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df.dropna(subset=['AQI'], inplace=True)
    numeric_cols = ['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene','AQI']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df.groupby('City')[col].transform(lambda x: x.fillna(x.median()))
    df['month']       = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['year']        = df['Date'].dt.year
    def get_season(m):
        if m in [12,1,2]:  return 'Winter'
        elif m in [3,4,5]: return 'Spring'
        elif m in [6,7,8]: return 'Monsoon'
        else:              return 'Autumn'
    df['season'] = df['month'].apply(get_season)
    le_city   = LabelEncoder()
    le_season = LabelEncoder()
    df['City_enc']   = le_city.fit_transform(df['City'])
    df['Season_enc'] = le_season.fit_transform(df['season'])
    return df, le_city, le_season

@st.cache_resource
def train_model(df):
    FEATURES = [c for c in ['PM2.5','PM10','NO2','CO','SO2','O3','City_enc','Season_enc','month','day_of_week','year'] if c in df.columns]
    X = df[FEATURES]
    y = df['AQI']
    imp = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=FEATURES)
    X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE":  mean_absolute_error(y_test, preds),
        "R2":   r2_score(y_test, preds)
    }
    return model, imp, FEATURES, metrics, X_test, y_test, preds

def aqi_category(aqi):
    if aqi <= 50:   return ("Good", "good")
    elif aqi <= 100: return ("Moderate", "moderate")
    elif aqi <= 200: return ("Poor", "poor")
    elif aqi <= 300: return ("Very Poor", "vpoor")
    else:            return ("Severe", "severe")

df, le_city, le_season = load_data()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 AQI India Dashboard")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Overview", "🔮 Predict AQI", "🏙️ City Analysis", "📈 Model Performance"])
    st.markdown("---")
    st.markdown("**Dataset Info**")
    st.markdown(f"- **Cities**: {df['City'].nunique()}")
    st.markdown(f"- **Records**: {len(df):,}")
    st.markdown(f"- **Period**: 2015 – 2020")

# ─────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🌿 Air Quality Index — India Dashboard")
    st.markdown("##### Predict & Monitor pollution levels across Indian cities (2015-2020)")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("📍 Cities Tracked", df['City'].nunique())
    with col2: st.metric("📄 Total Records",  f"{len(df):,}")
    with col3: st.metric("📊 Mean AQI",       f"{df['AQI'].mean():.1f}")
    with col4: st.metric("🔺 Max AQI Ever",   f"{df['AQI'].max():.0f}")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("AQI Distribution")
        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='#0a0f1e')
        ax.set_facecolor('#0d1629')
        ax.hist(df['AQI'], bins=60, color='#00d4ff', edgecolor='black', alpha=0.85)
        ax.set_xlabel('AQI', color='white'); ax.set_ylabel('Count', color='white')
        ax.tick_params(colors='white'); ax.spines['bottom'].set_color('#2d3748')
        ax.spines['left'].set_color('#2d3748'); ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)

    with c2:
        st.subheader("Average AQI by Month")
        monthly = df.groupby('month')['AQI'].mean()
        months  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        fig, ax = plt.subplots(figsize=(7, 3.5), facecolor='#0a0f1e')
        ax.set_facecolor('#0d1629')
        ax.plot(months, [monthly.get(i, 0) for i in range(1, 13)], marker='o', color='#00ff88', linewidth=2)
        ax.fill_between(months, [monthly.get(i, 0) for i in range(1, 13)], alpha=0.15, color='#00ff88')
        ax.set_xlabel('Month', color='white'); ax.set_ylabel('Avg AQI', color='white')
        ax.tick_params(colors='white', axis='x', rotation=30); ax.tick_params(colors='white', axis='y')
        for s in ax.spines.values(): s.set_color('#2d3748')
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("AQI Bucket Distribution")
    if 'AQI_Bucket' in df.columns:
        bucket_counts = df['AQI_Bucket'].value_counts()
        cols = st.columns(len(bucket_counts))
        for i, (label, count) in enumerate(bucket_counts.items()):
            pct = count / len(df) * 100
            cols[i].metric(label, f"{count:,}", f"{pct:.1f}%")

# ─────────────────────────────────────────────
# PAGE 2: PREDICT
# ─────────────────────────────────────────────
elif page == "🔮 Predict AQI":
    st.title("🔮 Predict Air Quality Index")
    st.markdown("Enter pollutant measurements to get an instant AQI prediction.")
    st.markdown("---")

    with st.spinner("Training Random Forest model…"):
        model, imp, FEATURES, metrics, X_test, y_test, preds = train_model(df)

    cities  = sorted(df['City'].unique())
    seasons = ['Winter', 'Spring', 'Monsoon', 'Autumn']

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("📍 Location & Time")
        city   = st.selectbox("City", cities)
        season = st.selectbox("Season", seasons)
        month  = st.slider("Month", 1, 12, 6)
        year   = st.slider("Year", 2015, 2026, 2024)
        dow    = st.slider("Day of Week (0=Mon)", 0, 6, 2)

    with col2:
        st.subheader("🧪 Pollutant Levels (μg/m³)")
        pm25 = st.slider("PM2.5",  0.0, 500.0, 65.0, 1.0)
        pm10 = st.slider("PM10",   0.0, 700.0, 120.0, 1.0)
        no2  = st.slider("NO2",    0.0, 200.0, 35.0, 1.0)
        co   = st.slider("CO",     0.0, 10.0,  1.5, 0.1)
        so2  = st.slider("SO2",    0.0, 200.0, 20.0, 1.0)
        o3   = st.slider("O3 (Ozone)", 0.0, 200.0, 30.0, 1.0)

    if st.button("🚀 Predict AQI Now", type="primary", use_container_width=True):
        city_enc   = le_city.transform([city])[0]
        season_enc = le_season.transform([season])[0]
        input_data = pd.DataFrame([[pm25, pm10, no2, co, so2, o3, city_enc, season_enc, month, dow, year]], columns=FEATURES)
        input_imp  = pd.DataFrame(imp.transform(input_data), columns=FEATURES)
        prediction = model.predict(input_imp)[0]
        label, cls = aqi_category(prediction)
        st.markdown(f"""
        <div class="prediction-box">
            <div class="aqi-value {cls}">{prediction:.0f}</div>
            <div class="aqi-label {cls}">AQI Level: {label}</div>
            <p style="color:#94a3b8; margin-top:10px;">City: {city} | Season: {season} | {month}/2024</p>
        </div>
        """, unsafe_allow_html=True)

        tip_map = {
            "Good": "✅ Air quality is satisfactory. Enjoy outdoor activities!",
            "Moderate": "⚠️ Air quality is acceptable. Unusually sensitive people should limit prolonged outdoor exertion.",
            "Poor": "😷 Members of sensitive groups may experience health effects. Wear a mask outdoors.",
            "Very Poor": "🚨 Health alert: everyone may experience more serious health effects. Avoid outdoor exertion.",
            "Severe": "☠️ Health emergency. Everyone is likely to be affected. Stay indoors!"
        }
        st.info(tip_map[label])

# ─────────────────────────────────────────────
# PAGE 3: CITY ANALYSIS
# ─────────────────────────────────────────────
elif page == "🏙️ City Analysis":
    st.title("🏙️ City-wise AQI Analysis")
    selected_city = st.selectbox("Select a City", sorted(df['City'].unique()))
    city_df = df[df['City'] == selected_city].sort_values('Date')

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg AQI",  f"{city_df['AQI'].mean():.1f}")
    c2.metric("Max AQI",  f"{city_df['AQI'].max():.0f}")
    c3.metric("Min AQI",  f"{city_df['AQI'].min():.0f}")

    st.subheader(f"AQI Trend — {selected_city}")
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#0a0f1e')
    ax.set_facecolor('#0d1629')
    ax.plot(city_df['Date'], city_df['AQI'], color='#00d4ff', linewidth=1.2, alpha=0.85)
    ax.fill_between(city_df['Date'], city_df['AQI'], alpha=0.1, color='#00d4ff')
    ax.axhline(200, color='#FF8C00', linestyle='--', linewidth=1, label='Poor threshold (200)')
    ax.axhline(300, color='#FF4500', linestyle='--', linewidth=1, label='Very Poor threshold (300)')
    ax.set_xlabel('Date', color='white'); ax.set_ylabel('AQI', color='white')
    ax.tick_params(colors='white'); ax.legend(labelcolor='white', facecolor='#0d1629')
    for s in ax.spines.values(): s.set_color('#2d3748')
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Monthly Average Pollutants")
    monthly_city = city_df.groupby('month')[['PM2.5','PM10','NO2','SO2','CO','O3']].mean()
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='#0a0f1e')
    ax.set_facecolor('#0d1629')
    for col, clr in zip(['PM2.5','PM10','NO2'], ['#00d4ff','#ff6b6b','#00ff88']):
        if col in monthly_city.columns:
            ax.plot(range(1,13), [monthly_city[col].get(i,0) for i in range(1,13)], marker='o', label=col, color=clr, linewidth=1.5)
    ax.set_xticks(range(1,13)); ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], rotation=30)
    ax.set_xlabel('Month', color='white'); ax.set_ylabel('Concentration', color='white')
    ax.tick_params(colors='white'); ax.legend(labelcolor='white', facecolor='#0d1629')
    for s in ax.spines.values(): s.set_color('#2d3748')
    fig.tight_layout()
    st.pyplot(fig)

# ─────────────────────────────────────────────
# PAGE 4: MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    with st.spinner("Training model…"):
        model, imp, FEATURES, metrics, X_test, y_test, preds = train_model(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{metrics['RMSE']:.2f}")
    c2.metric("MAE",  f"{metrics['MAE']:.2f}")
    c3.metric("R² Score", f"{metrics['R2']:.4f}")

    st.subheader("Actual vs Predicted AQI")
    sample = min(500, len(y_test))
    idx = np.random.choice(len(y_test), sample, replace=False)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor='#0a0f1e')
    ax.set_facecolor('#0d1629')
    ax.scatter(np.array(y_test)[idx], preds[idx], color='#00d4ff', alpha=0.5, s=18, label='Predictions')
    mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual AQI', color='white'); ax.set_ylabel('Predicted AQI', color='white')
    ax.tick_params(colors='white'); ax.legend(labelcolor='white', facecolor='#0d1629')
    for s in ax.spines.values(): s.set_color('#2d3748')
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Feature Importance")
    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(9, 4), facecolor='#0a0f1e')
    ax.set_facecolor('#0d1629')
    fi.plot(kind='barh', color='#00d4ff', ax=ax)
    ax.set_xlabel('Importance', color='white')
    ax.tick_params(colors='white')
    for s in ax.spines.values(): s.set_color('#2d3748')
    fig.tight_layout()
    st.pyplot(fig)
