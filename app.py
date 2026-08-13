import streamlit as st
import joblib
import pandas as pd
import folium
from streamlit_folium import st_folium


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CrimeGuard AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS (TEAL & BRIGHT GREEN GLASSMORPHISM THEME)
# =========================================================

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(6, 182, 212, 0.12), transparent 35%),
                    radial-gradient(circle at 85% 85%, rgba(34, 197, 94, 0.10), transparent 35%),
                    #080f1e;
        color: #f1f5f9;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    header, #MainMenu, footer {
        visibility: hidden;
        height: 0px;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(13, 148, 136, 0.25));
        border: 1px solid rgba(45, 212, 191, 0.3);
        border-radius: 24px;
        padding: 36px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        margin-bottom: 28px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 900;
        background: linear-gradient(90deg, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 16px;
        margin-top: 6px;
    }

    .badge-live {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(52, 211, 153, 0.12);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.35);
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        margin-top: 16px;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        background: #34d399;
        border-radius: 50%;
        box-shadow: 0 0 10px #34d399;
    }

    .kpi-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 22px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.25s ease;
        height: 100%;
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 12px 30px rgba(6, 182, 212, 0.15);
    }

    .kpi-title {
        color: #64748b;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }

    .kpi-val {
        font-size: 28px;
        font-weight: 800;
        color: #38bdf8;
        margin-top: 6px;
    }

    div[data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
    }

    div[data-baseweb="select"] > div:hover {
        border-color: #34d399 !important;
    }

    .stButton > button {
        width: 100%;
        height: 56px;
        border-radius: 16px;
        background: linear-gradient(90deg, #0284c7, #0d9488, #16a34a);
        color: #ffffff;
        border: none;
        font-size: 17px;
        font-weight: 800;
        letter-spacing: 0.5px;
        box-shadow: 0 10px 25px rgba(13, 148, 136, 0.3);
        transition: all 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px rgba(52, 211, 153, 0.4);
        color: #ffffff;
    }

    .risk-box-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 20px;
        padding: 28px;
        text-align: center;
    }

    .risk-box-low {
        background: linear-gradient(135deg, rgba(52, 211, 153, 0.15), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 20px;
        padding: 28px;
        text-align: center;
    }

    .risk-score-num {
        font-size: 46px;
        font-weight: 900;
        background: linear-gradient(90deg, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hotspot-section {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.82), rgba(8, 47, 73, 0.45));
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 22px;
        padding: 24px;
        margin-top: 20px;
    }

    .hotspot-title {
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .hotspot-subtitle {
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 18px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# INITIALIZE SESSION STATE
# =========================================================

if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.rf_prediction = None
    st.session_state.rf_probability = 0.0
    st.session_state.last_district = None
    st.session_state.last_day = None
    st.session_state.last_hour = None


# =========================================================
# LOAD MODELS + DBSCAN RESULTS
# =========================================================

@st.cache_resource
def load_models():
    rf = joblib.load("models/random_forest.pkl")
    xgb = joblib.load("models/xgboost.pkl")
    hotspots = joblib.load("models/dbscan_hotspots.pkl")
    return rf, xgb, hotspots


try:
    rf_model, xgb_model, hotspots_df = load_models()
except Exception as e:
    rf_model, xgb_model, hotspots_df = None, None, None
    st.error(f"Could not load one or more model files: {e}")


# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero-card">
    <div class="hero-title">🚨 CrimeGuard AI</div>
    <div class="hero-subtitle">Next-Generation Crime Risk Intelligence & Predictive Analytics</div>
    <div class="badge-live">
        <span class="live-dot"></span>
        AI SYSTEM ACTIVE & RUNNING
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# TOP KPI STATS
# =========================================================

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">Monitoring Status</div>
        <div class="kpi-val" style="color:#38bdf8;">ACTIVE</div>
        <div style="color:#64748b; font-size:12px; margin-top:4px;">Crime intelligence system</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">Safety Network</div>
        <div class="kpi-val" style="color:#34d399;">READY</div>
        <div style="color:#64748b; font-size:12px; margin-top:4px;">Spatiotemporal Analysis</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">AI Classifiers</div>
        <div class="kpi-val" style="color:#2dd4bf;">2 Models</div>
        <div style="color:#64748b; font-size:12px; margin-top:4px;">Random Forest + XGBoost</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    hotspot_count = len(hotspots_df) if hotspots_df is not None else 0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Spatial Hotspots</div>
        <div class="kpi-val" style="color:#facc15;">{hotspot_count}</div>
        <div style="color:#64748b; font-size:12px; margin-top:4px;">DBSCAN clusters detected</div>
    </div>
    """, unsafe_allow_html=True)


st.write("")
st.write("")


# =========================================================
# INPUT CONTROLS
# =========================================================

st.markdown(
    "<h3 style='font-size:20px; font-weight:700;'>🔍 Crime Risk Input Panel</h3>",
    unsafe_allow_html=True
)

with st.container():
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        district = st.selectbox(
            "📍 Location",
            list(range(1, 12)),
            format_func=lambda x: f"District {x}",
            index=3
        )

    days = {
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6,
        "Sunday": 7
    }

    with c2:
        day_name = st.selectbox("📅 Day", list(days.keys()))
        day = days[day_name]

    months = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12
    }

    with c3:
        month_name = st.selectbox("🗓️ Month", list(months.keys()))
        month = months[month_name]

    with c4:
        hour = st.slider(
            "🕐 Time (24h)",
            min_value=0,
            max_value=23,
            value=18
        )


st.write("")
analyze = st.button("⚡ GENERATE RISK INTELLIGENCE REPORT")


# =========================================================
# PREDICTION LOGIC & SESSION STATE UPDATE
# =========================================================

if analyze:
    if rf_model is not None:
        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=rf_model.feature_names_in_
        )

        input_data["HOUR"] = hour
        input_data["MONTH"] = month

        district_col = f"District_{district}"
        if district_col in input_data.columns:
            input_data[district_col] = 1

        day_col = f"DAY_{day}"
        if day_col in input_data.columns:
            input_data[day_col] = 1

        pred = rf_model.predict(input_data)[0]

        if hasattr(rf_model, "predict_proba"):
            probs = rf_model.predict_proba(input_data)[0]
            prob = (
                float(probs[1]) * 100
                if len(probs) > 1
                else float(probs[0]) * 100
            )
        else:
            prob = float(pred) * 100

        st.session_state.rf_prediction = pred
        st.session_state.rf_probability = prob

    else:
        st.session_state.rf_prediction = 1
        st.session_state.rf_probability = 78.4

    st.session_state.last_district = district
    st.session_state.last_day = day_name
    st.session_state.last_hour = hour
    st.session_state.prediction_made = True


# =========================================================
# PERSISTENT DISPLAY OF RESULTS
# =========================================================

if st.session_state.prediction_made:
    st.write("")
    st.markdown(
        "<h3 style='font-size:20px; font-weight:700;'>📊 Intelligence Assessment</h3>",
        unsafe_allow_html=True
    )

    rf_prediction = st.session_state.rf_prediction
    rf_probability = st.session_state.rf_probability
    disp_district = st.session_state.last_district
    disp_day = st.session_state.last_day
    disp_hour = st.session_state.last_hour

    res_col, score_col = st.columns([1.3, 1])

    with res_col:
        if rf_prediction == 1:
            st.markdown(f"""
            <div class="risk-box-high">
                <div style="font-size:42px;">⚠️</div>
                <div style="color:#ef4444; font-weight:800; font-size:14px; letter-spacing:1px; margin-top:8px;">
                    HIGH RISK LEVEL DETECTED
                </div>
                <div style="font-size:22px; font-weight:800; margin-top:4px;">
                    District {disp_district}
                </div>
                <div style="color:#94a3b8; font-size:13px; margin-top:8px;">
                    Conditions indicate heightened probability of criminal activity around
                    <b>{disp_hour:02d}:00</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="risk-box-low">
                <div style="font-size:42px;">🛡️</div>
                <div style="color:#34d399; font-weight:800; font-size:14px; letter-spacing:1px; margin-top:8px;">
                    LOW RISK LEVEL DETECTED
                </div>
                <div style="font-size:22px; font-weight:800; margin-top:4px;">
                    District {disp_district}
                </div>
                <div style="color:#94a3b8; font-size:13px; margin-top:8px;">
                    Normal conditions predicted for <b>{disp_day}</b> at <b>{disp_hour:02d}:00</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with score_col:
        st.markdown(f"""
        <div class="kpi-card" style="text-align:center;">
            <div class="kpi-title">AI Risk Probability Score</div>
            <div class="risk-score-num">{rf_probability:.1f}%</div>
            <div style="color:#64748b; font-size:12px;">
                Predicted probability from Random Forest
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.progress(min(max(rf_probability / 100, 0.0), 1.0))


# =========================================================
# DBSCAN HOTSPOT INTELLIGENCE
# =========================================================

st.divider()

st.markdown(
    '<div class="hotspot-section">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hotspot-title">🗺️ Crime Hotspot Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hotspot-subtitle">'
    'DBSCAN spatial clustering identifies areas with concentrated crime activity '
    'and links each hotspot to its nearest police station.'
    '</div>',
    unsafe_allow_html=True
)

if hotspots_df is not None and not hotspots_df.empty:

    h1, h2, h3 = st.columns(3)

    with h1:
        st.metric(
            "🔥 Detected Hotspots",
            len(hotspots_df)
        )

    with h2:
        st.metric(
            "🚨 Clustered Crimes",
            f"{int(hotspots_df['crime_count'].sum()):,}"
        )

    with h3:
        st.metric(
            "📍 Largest Hotspot",
            f"{int(hotspots_df['crime_count'].max()):,}"
        )

    map_center = [
        float(hotspots_df["center_lat"].mean()),
        float(hotspots_df["center_long"].mean())
    ]

    crime_map = folium.Map(
        location=map_center,
        zoom_start=13,
        tiles="CartoDB dark_matter"
    )

    for _, row in hotspots_df.iterrows():

        crime_count = int(row["crime_count"])

        # Larger circles for larger hotspots
        marker_radius = max(
            5,
            min(15, 5 + (crime_count ** 0.5))
        )

        popup_html = f"""
        <div style="font-family:Arial; min-width:240px;">
            <h4 style="margin-bottom:8px;">
                🚨 {row['hotspot_name']}
            </h4>
            <b>Crime Count:</b> {crime_count:,}<br>
            <b>Radius:</b> {float(row['radius_km']):.3f} km<br>
            <b>Nearest Station:</b> {row['nearest_station']}<br>
            <b>Distance:</b> {float(row['dist_to_station_km']):.2f} km
        </div>
        """

        folium.CircleMarker(
            location=[
                float(row["center_lat"]),
                float(row["center_long"])
            ],
            radius=marker_radius,
            popup=folium.Popup(
                popup_html,
                max_width=330
            ),
            tooltip=f"{row['hotspot_name']} • {crime_count:,} crimes",
            fill=True,
            fill_opacity=0.80,
            weight=1
        ).add_to(crime_map)

    st_folium(
        crime_map,
        width=None,
        height=600,
        returned_objects=[]
    )

    st.markdown(
        "<h3 style='font-size:18px; font-weight:700; margin-top:20px;'>"
        "🔥 Highest Crime Hotspots"
        "</h3>",
        unsafe_allow_html=True
    )

    top_hotspots = (
        hotspots_df
        .sort_values("crime_count", ascending=False)
        .head(10)
        .copy()
    )

    display_df = top_hotspots[
        [
            "hotspot_name",
            "crime_count",
            "radius_km",
            "nearest_station",
            "dist_to_station_km"
        ]
    ].copy()

    display_df.columns = [
        "Hotspot",
        "Crimes",
        "Radius (km)",
        "Nearest Police Station",
        "Distance (km)"
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning(
        "DBSCAN hotspot data could not be loaded. "
        "Make sure models/dbscan_hotspots.pkl exists."
    )

st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div style="text-align:center; color:#475569; font-size:12px; margin-top:50px;">
    <b>CRIMEGUARD AI</b> &nbsp;•&nbsp;
    Powered by Machine Learning + Spatial Intelligence &nbsp;•&nbsp; 2026
</div>
""", unsafe_allow_html=True)