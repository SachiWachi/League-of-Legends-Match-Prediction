import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="LoL Match Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_assets():
    with open('champion_info.json', 'r', encoding='utf-8') as f:
        champ_data = json.load(f)
        
    champ_mapping = {}
    name_to_id = {}
    if 'data' in champ_data:
        for k, info in champ_data['data'].items():
            champ_mapping[info['id']] = info['name']
            name_to_id[info['name']] = info['id']
            
    early_model_data = None
    if os.path.exists("early_model.joblib"):
        early_model_data = joblib.load("early_model.joblib")
        
    df = None
    if os.path.exists("games.csv"):
        df = pd.read_csv("games.csv")
        
    return champ_mapping, name_to_id, early_model_data, df

champ_mapping, name_to_id, early_model_data, df = load_assets()

# Client-accurate UI CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* Main Background */
.stApp {
    background-color: #010a13 !important;
    background-image: radial-gradient(circle at 10% 20%, rgba(0, 40, 60, 0.2) 0%, transparent 40%),
                      radial-gradient(circle at 90% 80%, rgba(0, 40, 60, 0.15) 0%, transparent 40%) !important;
    color: #f0e6d2 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Cinzel', serif !important;
    color: #c89b3c !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    text-shadow: 0 0 10px rgba(200, 155, 60, 0.2) !important;
}

h4, h5, h6, p, span {
    font-family: 'Inter', sans-serif !important;
    color: #a0aab5 !important;
}

label {
    font-family: 'Inter', sans-serif !important;
    color: #a0aab5 !important;
    font-size: 11px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

/* Tabs like LoL Client */
button[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #a0aab5 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 10px 20px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #f0e6d2 !important;
    border-bottom: 2px solid #c89b3c !important;
}

/* Form inputs & Selectboxes */
div[data-baseweb="select"] {
    background-color: #05101a !important;
    border: 1px solid rgba(200, 155, 60, 0.3) !important;
    border-radius: 0px !important;
    transition: all 0.2s ease;
}

div[data-baseweb="select"]:hover {
    border: 1px solid #c89b3c !important;
    box-shadow: 0 0 8px rgba(200, 155, 60, 0.3) !important;
}

div[data-baseweb="select"] > div {
    background-color: transparent !important;
    color: #f0e6d2 !important;
}

/* Radio Buttons */
div[role="radiogroup"] {
    display: flex;
    gap: 15px;
    padding: 5px 0;
}

/* Big Gold Button */
.stButton > button {
    background: transparent !important;
    color: #c89b3c !important;
    font-family: 'Cinzel', serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border: 2px solid #c89b3c !important;
    border-radius: 0 !important;
    letter-spacing: 3px !important;
    padding: 15px 40px !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    position: relative;
    overflow: hidden;
    margin-top: 30px !important;
}

.stButton > button:hover {
    background: rgba(200, 155, 60, 0.1) !important;
    box-shadow: 0 0 20px rgba(200, 155, 60, 0.4) !important;
    color: #f0e6d2 !important;
}

/* Victory/Defeat Banner Style */
.result-banner {
    background: linear-gradient(90deg, transparent, rgba(5, 15, 25, 0.9), transparent);
    border-top: 1px solid #c89b3c;
    border-bottom: 1px solid #c89b3c;
    padding: 30px 0;
    text-align: center;
    margin: 40px 0;
}

/* Background for Team Columns */
div[data-testid="column"]:has(#blue-team-marker) {
    background: rgba(0, 40, 80, 0.3) !important;
    border: 1px solid rgba(0, 163, 224, 0.4) !important;
    border-top: 4px solid #00a3e0 !important;
    border-radius: 6px !important;
    padding: 20px !important;
    box-shadow: 0 4px 25px rgba(0, 90, 130, 0.2) !important;
}

div[data-testid="column"]:has(#red-team-marker) {
    background: rgba(60, 10, 10, 0.3) !important;
    border: 1px solid rgba(238, 51, 51, 0.4) !important;
    border-top: 4px solid #ee3333 !important;
    border-radius: 6px !important;
    padding: 20px !important;
    box-shadow: 0 4px 25px rgba(150, 0, 0, 0.2) !important;
}

.result-title {
    font-family: 'Cinzel', serif;
    font-size: 42px;
    letter-spacing: 10px;
    margin: 0 0 10px 0;
    text-shadow: 0 0 20px rgba(200, 155, 60, 0.5);
}

.stat-row {
    display: flex;
    justify-content: center;
    gap: 50px;
    margin-top: 20px;
}

.stat-box {
    text-align: center;
}

.stat-value {
    font-family: 'Inter', sans-serif;
    font-size: 24px;
    font-weight: 600;
    color: #f0e6d2;
}

.stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    letter-spacing: 1px;
    color: #a0aab5;
    text-transform: uppercase;
}

</style>
""", unsafe_allow_html=True)

# App Title mimicking LoL top-left UI
st.markdown("""
<div style='display: flex; align-items: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #334455;'>
    <div style='width: 40px; height: 40px; border: 2px solid #c89b3c; transform: rotate(45deg); margin-left: 20px; margin-right: 30px; display: flex; align-items: center; justify-content: center;'>
        <div style='width: 20px; height: 20px; background-color: #00a3e0; transform: rotate(-45deg);'></div>
    </div>
    <div>
        <h1 style='margin: 0; font-size: 28px; letter-spacing: 4px; color: #f0e6d2 !important;'>LOL MATCH PREDICTION</h1>
    </div>
</div>
""", unsafe_allow_html=True)

if early_model_data is None:
    st.error("Model missing. Run train script.")
    st.stop()

model = early_model_data['model']
scaler = early_model_data['scaler']
features = scaler.feature_names_in_

t1, t2, t3 = st.tabs(["PREDICTION", "DATA GRAPHS", "MODEL EVALUATION"])

with t1:
    col1, col2 = st.columns(2)
    champs = sorted(list(name_to_id.keys()))
    
    with col1:
        st.markdown("<span id='blue-team-marker'></span>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #00a3e0; border-bottom: 1px solid #005a82; padding-bottom: 10px; margin-bottom: 15px;'>TEAM BLUE</h3>", unsafe_allow_html=True)
        b1 = st.selectbox("TOP", champs, index=0, key="b1")
        b2 = st.selectbox("JUNGLE", champs, index=1, key="b2")
        b3 = st.selectbox("MID", champs, index=2, key="b3")
        b4 = st.selectbox("BOT", champs, index=3, key="b4")
        b5 = st.selectbox("SUPPORT", champs, index=4, key="b5")
        
    with col2:
        st.markdown("<span id='red-team-marker'></span>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #ee3333; border-bottom: 1px solid #9a0000; padding-bottom: 10px; margin-bottom: 15px;'>TEAM RED</h3>", unsafe_allow_html=True)
        r1 = st.selectbox("TOP", champs, index=5, key="r1")
        r2 = st.selectbox("JUNGLE", champs, index=6, key="r2")
        r3 = st.selectbox("MID", champs, index=7, key="r3")
        r4 = st.selectbox("BOT", champs, index=8, key="r4")
        r5 = st.selectbox("SUPPORT", champs, index=9, key="r5")
        
    st.markdown("<h3 style='border-bottom: 1px solid #334455; padding-bottom: 10px; margin-top: 40px; margin-bottom: 15px;'>COMBAT & OBJECTIVES</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<span style='color: #c89b3c; font-size: 11px; letter-spacing: 1px; font-weight: 600;'>FIRST BLOOD</span>", unsafe_allow_html=True)
        fb = st.radio("fb", ["None", "Blue", "Red"], index=0, label_visibility="collapsed")
    with c2:
        st.markdown("<span style='color: #c89b3c; font-size: 11px; letter-spacing: 1px; font-weight: 600;'>FIRST TOWER</span>", unsafe_allow_html=True)
        ft = st.radio("ft", ["None", "Blue", "Red"], index=0, label_visibility="collapsed")
    with c3:
        st.markdown("<span style='color: #c89b3c; font-size: 11px; letter-spacing: 1px; font-weight: 600;'>FIRST DRAGON</span>", unsafe_allow_html=True)
        fd = st.radio("fd", ["None", "Blue", "Red"], index=0, label_visibility="collapsed")
    with c4:
        st.markdown("<span style='color: #c89b3c; font-size: 11px; letter-spacing: 1px; font-weight: 600;'>FIRST HERALD</span>", unsafe_allow_html=True)
        fh = st.radio("fh", ["None", "Blue", "Red"], index=0, label_visibility="collapsed")
        
    def val(x): return 1 if x == "Blue" else -1 if x == "Red" else 0

    if st.button("RUN ANALYSIS"):
        vec = pd.DataFrame(0, index=[0], columns=features)
        vec.at[0, 'firstBlood'] = val(fb)
        vec.at[0, 'firstTower'] = val(ft)
        vec.at[0, 'firstDragon'] = val(fd)
        vec.at[0, 'firstRiftHerald'] = val(fh)
        
        for c in [b1,b2,b3,b4,b5]:
            col = f'champ_{name_to_id[c]}'
            if col in vec.columns: vec.at[0, col] = 1
                
        for c in [r1,r2,r3,r4,r5]:
            col = f'champ_{name_to_id[c]}'
            if col in vec.columns: vec.at[0, col] = -1
                
        prob = model.predict_proba(scaler.transform(vec))[0]
        p_blue = prob[1] * 100
        p_red = prob[0] * 100
        
        w_text = "BLUE VICTORY" if p_blue > p_red else "RED VICTORY"
        w_color = "#00a3e0" if p_blue > p_red else "#ee3333"
        
        # Single-line HTML string guarantees Streamlit will NOT render it as a <pre> code block.
        html_str = (
            f'<div class="result-banner">'
            f'<h2 class="result-title" style="color: {w_color};">{w_text}</h2>'
            f'<div class="stat-row">'
            f'<div class="stat-box"><div class="stat-value" style="color: #00a3e0;">{p_blue:.1f}%</div><div class="stat-label">Blue Probability</div></div>'
            f'<div class="stat-box"><div class="stat-value" style="color: #ee3333;">{p_red:.1f}%</div><div class="stat-label">Red Probability</div></div>'
            f'<div class="stat-box"><div class="stat-value" style="color: #c89b3c;">HIGH</div><div class="stat-label">Confidence</div></div>'
            f'</div></div>'
        )
        st.markdown(html_str, unsafe_allow_html=True)

with t2:
    st.markdown("<h3 style='border-bottom: 1px solid #334455; padding-bottom: 10px; margin-bottom: 15px;'>DATA GRAPHS</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("plots/win_distribution.png"): st.image("plots/win_distribution.png")
        if os.path.exists("plots/game_duration.png"): st.image("plots/game_duration.png")
    with c2:
        if os.path.exists("plots/objective_win_rates.png"): st.image("plots/objective_win_rates.png")

with t3:
    st.markdown("<h3 style='border-bottom: 1px solid #334455; padding-bottom: 10px; margin-bottom: 15px;'>MODEL EVALUATION</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 13px; color: #a0aab5;'>Data leakage comparison.</p>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<span style='color: #c89b3c; font-size: 14px; letter-spacing: 2px;'>EARLY GAME MODEL</span>", unsafe_allow_html=True)
        if os.path.exists("plots/early-game_confusion_matrix.png"): st.image("plots/early-game_confusion_matrix.png", caption="Confusion Matrix")
        if os.path.exists("plots/early-game_feature_importance.png"): st.image("plots/early-game_feature_importance.png", caption="Feature Importance")
    with c2:
        st.markdown("<span style='color: #c89b3c; font-size: 14px; letter-spacing: 2px;'>FULL MATCH MODEL (DATA LEAKAGE)</span>", unsafe_allow_html=True)
        if os.path.exists("plots/full-match_confusion_matrix.png"): st.image("plots/full-match_confusion_matrix.png", caption="Confusion Matrix")
        if os.path.exists("plots/full-match_feature_importance.png"): st.image("plots/full-match_feature_importance.png", caption="Feature Importance")
