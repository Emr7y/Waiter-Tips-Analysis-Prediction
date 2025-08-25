# app.py — Streamlit Tip Prediction (DE) — loads only model.pkl
import os
import numpy as np
import streamlit as st
import joblib

st.set_page_config(page_title="Trinkgeld-Vorhersage", page_icon="💵", layout="centered")
st.title("🧮 Trinkgeld-Vorhersage (RandomForest) — Schnellstart")
st.write("Diese App lädt **nur** ein bereits trainiertes Modell (`model.pkl`) und sagt das Trinkgeld voraus. Kein Training beim Start.")

MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' wurde nicht gefunden. Bitte lege die Datei in denselben Ordner wie app.py.")
    return joblib.load(path)

# Feature mapping helper (muss zum Training passen)
SEX_MAP = {"Female": 0, "Male": 1}
SMOKER_MAP = {"No": 0, "Yes": 1}
DAY_MAP = {"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3}
TIME_MAP = {"Lunch": 0, "Dinner": 1}

def build_features(total_bill, sex, smoker, day, time, size):
    sex_val = SEX_MAP[sex]
    smoker_val = 1 if smoker == "Yes" else 0   # Achtung: Training nutzte (No=0, Yes=1)
    day_val = DAY_MAP[day]
    time_val = TIME_MAP[time]
    size = max(1, int(size))
    avg_bill = total_bill / size
    weekend = 1 if day_val in [2, 3] else 0
    # Feature-Reihenfolge muss exakt dem Training entsprechen:
    return np.array([[
        total_bill, sex_val, smoker_val, day_val, time_val,
        size, avg_bill, weekend
    ]], dtype=float)

try:
    model = load_model(MODEL_PATH)
    st.success("✅ Modell geladen: model.pkl")
except Exception as e:
    st.error(f"❌ Konnte das Modell nicht laden: {e}")
    st.stop()

# UI
col1, col2 = st.columns(2)
with col1:
    total_bill = st.slider("Gesamtrechnung (total_bill)", 5.0, 1000.0, 150.0, step=1.0)
    sex = st.radio("Geschlecht (sex)", ["Female", "Male"], horizontal=True)
    smoker = st.radio("Raucher? (smoker)", ["Yes", "No"], horizontal=True)
with col2:
    day = st.radio("Wochentag (day)", ["Thur", "Fri", "Sat", "Sun"], horizontal=True)
    time = st.radio("Mahlzeit (time)", ["Lunch", "Dinner"], horizontal=True)
    size = st.slider("Personenanzahl (size)", 1, 20, 2, step=1)

if st.button("💡 Trinkgeld schätzen"):
    feats = build_features(total_bill, sex, smoker, day, time, size)
    pred = float(model.predict(feats)[0])
    st.success(f"💵 Erwartetes Trinkgeld: **{pred:.2f}** (Einheit wie in den Daten)")

st.markdown("---")
st.caption("Made with ❤️ by Emr7y | Modell: RandomForest | Datenquelle: tip.xlsx / Notebook-Export (model.pkl)")