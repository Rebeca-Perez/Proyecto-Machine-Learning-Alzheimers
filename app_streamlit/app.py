import streamlit as st
import pickle
import pandas as pd

# ---------------- CONFIG ----------------
MODEL_PATH = "../models/modelo_final/06_modelo_RandomForest.pkl"

st.set_page_config(
    page_title="Predicción de Alzheimer",
    page_icon="🧠",
    layout="centered"
)
# ---------------- CARGA MODELO ----------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
    
# ---------------- INTERFAZ ----------------
st.title("🧠 Predicción de Alzheimer")
st.write(
    """
    Aplicación web para estimar el **riesgo de Alzheimer** a partir de variables clínicas.
    El modelo utilizado es un **Random Forest**, seleccionado por su alto recall y estabilidad.
    """
)
# ---------------- IMAGEN PRINCIPAL ----------------
st.image(
    "img/alzheimers.webp",
    use_container_width=True
)
st.divider()

# ---------------- INPUTS ----------------
mmse = st.slider(
    "MMSE – Deterioro cognitivo",
    min_value=0.0,
    max_value=30.0,
    value=15.0
)

functional = st.slider(
    "Evaluación Funcional",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

memory = st.selectbox(
    "¿Presenta quejas de memoria?",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Sí"
)

behavior = st.selectbox(
    "¿Presenta problemas de comportamiento?",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Sí"
)

adl = st.slider(
    "ADL – Actividades de la vida diaria",
    min_value=0.0,
    max_value=10.0,
    value=5.0
)

# ---------------- DATAFRAME DE ENTRADA ----------------
input_data = pd.DataFrame(
    [[mmse, functional, memory, behavior, adl]],
    columns=model.feature_names_in_
)

# ---------------- PREDICCIÓN ----------------
st.divider()

if st.button("🔍 Predecir diagnóstico"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(
            f"⚠️ **Riesgo de Alzheimer detectado**\n\n"
            f"Probabilidad estimada: **{prob:.2%}**"
        )
    else:
        st.success(
            f"✅ **Bajo riesgo de Alzheimer**\n\n"
            f"Probabilidad estimada: **{prob:.2%}**"
        )

    st.caption(
        "⚠️ Esta predicción es orientativa y no sustituye un diagnóstico médico profesional."
    )
