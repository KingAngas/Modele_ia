import numpy as np
import streamlit as st

from utils import WARNING_EDU, footer, header, load_joblib, load_keras_model, setup_page

setup_page("Indice de performance", "🎓")
header("Prédiction de l'indice de performance", "🎓", WARNING_EDU)

D = "etudiant"
model = load_keras_model(D, "DNN_model.h5")
encoder = load_joblib(D, "encoder.joblib")
scaler = load_joblib(D, "scaler.joblib")

with st.form("form_etudiant"):
    hours = st.number_input("Heures d'étude (Hours_Studied)", 0.0, 24.0, 7.0)
    prev = st.number_input("Notes précédentes (Previous_Scores)", 0.0, 100.0, 70.0)
    extra = st.selectbox("Activités extrascolaires (Extracurricular_Activities)", encoder.classes_.tolist())
    sleep = st.number_input("Heures de sommeil (Sleep_Hours)", 0.0, 24.0, 7.0)
    papers = st.number_input("Sujets d'examen pratiqués (Sample_Question_Papers_Practiced)", min_value=0.0, value=5.0)
    submit = st.form_submit_button("Prédire", use_container_width=True)

if submit:
    extra_enc = encoder.transform([extra])[0]
    x = np.array([hours, prev, extra_enc, sleep, papers], dtype=float).reshape(1, -1)
    raw = float(model.predict(scaler.transform(x), verbose=0)[0][0])
    shown = min(max(raw, 0.0), 100.0)

    st.subheader("Résultat")
    st.metric("Performance Index prédit", f"{shown:.1f} / 100")
    st.progress(shown / 100)
    if shown != raw:
        st.caption(f"Valeur brute du modèle : {raw:.1f} (bornée entre 0 et 100 à l'affichage).")

footer("Réseau de neurones (DNN) — régression")
