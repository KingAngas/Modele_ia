import numpy as np
import streamlit as st

from utils import WARNING_BANK, footer, header, load_joblib, load_keras_model, setup_page

setup_page("Souscription bancaire", "🏦")
header("Prédiction de souscription bancaire", "🏦", WARNING_BANK)

D = "banque"
model = load_keras_model(D, "DNN_model.h5")
encoders = load_joblib(D, "encoders.joblib")
scaler = load_joblib(D, "scaler.joblib")
uniques = load_joblib(D, "uniques.joblib")

with st.form("form_banque"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("age", 18, 100, 40)
        job = st.selectbox("job", uniques[0])
        marital = st.selectbox("marital", uniques[1])
        education = st.selectbox("education", uniques[2])
        default = st.selectbox("default", uniques[3])
        housing = st.selectbox("housing", uniques[4])
        loan = st.selectbox("loan", uniques[5])
        contact = st.selectbox("contact", uniques[6])
    with c2:
        month = st.selectbox("month", uniques[7])
        day_of_week = st.selectbox("day_of_week", uniques[8])
        duration = st.number_input("duration", min_value=0, value=0)
        campaign = st.number_input("campaign", min_value=0, value=1)
        pdays = st.number_input("pdays (999 = jamais contacté)", min_value=0, value=999)
        previous = st.number_input("previous", min_value=0, value=0)
        poutcome = st.selectbox("poutcome", uniques[9])
    submit = st.form_submit_button("Prédire", use_container_width=True)

if submit:
    cats = [job, marital, education, default, housing, loan, contact, month, day_of_week]
    enc = [encoders[i].transform([v])[0] for i, v in enumerate(cats)]
    poutcome_enc = encoders[9].transform([poutcome])[0]
    x = np.array([age, *enc[:9], duration, campaign, pdays, previous, poutcome_enc], dtype=float).reshape(1, -1)
    x = scaler.transform(x)
    p = float(model.predict(x, verbose=0)[0][0])

    st.subheader("Résultat")
    if p >= 0.5:
        st.success(f"**Souscrire** ({p:.1%})")
    else:
        st.error(f"**Pas souscrire** ({1 - p:.1%})")
    st.progress(p, text=f"Probabilité de souscription : {p:.1%}")

footer("Réseau de neurones (DNN) sur données tabulaires")
