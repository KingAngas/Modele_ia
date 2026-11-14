"""Fonctions communes à toutes les pages."""
import os

import numpy as np
import streamlit as st
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FOOTER = "DIT & AfriAI Solutions"

WARNING_MEDICAL = "Outil pédagogique : il ne remplace pas un diagnostic médical."
WARNING_BANK = "Outil pédagogique : il ne doit pas servir à décider seul d'une offre commerciale."
WARNING_EDU = "Outil pédagogique : l'estimation ne remplace pas une évaluation réelle."


def setup_page(title, icon):
    """À appeler en premier dans chaque page."""
    st.set_page_config(page_title=title, page_icon=icon, layout="centered")


def header(title, icon, warning=None):
    st.page_link("app.py", label="Retour à l'accueil", icon="🏠")
    st.title(f"{icon} {title}")
    if warning:
        st.warning(warning)


def footer(text):
    st.caption(f"{FOOTER} · {text}")


def model_file(folder, filename):
    path = os.path.join(MODELS_DIR, folder, filename)
    if not os.path.exists(path):
        st.error(f"Fichier introuvable : models/{folder}/{filename}")
        st.stop()
    return path


@st.cache_resource
def _load_keras(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


@st.cache_resource
def _load_joblib(path):
    import joblib
    return joblib.load(path)


def load_keras_model(folder, filename):
    """Charge un .h5 une seule fois (mis en cache)."""
    return _load_keras(model_file(folder, filename))


def load_joblib(folder, filename):
    return _load_joblib(model_file(folder, filename))


def render_xray_page(title, icon, folder, filename, positive_label, footer_text):
    """Page commune aux radiographies (tuberculose, pneumonie) : même prétraitement."""
    classes = ["Normal", positive_label]
    setup_page(title, icon)
    header(f"Détection : {title}", icon, WARNING_MEDICAL)
    st.write("Téléversez une radiographie thoracique pour estimer la probabilité.")

    model = load_keras_model(folder, filename)
    uploaded = st.file_uploader("Radiographie thoracique", type=["png", "jpg", "jpeg"])

    if uploaded is None:
        st.info("Aucune image téléversée pour le moment.")
        footer(footer_text)
        return

    image = Image.open(uploaded)
    col_img, col_res = st.columns(2)
    with col_img:
        st.image(image, caption="Image téléversée")
    with col_res:
        with st.spinner("Analyse en cours..."):
            arr = np.asarray(image.convert("RGB").resize((150, 150), Image.NEAREST), dtype=np.float32) / 255.0
            p = float(model.predict(arr[None, ...], verbose=0)[0][0])
        probas = {classes[1]: p, classes[0]: 1.0 - p}
        pred = max(probas, key=probas.get)

        st.subheader("Résultat")
        (st.error if pred == positive_label else st.success)(f"**{pred}** ({probas[pred]:.1%})")
        for c, v in sorted(probas.items(), key=lambda x: x[1], reverse=True):
            st.write(f"{c} : {v:.1%}")
            st.progress(v)
    footer(footer_text)
