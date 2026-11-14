import numpy as np
import streamlit as st
from PIL import Image

from utils import footer, header, load_keras_model, setup_page

setup_page("CIFAR-10", "🖼️")
header("Classificateur CIFAR-10", "🖼️")

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
FR = ["avion", "automobile", "oiseau", "chat", "cerf", "chien", "grenouille", "cheval", "bateau", "camion"]

st.write("Téléversez une image pour prédire sa classe parmi les 10 catégories CIFAR-10.")
st.info("Le modèle ne connaît que ces 10 catégories : un autre objet sera quand même classé dans l'une d'elles.")

model = load_keras_model("cifar10", "cnn_model.h5")
uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("Aucune image téléversée pour le moment.")
else:
    image = Image.open(uploaded)
    col_img, col_res = st.columns(2)
    with col_img:
        st.image(image, caption="Image téléversée")
    with col_res:
        with st.spinner("Analyse en cours..."):
            arr = np.array(image.convert("RGB").resize((32, 32))) / 255.0
            preds = model.predict(arr.reshape(1, 32, 32, 3), verbose=0)[0]
        top = np.argsort(preds)[::-1][:4]
        st.subheader("Résultat")
        st.success(f"**{CLASSES[top[0]]}** ({FR[top[0]]}) : {preds[top[0]]:.1%}")
        for i in top:
            st.write(f"{CLASSES[i]} ({FR[i]}) : {preds[i]:.1%}")
            st.progress(float(preds[i]))

footer("CNN entraîné sur CIFAR-10")
