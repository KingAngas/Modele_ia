import streamlit as st

from utils import FOOTER, setup_page

setup_page("Démonstrateur de modèles IA", "🤖")

MODELES = [
    ("🏦", "Souscription bancaire", "pages/1_Banque.py",
     "Prédit si un client va souscrire à un dépôt à terme (Bank Marketing)."),
    ("🎓", "Indice de performance étudiant", "pages/2_Etudiant.py",
     "Estime la note d'un étudiant selon ses habitudes de travail et de sommeil."),
    ("🖼️", "Classification CIFAR-10", "pages/3_CIFAR10.py",
     "Reconnaît 10 catégories d'objets et d'animaux sur une image (CNN)."),
    ("🫁", "Détection de la tuberculose", "pages/4_Tuberculose.py",
     "Estime la probabilité de tuberculose sur une radiographie thoracique (VGG16)."),
    ("🩻", "Détection de la pneumonie", "pages/5_Pneumonie.py",
     "Estime la probabilité de pneumonie sur une radiographie thoracique (VGG16)."),
]

st.title("🤖 Démonstrateur de modèles d'IA")
st.write("Choisissez un modèle à tester.")

for i in range(0, len(MODELES), 2):
    cols = st.columns(2)
    for col, (icon, titre, page, desc) in zip(cols, MODELES[i:i + 2]):
        with col.container(border=True):
            st.subheader(f"{icon} {titre}")
            st.write(desc)
            if st.button("Tester ce modèle", key=page, use_container_width=True):
                st.switch_page(page)

st.caption(FOOTER)
