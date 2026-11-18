from utils import render_xray_page

render_xray_page(
    title="Pneumonie", icon="🩻",
    folder="pneumonie", filename="densenet_model_finetuned.h5",  # adapter au vrai nom du fichier
    positive_label="Pneumonia",
    footer_text="Modèle densenet (transfer learning)",
)
