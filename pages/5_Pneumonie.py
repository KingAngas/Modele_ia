from utils import render_xray_page

render_xray_page(
    title="Pneumonie", icon="🩻",
    folder="pneumonie", filename="vgg16_pneumonia_finetuned.h5",  # adapter au vrai nom du fichier
    positive_label="Pneumonia",
    footer_text="Modèle VGG16 (transfer learning)",
)
