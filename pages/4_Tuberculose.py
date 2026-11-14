from utils import render_xray_page

render_xray_page(
    title="Tuberculose", icon="🫁",
    folder="tuberculose", filename="vgg16_model_finetuned.h5",
    positive_label="Tuberculosis",
    footer_text="Modèle VGG16 (transfer learning)",
)
