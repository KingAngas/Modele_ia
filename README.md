# 🤖 Démonstrateur de modèles d'IA

Application web **Streamlit** qui regroupe **5 modèles de deep learning** indépendants. Une page d'accueil présente chaque modèle sous forme de carte cliquable, et chaque modèle dispose de sa propre page de test.

> Projet pédagogique réalisé par **DIT & AfriAI Solutions**. Les modèles ne remplacent ni un diagnostic médical, ni une décision commerciale ou académique.

## Les modèles

| N° | Modèle | Type de données | Tâche | Architecture |
|----|--------|-----------------|-------|--------------|
| 1 | Souscription bancaire | Tabulaire | Classification binaire (Bank Marketing) | Réseau dense (DNN) |
| 2 | Indice de performance étudiant | Tabulaire | Régression | Réseau dense (DNN) |
| 3 | CIFAR-10 | Image | Classification en 10 classes | CNN |
| 4 | Tuberculose | Radiographie thoracique | Classification binaire (Normal / Tuberculosis) | VGG16 (transfer learning) |
| 5 | Pneumonie | Radiographie thoracique | Classification binaire (Normal / Pneumonia) | VGG16 (transfer learning) |

## Structure du projet

```
.
├── app.py                  # Page d'accueil (cartes cliquables)
├── utils.py                # Fonctions communes (chargement, en-tête, pied de page)
├── requirements.txt
├── pages/
│   ├── 1_Banque.py
│   ├── 2_Etudiant.py
│   ├── 3_CIFAR10.py
│   ├── 4_Tuberculose.py
│   └── 5_Pneumonie.py
└── models/                 # Fichiers des modèles, un sous-dossier par modèle
    ├── banque/             # DNN_model.h5, encoders.joblib, scaler.joblib, uniques.joblib
    ├── etudiant/           # DNN_model.h5, encoder.joblib, scaler.joblib
    ├── cifar10/            # cnn_model.h5
    ├── tuberculose/        # vgg16_model_finetuned.h5
    └── pneumonie/          # vgg16_pneumonia_finetuned.h5
```

## Installation et lancement

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-compte>/<votre-depot>.git
cd <votre-depot>

# 2. (Recommandé) créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

L'application s'ouvre sur `http://localhost:8501`.

## Fonctionnement

- **Chargement à la demande** : un modèle n'est chargé que lorsque sa page est ouverte, puis gardé en mémoire (`st.cache_resource`).
- **Modèles d'images** : l'image est convertie en RGB, redimensionnée (150×150 pour les radiographies, 32×32 pour CIFAR-10) puis normalisée (division par 255).
- **Modèles tabulaires** : les variables catégorielles sont encodées et les valeurs normalisées avec les mêmes `encoders` et `scaler` que ceux de l'entraînement.

## Limites

- Le modèle CIFAR-10 ne connaît que 10 catégories : tout autre objet sera classé dans l'une d'elles.
- Les modèles de radiographie sont des outils de démonstration, non validés cliniquement.
- Les fichiers `.joblib` doivent être lus avec une version de scikit-learn proche de celle utilisée à l'entraînement.

## Fichiers volumineux

Les fichiers `.h5` (en particulier les VGG16) peuvent dépasser la limite de 100 Mo de GitHub. Dans ce cas, utilisez [Git LFS](https://git-lfs.com) :

```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
```

## Technologies

Python · Streamlit · TensorFlow / Keras · scikit-learn · NumPy · Pillow

## Auteurs

DIT & AfriAI Solutions
