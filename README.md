# Butterfly GAN Benchmark

Benchmark de GANs conditionnels pour la génération d'images de papillons en 64x64, basé sur le dataset Kaggle **Butterfly Image Classification**.

Ce projet implémente et compare trois architectures GAN :

- **CDCGAN** (Conditional DCGAN) - Loss BCE + BatchNorm
- **CWGAN-GP** (Conditional Wasserstein GAN) - Loss Wasserstein + Gradient Penalty
- **CSNGAN** (Conditional Spectral Normalization GAN) - Loss Hinge + Normalisation Spectrale

L'objectif est d'évaluer la qualité et la fidélité des images générées conditionnellement sur l'espèce de papillon, en utilisant des métriques quantitatives et des visualisations comparatives.

## Structure du projet

```
.
├── config.py              # Configuration globale (hyperparamètres, chemins)
├── models.py              # Architectures Générateur et Discriminateur
├── losses.py              # Fonctions de loss (BCE, Wasserstein, Hinge)
├── dataset.py             # Chargement du dataset et transformations
├── train.py               # Boucle d'entraînement et gestion des checkpoints
├── evaluation.py          # FID, Intra-class FID, Precision & Recall
├── visualization.py       # Génération de graphiques et grilles comparatives
├── main.py                # Script principal orchestrant le pipeline complet
├── requirements.txt       # Dépendances Python
├── download_data.sh       # Script de téléchargement du dataset Kaggle
└── .gitignore             # Exclusion data, outputs, checkpoints
```

## Installation

### 1. Cloner le dépôt

```bash
git clone <url-de-votre-repo>
cd projet-genai-butterfly-gan
```

### 2. Créer l'environnement virtuel

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Télécharger le dataset

Configurer l'API Kaggle :

```bash
# Récupérer kaggle.json depuis https://www.kaggle.com/account
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Télécharger le dataset (226MB compressé) :

```bash
bash download_data.sh
```

Structure attendue :
```
data/
├── Training_set.csv
└── train/
    ├── ADONIS/
    ├── AFRICAN GIANT SWALLOWTAIL/
    └── ...
```

## Utilisation

### Démarrage rapide

Lancer le benchmark complet (entraînement + évaluation) :

```bash
python main.py
```

Cela va :
1. Charger le dataset (6499 images, 75 espèces)
2. Entraîner CDCGAN pour 100 époques
3. Entraîner CWGAN-GP pour 100 époques
4. Entraîner CSNGAN pour 100 époques
5. Évaluer tous les modèles (FID, Intra-class FID, Precision & Recall)
6. Générer les visualisations comparatives

### Options d'entraînement

Modifier [config.py](config.py) pour personnaliser :

```python
N_EPOCHS = 100          # Nombre d'époques d'entraînement
BATCH_SIZE = 64         # Taille du batch
N_EVAL_IMG = 5000       # Images générées par classe pour le FID
TOP_K = None            # Espèces à conserver (None = toutes les 75)
LATENT_DIM = 128        # Dimension du vecteur de bruit
LR_G = 1e-4             # Learning rate du générateur
LR_D = 1e-4             # Learning rate du discriminateur
```

### Reprendre l'entraînement

L'entraînement reprend automatiquement depuis le dernier checkpoint si disponible :

```python
# Dans train.py
resume = True  # Mettre à False pour recommencer de zéro
```

## Sorties

Toutes les sorties sont sauvegardées dans `./outputs/` :

```
outputs/
├── checkpoints/
│   ├── CDCGAN_latest.pt
│   ├── CWGAN-GP_latest.pt
│   └── CSNGAN_latest.pt
├── generated_images/
│   └── (images d'évaluation)
└── results/
    ├── benchmark_results.csv      # Tableau des métriques
    ├── benchmark_report.json      # Rapport complet
    ├── comparative_grid.png       # Comparaison Réel vs Généré
    ├── losses_all.png             # Courbes d'entraînement
    └── radar_chart.png            # Comparaison des architectures
```

## Métriques d'évaluation

Le benchmark utilise trois métriques quantitatives :

1. **FID Global** - Fréchet Inception Distance entre distributions réelles et générées
2. **Intra-class FID** - FID calculé par espèce (moyenne sur les 10 espèces les plus représentées)
3. **Precision & Recall** - Métriques de couverture des distributions

## Références

Ce benchmark s'inspire de :
- *StudioGAN: A Taxonomy and Benchmark of GANs for Image Synthesis* (Kang et al., 2023)
