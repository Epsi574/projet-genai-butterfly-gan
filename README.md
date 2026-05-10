# Projet GenAI - Butterfly GAN Benchmark

Benchmark de GANs conditionnels pour la génération d'images de papillons en `64x64`, basé sur le dataset Kaggle **Butterfly Image Classification**.

Le notebook compare trois architectures:

- `CDCGAN`
- `CWGAN-GP`
- `CSNGAN`

L'objectif est d'évaluer la qualité et la fidélité des images générées selon l'espèce conditionnelle, avec des métriques quantitatives et des visualisations comparatives.

## Contenu du notebook

1. Vérification GPU et installation des dépendances
2. Téléchargement du dataset depuis Kaggle
3. Prétraitement des données et chargement des classes
4. Définition d'un backbone commun pour les architectures GAN
5. Entraînement des trois modèles
6. Évaluation quantitative:
   - `FID` global
   - `Intra-class FID` par espèce
   - `Precision` et `Recall`
7. Analyse qualitative:
   - grilles d'images réelles vs générées
   - courbes de loss
   - tableau comparatif final

## Démarrage

Le projet est pensé pour être exécuté dans **Google Colab** avec GPU activé.

### Prérequis

- un compte Kaggle
- un fichier `kaggle.json`
- un GPU activé dans Colab

### Entrées du notebook

- `kaggle.json` pour l'accès au dataset
- le dataset `phucthaiv02/butterfly-image-classification`

### Sorties attendues

- modèles entraînés sauvegardés sur Google Drive
- images générées par époque
- résultats dans `results/`
- export `CSV` et `JSON` de synthèse

## Arborescence des résultats

Le notebook crée notamment:

- `checkpoints/` pour les modèles
- `generated_images/` pour les visuels générés
- `results/` pour les métriques et graphiques

## Référence

Le benchmark s'inspire de *StudioGAN: A Taxonomy and Benchmark of GANs for Image Synthesis* (Kang et al., 2023).
