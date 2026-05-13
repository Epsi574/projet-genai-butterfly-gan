#!/bin/bash
# Script to download Butterfly dataset from Kaggle

echo "========================================"
echo "  Butterfly Dataset Setup"
echo "========================================"

if ! command -v kaggle &> /dev/null; then
    echo "ERROR: Kaggle CLI not found. Installing..."
    pip install kaggle
fi

if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "ERROR: kaggle.json not found"
    echo ""
    echo "To download data from Kaggle:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Click 'Create New API Token'"
    echo "3. Place kaggle.json in ~/.kaggle/"
    echo ""
    echo "   mkdir -p ~/.kaggle"
    echo "   mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "   chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

mkdir -p data

echo ""
echo "Downloading dataset..."
kaggle datasets download -d phucthaiv02/butterfly-image-classification -p data --unzip

echo ""
echo "Dataset downloaded to ./data/"
echo ""
echo "Expected structure:"
echo "  data/"
echo "  ├── Training_set.csv"
echo "  └── train/"
echo "      ├── image1.jpg"
echo "      ├── image2.jpg"
echo "      └── ..."
echo ""
