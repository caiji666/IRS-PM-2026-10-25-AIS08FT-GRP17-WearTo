# Outfit Compatibility Prediction

## Overview

This module predicts outfit compatibility using FashionCLIP image embeddings and a Multi-Layer Perceptron (MLP) classifier.

Each outfit consists of multiple fashion products. Product embeddings are concatenated with zero-padding to construct fixed-length outfit representations for binary classification.

## Files

| File | Description |
|---|---|
| `clean_outfits_compatibility_prediction.csv` | Preprocessed outfit dataset |
| `fashion_compatibility_prediction.txt` | Original outfit compatibility labels |
| `fashionclip_embeddings.npz` | Cached product embeddings |
| `image_embedding.ipynb` | FashionCLIP embedding extraction |
| `model_training.ipynb` | MLP training and evaluation |
| `outfits_compatibility_mlp.pkl` | Trained MLP classifier |
| `pre-processing.ipynb` | Dataset preprocessing |

## Pipeline

1. **Product Extraction** (`fashion_compatibility_prediction.txt`, `pre-processing.ipynb`): Parse the original outfit compatibility dataset and extract unique product IDs.

2. **Image Retrieval** : Since the original Polyvore image URLs are no longer accessible, use images from a newer Polyvore dataset, where only a subset of the original products can be matched.

3. **Data Preprocessing** (`pre-processing.ipynb`, `clean_outfits_compatibility_prediction.csv`): Remove products without available images and retain outfits with at least 3 valid products, align with our core fashion categories of tops, bottoms, and footwear. Save the cleaned outfits to CSV.

4. **Image Embedding** (`image_embedding.ipynb`, `fashionclip_embeddings.npz`): Extract 512-dimensional image embeddings for all available products using FashionCLIP and cache the results.

5. **Model Training** (`model_training.ipynb`, `outfits_compatibility_mlp.pkl`): Apply zero-padding and concatenation to construct fixed-length outfit embeddings (8 × 512 = 4096 dimensions). Train a two-hidden-layer MLP (128, 64), evaluate its performance on validation and test sets, and save the trained model for compatibility prediction.

## Model Configuration

- Embedding model: FashionCLIP
- Embedding dimension: 512
- Maximum products per outfit: 8
- Outfit feature dimension: 4096
- Classifier: MLPClassifier
- Hidden layers: (128, 64)
- Output: Binary compatibility prediction (0 = Incompatible, 1 = Compatible)

## Inference

The trained MLP can generate compatibility scores using:

```python
import joblib

model = joblib.load("outfits_compatibility_mlp.pkl")

# X_new must be constructed and standardized
# using the same preprocessing as training.
scores = model.predict_proba(X_new)[:, 1]
```

Higher scores indicate higher predicted outfit compatibility.

## Notes

The original image archive is not included in this repository. Product embeddings are provided in `fashionclip_embeddings.npz`.