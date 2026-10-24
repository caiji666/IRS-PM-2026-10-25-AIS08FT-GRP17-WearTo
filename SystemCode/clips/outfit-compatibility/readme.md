# Outfit Compatibility Prediction

## File Descriptions

| File | Description |
|---|---|
| `fashion_compatibility_prediction.txt` | Original PolyVore dataset containing outfit combinations and binary compatibility labels. |
| `pre-processing.ipynb` | Preprocesses the dataset by filtering clothing categories, matching product images, and removing invalid outfits. |
| `clean_outfits_compatibility_prediction.csv` | Cleaned outfit dataset containing product combinations and compatibility labels for model training. |
| `product_category.json` | Maps product IDs to their standardized clothing categories. |
| `polyvore_selected_images.zip` | Stores product images selected during preprocessing. |
| `image_embedding.ipynb` | Extracts unique products from the cleaned dataset and generates FashionCLIP image embeddings. |
| `fashionclip_embeddings.npz` | Stores the extracted 512-dimensional FashionCLIP embeddings for reuse during model training. |
| `model_training.ipynb` | Constructs outfit embeddings, trains the MLP compatibility classifier, and evaluates model performance. |
| `outfits_compatibility_model.pkl` | Saved scikit-learn Pipeline containing the trained StandardScaler and MLPClassifier. |
| `result.json` | Input JSON provided by student 3 containing Top-K candidate products and their embeddings for outfit generation and scoring. |
| `compatibility_service.py` | Generates valid outfit combinations, calculates compatibility scores using the trained model, ranks outfits, and returns a JSON-formatted string. |
| `readme.md` | Documents the complete outfit compatibility pipeline and its associated files. |

## 1. Data Preprocessing

The preprocessing stage prepares the **Re-PolyVore dataset** for outfit compatibility prediction by filtering clothing categories, cleaning outfit combinations, and extracting the corresponding product images.

### 1.1 Load Compatibility Labels

Read `fashion_compatibility_prediction.txt`, where each record contains a binary compatibility label followed by product IDs.

The records are converted into a structured table containing:

- `outfit_id`: Unique outfit identifier.
- `label`: Binary compatibility label (`0` = incompatible, `1` = compatible).
- `products`: List of product IDs belonging to the outfit.

### 1.2 Define Garment Categories

Only five original garment categories are retained and mapped into four standardized categories:

| Original Category | Standardized Category |
|---|---|
| `outwear` | `OUTERWEAR` |
| `top` | `TOP` |
| `pants` | `BOTTOM` |
| `skirt` | `BOTTOM` |
| `shoes` | `FOOTWEAR` |

The category order is fixed as:

`OUTERWEAR → TOP → BOTTOM → FOOTWEAR`

### 1.3 Match Products with Images

Search `Re-PolyVore.zip` for JPG images belonging to the selected garment categories.

Match image filenames with the product IDs extracted from the compatibility labels, and build mappings between product IDs, categories, and image paths.

Products without matching images in the selected categories are excluded.

### 1.4 Clean Outfit Combinations

Apply the following preprocessing rules:

1. Remove products without matching images in the selected categories.
2. Sort products according to the predefined category order.
3. Remove outfits containing fewer than two remaining products.
4. Remove outfits containing multiple products from the same standardized category.

The original compatibility labels are preserved for all retained outfits.

### 1.5 Export Preprocessed Data

The preprocessing stage produces three outputs:

| Output | Description |
|---|---|
| `clean_outfits_compatibility_prediction.csv` | Cleaned outfits with IDs, compatibility labels, and product lists |
| `product_category.json` | Mapping from retained product IDs to standardized categories |
| `polyvore_selected_images.zip` | JPG images corresponding to products in the cleaned dataset |

These outputs are used in the subsequent embedding extraction and model training stages.

## 2. Image Embedding Extraction

Extract all unique product IDs from `clean_outfits_compatibility_prediction.csv` to avoid processing duplicate products.

For each unique product, use its corresponding image and FashionCLIP to generate a 512-dimensional image embedding.

The extracted embeddings are saved in `fashionclip_embeddings.npz` for reuse in the subsequent model training stage, avoiding repeated image encoding.

## 3. Model Training

### 3.1 Outfit Embedding Construction

Load the saved FashionCLIP embeddings, product categories, and cleaned outfit dataset.

Construct a fixed **2048-dimensional outfit embedding** by concatenating four 512-dimensional product embeddings in the following order:

`OUTERWEAR → TOP → BOTTOM → FOOTWEAR`

Missing categories are filled with zero vectors to maintain a consistent input dimension.

### 3.2 Model Training

Split the dataset into **70% training, 15% validation, and 15% testing**, using stratified sampling.

Apply `StandardScaler` to normalize the outfit embeddings, then train an `MLPClassifier` to predict binary outfit compatibility:

- `0`: Incompatible
- `1`: Compatible

The MLP uses one hidden layer with 16 neurons, ReLU activation, L2 regularization (`alpha=0.1`), and early stopping.

### 3.3 Model Evaluation and Saving

Evaluate the trained model on the validation and test sets using accuracy and classification reports (precision, recall, and F1-score).

Finally, save the fitted scaler and MLP classifier together as a scikit-learn Pipeline:

`outfits_compatibility_model.pkl`

This Pipeline is loaded by the compatibility service to calculate outfit compatibility scores without requiring separate feature scaling.

## 4. Outfit Compatibility Scoring and Ranking

### 4.1 Generate Outfit Combinations

Read the Top-K candidate products provided by student3.

Generate all possible outfit combinations using the following rules:

- `TOP`, `BOTTOM`, and `FOOTWEAR` are mandatory.
- `OUTERWEAR` is optional.
- Each category contains at most one product.

If each category contains 5 candidate products, a total of **750 valid outfits** are generated.

### 4.2 Calculate Compatibility Scores

For each generated outfit, concatenate the FashionCLIP embeddings in the predefined category order to construct a **2048-dimensional feature vector**. Missing `OUTERWEAR` embeddings are replaced with zero vectors.

Load the trained `outfits_compatibility_model.pkl` Pipeline and use `predict_proba()` to calculate the probability of class `1` (Compatible).

This probability is used as the outfit's `compatibility_score`.

### 4.3 Rank and Return Results

Sort all outfits by `compatibility_score` in descending order and assign rank-based IDs (`O0001`, `O0002`, etc.).

Return a JSON string containing:

- `outfit_id`: Rank-based outfit identifier.
- `compatibility_score`: Predicted probability of compatibility.
- `products`: Complete original product information from `result.json`, including embeddings and other fields.

The final output is a JSON-formatted string containing all outfit combinations ranked by compatibility score in descending order, with the highest-scoring outfit appearing first.

## Notes

1. The original image archive is not included in this repository. Product embeddings are provided in `fashionclip_embeddings.npz`.

2. To use the trained outfit compatibility model, import `compatibility_service.py` and call the `run()` function:

```python
import compatibility_service as service

output = service.run()
```
## Dynamic project integration

The standalone fixed-file entry was replaced by a facade over `travel_agent.outfits.service.run(payload)`. Use [the backend integration guide](../docs/outfit-integration.md) for the current API and commands. The original raw-vector model and training notebooks are retained as historical training artifacts. Production-style backend inference now uses a separately trained, checksum-validated model with the same pinned normalized FashionCLIP feature space as Student 3. No implicit `result.json` fallback remains.
