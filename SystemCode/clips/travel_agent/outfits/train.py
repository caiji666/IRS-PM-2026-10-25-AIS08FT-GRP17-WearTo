"""Reproducible normalized-feature training: python -m travel_agent.outfits.train."""

import argparse
import csv
import hashlib
import io
import json
import platform
import zipfile
from pathlib import Path

import numpy as np

CATEGORIES = ("OUTERWEAR", "TOP", "BOTTOM", "FOOTWEAR")
DIMENSION = 512
DEFAULT_DIR = Path("artifacts/outfits/normalized-v1")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("outfit-compatibility"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--reencode", action="store_true")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    import joblib
    import sklearn
    from PIL import Image, ImageOps
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from travel_agent.products.retrieval.m5 import FashionCLIPEncoder
    from travel_agent.products.retrieval.m5.encoder import MODEL_IDENTITY

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "normalized_embeddings.npz"
    feature_meta = args.output_dir / "embedding_manifest.json"
    archive = args.data_dir / "polyvore_selected_images.zip"
    if args.reencode:
        encoder = FashionCLIPEncoder(
            Path(".cache/huggingface"), offline=True, device=args.device, batch_size=32
        )
        vectors = {}
        with zipfile.ZipFile(archive) as source:
            names = sorted(n for n in source.namelist() if n.lower().endswith(".jpg"))
            for start in range(0, len(names), 32):
                batch = names[start : start + 32]
                images = []
                for name in batch:
                    with Image.open(io.BytesIO(source.read(name))) as image:
                        images.append(ImageOps.exif_transpose(image).convert("RGB").copy())
                for name, vector in zip(batch, encoder.encode_images(images)):
                    vectors[Path(name).stem] = np.asarray(vector.values, dtype=np.float32)
                print(
                    f"Encoded {min(start + 32, len(names))}/{len(names)} training images",
                    flush=True,
                )
        np.savez_compressed(cache, **vectors)
        feature_meta.write_text(
            json.dumps(
                {
                    "encoder_identity": encoder.identity,
                    "dimension": DIMENSION,
                    "normalization": "per-garment L2 unit norm; absent category is zero",
                    "archive_sha256": digest(archive),
                    "embeddings_sha256": digest(cache),
                    "image_count": len(vectors),
                    "device": args.device,
                },
                indent=2,
            )
            + "\n"
        )
    if not cache.exists() or not feature_meta.exists():
        parser.error("First run requires --reencode to produce pinned FashionCLIP unit vectors")
    metadata = json.loads(feature_meta.read_text())
    if (
        metadata["encoder_identity"] != MODEL_IDENTITY
        or digest(cache) != metadata["embeddings_sha256"]
    ):
        parser.error("Feature cache identity/hash mismatch; rerun with --reencode")
    with np.load(cache, allow_pickle=False) as source:
        vectors = {k: source[k] for k in source.files}
    if any(
        v.shape != (DIMENSION,)
        or not np.isfinite(v).all()
        or not np.isclose(np.linalg.norm(v), 1, atol=1e-5)
        for v in vectors.values()
    ):
        raise ValueError("Training features must be finite 512-dimensional unit vectors")
    category_path = args.data_dir / "product_category.json"
    csv_path = args.data_dir / "clean_outfits_compatibility_prediction.csv"
    categories = json.loads(category_path.read_text())
    rows = list(csv.DictReader(csv_path.open()))
    features, labels, product_sets = [], [], []
    for row in rows:
        slots = {c: np.zeros(DIMENSION, dtype=np.float32) for c in CATEGORIES}
        ids = json.loads(row["products"])
        seen = set()
        for pid in ids:
            category = categories[pid]
            if category not in slots or category in seen or pid not in vectors:
                raise ValueError("Invalid/duplicate training role or missing embedding")
            seen.add(category)
            slots[category] = vectors[pid]
        features.append(np.concatenate(list(slots.values())))
        labels.append(int(row["label"]))
        product_sets.append(set(ids))
    x, y = np.stack(features), np.asarray(labels)
    train, rest = train_test_split(np.arange(len(y)), test_size=0.3, random_state=42, stratify=y)
    val, test = train_test_split(rest, test_size=0.5, random_state=42, stratify=y[rest])
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(16,),
                    activation="relu",
                    alpha=0.1,
                    max_iter=300,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x[train], y[train])
    model_path = args.output_dir / "model.joblib"
    joblib.dump(model, model_path)
    train_ids = set().union(*(product_sets[i] for i in train))
    test_ids = set().union(*(product_sets[i] for i in test))
    evaluation = {
        "validation_accuracy": accuracy_score(y[val], model.predict(x[val])),
        "test_accuracy": accuracy_score(y[test], model.predict(x[test])),
        "test_report": classification_report(y[test], model.predict(x[test]), output_dict=True),
        "split_counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "train_test_shared_product_count": len(train_ids & test_ids),
        "limitations": [
            "Outfit-row split, not product-disjoint; shared products may inflate metrics.",
            "Offline Polyvore evaluation is not an evaluation on current eBay products.",
            "Compatibility probabilities are model scores, not calibrated guarantees.",
        ],
    }
    manifest = {
        **metadata,
        "schema_version": "outfits.model.1",
        "categories": list(CATEGORIES),
        "feature_dimension": DIMENSION * len(CATEGORIES),
        "model_sha256": digest(model_path),
        "model_version": "normalized-v1-" + digest(model_path)[:16],
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "training_csv_sha256": digest(csv_path),
        "categories_sha256": digest(category_path),
        "random_state": 42,
        "evaluation": evaluation,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "model_dir": str(args.output_dir),
                "model_version": manifest["model_version"],
                **evaluation,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
