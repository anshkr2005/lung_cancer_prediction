"""
LungScan AI - Flask Backend
CNN Transfer Learning with Xception for Lung Cancer Classification
"""

import warnings
warnings.filterwarnings('ignore')

import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image as keras_image

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
IMAGE_SIZE     = (350, 350)
OUTPUT_SIZE    = 4
MODEL_PATH     = "trained_lung_cancer_model.h5"
WEIGHTS_PATH   = "best_model.weights.h5"

CLASS_LABELS = [
    "Normal",
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Squamous Cell Carcinoma"
]

CLASS_DETAILS = {
    "Normal":                    "No malignancy detected",
    "Adenocarcinoma":            "Left lower lobe · T2 N0 M0 · Stage Ib",
    "Large Cell Carcinoma":      "Left hilum · T2 N2 M0 · Stage IIIa",
    "Squamous Cell Carcinoma":   "Left hilum · T1 N2 M0 · Stage IIIa"
}

# ─────────────────────────────────────────────
# Build / Load Model
# ─────────────────────────────────────────────
def build_model():
    """Reconstruct the Xception transfer-learning model."""
    pretrained = tf.keras.applications.Xception(
        weights='imagenet',
        include_top=False,
        input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained.trainable = False

    model = Sequential([
        pretrained,
        GlobalAveragePooling2D(),
        Dense(OUTPUT_SIZE, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_or_build_model():
    """Load saved model if available, otherwise build fresh."""
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading model from {MODEL_PATH}")
        try:
            model = load_model(MODEL_PATH)
            print("[INFO] Full model loaded successfully.")
            return model
        except Exception as e:
            print(f"[WARN] Could not load full model: {e}. Rebuilding...")

    model = build_model()

    if os.path.exists(WEIGHTS_PATH):
        print(f"[INFO] Loading weights from {WEIGHTS_PATH}")
        try:
            model.load_weights(WEIGHTS_PATH)
            print("[INFO] Weights loaded successfully.")
        except Exception as e:
            print(f"[WARN] Could not load weights: {e}. Using ImageNet init.")
    else:
        print("[WARN] No saved weights found. Model uses ImageNet init only.")

    return model


# ─────────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────────
def preprocess_image(img_bytes: bytes) -> np.ndarray:
    """
    Load image from raw bytes, resize to IMAGE_SIZE,
    convert to RGB, rescale to [0, 1], and add batch dim.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)   # (1, 350, 350, 3)
    return img_array


def preprocess_from_path(img_path: str) -> np.ndarray:
    """Load and preprocess an image from a file path."""
    img = keras_image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
def predict(model, img_array: np.ndarray) -> dict:
    """
    Run model inference and return structured result.

    Returns
    -------
    {
        predicted_class : str,
        confidence      : float,   # 0–100
        detail          : str,
        probabilities   : [{"label": str, "probability": float}, ...]
    }
    """
    preds = model.predict(img_array, verbose=0)[0]   # shape: (4,)

    predicted_idx   = int(np.argmax(preds))
    predicted_label = CLASS_LABELS[predicted_idx]
    confidence      = float(preds[predicted_idx]) * 100

    probabilities = [
        {
            "label":       CLASS_LABELS[i],
            "probability": round(float(preds[i]) * 100, 2)
        }
        for i in range(OUTPUT_SIZE)
    ]
    # Sort descending
    probabilities.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "predicted_class": predicted_label,
        "confidence":      round(confidence, 2),
        "detail":          CLASS_DETAILS.get(predicted_label, ""),
        "probabilities":   probabilities
    }


# ─────────────────────────────────────────────
# Training (optional helper)
# ─────────────────────────────────────────────
def train_model(
    train_folder="dataset/train",
    val_folder="dataset/test",
    epochs=50,
    batch_size=8,
    steps_per_epoch=25,
    validation_steps=20,
    save_path=MODEL_PATH,
    weights_path=WEIGHTS_PATH,
):
    """
    Train the model from scratch using ImageDataGenerator.
    Call this function if you haven't trained yet.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import (
        ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
    )

    train_datagen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True)
    val_datagen   = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_folder,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        color_mode="rgb",
        class_mode="categorical"
    )
    val_gen = val_datagen.flow_from_directory(
        val_folder,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        color_mode="rgb",
        class_mode="categorical"
    )

    model = build_model()

    callbacks = [
        ReduceLROnPlateau(monitor='loss', patience=5, verbose=1,
                          factor=0.5, min_lr=1e-6),
        EarlyStopping(monitor='loss', min_delta=0, patience=6,
                      verbose=1, mode='auto'),
        ModelCheckpoint(filepath=weights_path, verbose=1,
                        save_best_only=True, save_weights_only=True),
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_gen,
        validation_steps=validation_steps
    )

    model.save(save_path)
    print(f"[INFO] Model saved to {save_path}")
    print(f"[INFO] Final train accuracy : {history.history['accuracy'][-1]:.4f}")
    print(f"[INFO] Final val accuracy   : {history.history['val_accuracy'][-1]:.4f}")

    return model, history


# ─────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────
app   = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend UI
model = None  # Lazy-loaded on first request


@app.before_request
def load_model_once():
    """Load model into memory on first request."""
    global model
    if model is None:
        print("[INFO] Loading model for the first time...")
        model = load_or_build_model()
        print("[INFO] Model ready.")


# ── Routes ────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health-check endpoint."""
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    POST /predict
    Body: multipart/form-data with field 'file' containing a CT scan image.

    Response JSON:
    {
        "predicted_class": "Adenocarcinoma",
        "confidence": 87.43,
        "detail": "Left lower lobe · T2 N0 M0 · Stage Ib",
        "probabilities": [
            {"label": "Adenocarcinoma", "probability": 87.43},
            ...
        ]
    }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use field name 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    allowed = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    try:
        img_bytes = file.read()
        img_array = preprocess_image(img_bytes)
        result    = predict(model, img_array)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/base64", methods=["POST"])
def predict_base64():
    """
    POST /predict/base64
    Body JSON: { "image": "<base64-encoded image string>" }
    Useful when calling from the frontend with FileReader.
    """
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "JSON body must contain 'image' key."}), 400

    try:
        # Strip data URL prefix if present (e.g. "data:image/png;base64,...")
        b64 = data["image"]
        if "," in b64:
            b64 = b64.split(",", 1)[1]

        img_bytes = base64.b64decode(b64)
        img_array = preprocess_image(img_bytes)
        result    = predict(model, img_array)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/classes", methods=["GET"])
def get_classes():
    """Return the list of class labels and their clinical details."""
    return jsonify([
        {"label": label, "detail": CLASS_DETAILS[label]}
        for label in CLASS_LABELS
    ])


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return model architecture metadata."""
    return jsonify({
        "architecture":  "Xception + GlobalAveragePooling2D + Dense(4, softmax)",
        "input_shape":   list(IMAGE_SIZE) + [3],
        "output_classes": OUTPUT_SIZE,
        "optimizer":     "adam",
        "loss":          "categorical_crossentropy",
        "base_trainable": False,
        "rescale":       "/ 255.0",
        "weights":       "imagenet"
    })


# ─────────────────────────────────────────────
# CLI: predict a single image from terminal
# ─────────────────────────────────────────────
def cli_predict(img_path: str):
    """Predict a single image from the command line."""
    print(f"\n[LungScan AI] Predicting: {img_path}")
    m = load_or_build_model()
    arr = preprocess_from_path(img_path)
    result = predict(m, arr)

    print(f"\n  Predicted class : {result['predicted_class']}")
    print(f"  Confidence      : {result['confidence']:.2f}%")
    print(f"  Detail          : {result['detail']}")
    print("\n  All probabilities:")
    for p in result["probabilities"]:
        bar = "█" * int(p["probability"] / 5)
        print(f"    {p['label']:<30} {p['probability']:6.2f}%  {bar}")
    print()


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        # Usage: python app.py path/to/image.png
        cli_predict(sys.argv[1])
    elif len(sys.argv) == 2 and sys.argv[1] == "train":
        # Usage: python app.py train
        print("[INFO] Starting training...")
        train_model()
    else:
        # Default: start Flask server
        print("[INFO] Starting LungScan AI server on http://localhost:5000")
        app.run(host="0.0.0.0", port=5000, debug=False)