"""Keras (TensorFlow) neural baseline for IT ticket classification.

Architecture: TextVectorization -> Embedding -> Conv1D -> GlobalMaxPooling -> Dense (softmax), trained on the
same train/val split as the TF-IDF and DistilBERT models so all three are benchmarked on the same test set.
The TextVectorization layer is part of the saved model, so it predicts directly from raw text.

Usage: python -m ticketlens.train_keras --mode full
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from ticketlens.config import load_config
from ticketlens.data import get_dataset_splits

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ticketlens.train_keras")

KERAS_MODEL_FILENAME = "model.keras"


def _as_model_input(texts: List[str]):
    """Keras string models take a (batch, 1) string tensor."""
    import tensorflow as tf

    return tf.constant([[t] for t in texts], dtype=tf.string)


def build_keras_model(train_texts: List[str], num_classes: int, k_cfg: Dict[str, Any], seed: int):
    """Build and compile the end-to-end (raw text in, class probabilities out) Keras model."""
    import tensorflow as tf
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    max_tokens = int(k_cfg.get("max_tokens", 30000))
    seq_len = int(k_cfg.get("sequence_length", 256))

    vectorize = keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=seq_len,
        name="text_vectorization",
    )
    vectorize.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(512))

    inputs = keras.Input(shape=(1,), dtype="string", name="text")
    x = vectorize(inputs)
    x = keras.layers.Embedding(max_tokens, int(k_cfg.get("embedding_dim", 128)), name="embedding")(x)
    x = keras.layers.Dropout(float(k_cfg.get("dropout", 0.3)))(x)
    x = keras.layers.Conv1D(int(k_cfg.get("conv_filters", 128)), int(k_cfg.get("kernel_size", 5)), activation="relu")(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dense(int(k_cfg.get("dense_units", 64)), activation="relu")(x)
    x = keras.layers.Dropout(float(k_cfg.get("dropout", 0.3)))(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="probabilities")(x)

    model = keras.Model(inputs, outputs, name="ticketlens_keras_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(float(k_cfg.get("learning_rate", 1e-3))),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_keras_model(model_dir: Path):
    from tensorflow import keras

    return keras.models.load_model(model_dir / KERAS_MODEL_FILENAME)


def keras_predict_proba(model, texts: List[str], batch_size: int = 256) -> np.ndarray:
    return np.asarray(model.predict(_as_model_input(texts), batch_size=batch_size, verbose=0))


def train_keras_model(
    cfg: Optional[Dict[str, Any]] = None,
    mode: str = "full",
    use_fixture: bool = False,
) -> Dict[str, Any]:
    """Train the Keras CNN baseline, save it to artifacts/models/keras and write reports/metrics_keras.json."""
    from tensorflow import keras

    if cfg is None:
        cfg = load_config()
    k_cfg = cfg.get("keras", {})
    seed = int(cfg.get("random_seed", 42))

    train_df, val_df, test_df, label2id = get_dataset_splits(cfg, mode=mode, use_fixture=use_fixture)
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)
    train_texts, val_texts = train_df["text"].tolist(), val_df["text"].tolist()
    y_train, y_val = train_df["label"].to_numpy(), val_df["label"].to_numpy()

    model = build_keras_model(train_texts, num_classes, k_cfg, seed)
    model.summary(print_fn=logger.info)

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(k_cfg.get("early_stopping_patience", 2)),
            restore_best_weights=True,
        )
    ]
    start_time = time.time()
    history = model.fit(
        _as_model_input(train_texts),
        y_train,
        validation_data=(_as_model_input(val_texts), y_val),
        epochs=int(k_cfg.get("epochs", 10)),
        batch_size=int(k_cfg.get("batch_size", 64)),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    training_duration = round(time.time() - start_time, 3)

    val_probs = keras_predict_proba(model, val_texts)
    val_preds = val_probs.argmax(axis=1)
    target_names = [id2label[i] for i in range(num_classes)]
    val_acc = float(accuracy_score(y_val, val_preds))
    val_macro_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    val_weighted_f1 = float(f1_score(y_val, val_preds, average="weighted", zero_division=0))
    report = classification_report(
        y_val, val_preds, labels=list(range(num_classes)), target_names=target_names, output_dict=True, zero_division=0
    )

    latencies = []
    for text in val_texts[:50]:
        t0 = time.perf_counter()
        model(_as_model_input([text]), training=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    model_dir = Path(cfg["paths"]["model_dir"]) / "keras"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / KERAS_MODEL_FILENAME)
    with open(model_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    # Guard against save/load issues (e.g. TextVectorization vocabulary not persisted).
    reloaded = load_keras_model(model_dir)
    if not np.allclose(keras_predict_proba(reloaded, val_texts[:32]), val_probs[:32], atol=1e-5):
        raise RuntimeError("Reloaded Keras model predictions differ from the trained model.")

    metrics = {
        "model_type": "Keras CNN (TextVectorization + Embedding + Conv1D)",
        "framework": "TensorFlow / Keras",
        "mode": mode,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "classes": target_names,
        "epochs_run": len(history.history["loss"]),
        "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "training_duration_seconds": training_duration,
        "validation_accuracy": round(val_acc, 4),
        "validation_macro_f1": round(val_macro_f1, 4),
        "validation_weighted_f1": round(val_weighted_f1, 4),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "history": {k: [round(float(v), 4) for v in vals] for k, vals in history.history.items()},
        "classification_report": report,
    }
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics_keras.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Keras validation: Macro-F1=%.4f, Acc=%.4f (%d epochs, %.1fs)", val_macro_f1, val_acc, metrics["epochs_run"], training_duration)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Keras (TensorFlow) CNN baseline")
    parser.add_argument("--mode", choices=["quick", "full", "all"], default="full", help="Execution mode")
    parser.add_argument("--fixture", action="store_true", help="Force synthetic fixture usage")
    args = parser.parse_args()

    train_keras_model(mode=args.mode, use_fixture=args.fixture)
