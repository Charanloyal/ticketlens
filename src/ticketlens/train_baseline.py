"""TF-IDF + Logistic Regression baseline model training and persistence."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from ticketlens.config import load_config
from ticketlens.data import get_dataset_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ticketlens.train_baseline")


def train_baseline_model(
    cfg: Optional[Dict[str, Any]] = None,
    mode: str = "quick",
    use_fixture: bool = False,
) -> Dict[str, Any]:
    """Train TF-IDF + Logistic Regression baseline pipeline and save artifacts."""
    if cfg is None:
        cfg = load_config()

    logger.info("Preparing data splits for baseline training (mode=%s)...", mode)
    train_df, val_df, test_df, label2id = get_dataset_splits(cfg, mode=mode, use_fixture=use_fixture)
    id2label = {v: k for k, v in label2id.items()}

    b_cfg = cfg["baseline"]
    logger.info(
        "Configuring TF-IDF (ngram_range=%s, max_features=%d, sublinear_tf=%s)...",
        b_cfg["ngram_range"],
        b_cfg["max_features"],
        b_cfg["sublinear_tf"],
    )

    vectorizer = TfidfVectorizer(
        ngram_range=tuple(b_cfg["ngram_range"]),
        max_features=b_cfg["max_features"],
        min_df=b_cfg.get("min_df", 2),
        sublinear_tf=b_cfg.get("sublinear_tf", True),
    )

    # Fit vectorizer strictly on training set
    start_time = time.time()
    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"].values

    classifier = LogisticRegression(
        C=b_cfg.get("c_param", 1.0),
        max_iter=b_cfg.get("max_iter", 1000),
        class_weight=b_cfg.get("class_weight", "balanced"),
        random_state=cfg.get("random_seed", 42),
    )

    logger.info("Fitting Logistic Regression classifier on %d training samples...", len(train_df))
    classifier.fit(X_train, y_train)
    training_duration_seconds = round(time.time() - start_time, 4)
    logger.info("Baseline training completed in %.3f seconds.", training_duration_seconds)

    # Validation evaluation
    X_val = vectorizer.transform(val_df["text"])
    y_val = val_df["label"].values
    val_preds = classifier.predict(X_val)
    val_probs = classifier.predict_proba(X_val)

    val_acc = float(accuracy_score(y_val, val_preds))
    val_macro_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    val_weighted_f1 = float(f1_score(y_val, val_preds, average="weighted", zero_division=0))

    target_names = [id2label[i] for i in range(len(label2id))]
    report_dict = classification_report(y_val, val_preds, target_names=target_names, output_dict=True, zero_division=0)

    # Benchmark validation latency
    latency_trials = []
    sample_texts = val_df["text"].iloc[: min(50, len(val_df))].tolist()
    for text in sample_texts:
        t0 = time.perf_counter()
        vec = vectorizer.transform([text])
        _ = classifier.predict_proba(vec)
        latency_trials.append((time.perf_counter() - t0) * 1000)

    p50_latency_ms = round(float(np.percentile(latency_trials, 50)), 3)
    p95_latency_ms = round(float(np.percentile(latency_trials, 95)), 3)

    metrics = {
        "model_type": "TF-IDF + Logistic Regression",
        "mode": mode,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "classes": target_names,
        "training_duration_seconds": training_duration_seconds,
        "validation_accuracy": round(val_acc, 4),
        "validation_macro_f1": round(val_macro_f1, 4),
        "validation_weighted_f1": round(val_weighted_f1, 4),
        "latency_p50_ms": p50_latency_ms,
        "latency_p95_ms": p95_latency_ms,
        "classification_report": report_dict,
    }

    # Save artifacts
    model_dir = Path(cfg["paths"]["model_dir"]) / "baseline"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")
    joblib.dump(classifier, model_dir / "logistic_regression.joblib")
    with open(model_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics_baseline.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Baseline validation: Macro-F1=%.4f, Acc=%.4f, P50 Latency=%.2f ms", val_macro_f1, val_acc, p50_latency_ms)
    logger.info("Saved baseline artifacts to %s", model_dir)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick", help="Execution mode")
    parser.add_argument("--fixture", action="store_true", help="Force synthetic fixture usage")
    args = parser.parse_args()

    train_baseline_model(mode=args.mode, use_fixture=args.fixture)
