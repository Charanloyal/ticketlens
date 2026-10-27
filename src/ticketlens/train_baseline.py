"""TF-IDF + Logistic Regression baseline model training and persistence."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from ticketlens.config import load_config
from ticketlens.data import build_baseline_corpus, load_baseline_corpus, short_text_variants  # noqa: F401 (re-export)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ticketlens.train_baseline")


def build_vectorizer(b_cfg: Dict[str, Any]) -> TfidfVectorizer:
    """Word n-gram TF-IDF (float32 keeps peak training memory low enough for 512 MB build machines)."""
    return TfidfVectorizer(
        ngram_range=tuple(b_cfg["ngram_range"]),
        max_features=b_cfg["max_features"],
        min_df=b_cfg.get("min_df", 2),
        sublinear_tf=b_cfg.get("sublinear_tf", True),
        dtype=np.float32,
    )


def train_baseline_model(
    cfg: Optional[Dict[str, Any]] = None,
    mode: str = "quick",
    use_fixture: bool = False,
    require_dataset: bool = False,
    from_corpus: bool = False,
    model_name: str = "baseline",
    augment: Optional[bool] = None,
) -> Dict[str, Any]:
    """Train TF-IDF + Logistic Regression baseline pipeline and save artifacts.

    With from_corpus=True the pre-built corpus from 'python -m ticketlens.data --prepare-corpus' is used, so
    this process never holds the raw dataset (keeps peak memory low on Render build machines).
    """
    if cfg is None:
        cfg = load_config()
    if augment is not None:
        cfg = {**cfg, "baseline": {**cfg["baseline"], "augment_short_text": augment}}

    if from_corpus:
        logger.info("Loading pre-built training corpus (mode=%s)...", mode)
        corpus = load_baseline_corpus(cfg, expected_mode=mode)
    else:
        logger.info("Preparing data splits for baseline training (mode=%s)...", mode)
        corpus = build_baseline_corpus(cfg, mode=mode, use_fixture=use_fixture, require_dataset=require_dataset)

    label2id = corpus["label2id"]
    id2label = {v: k for k, v in label2id.items()}
    val_texts = corpus["val_text"]

    b_cfg = cfg["baseline"]
    logger.info(
        "Configuring TF-IDF (ngram_range=%s, max_features=%s, sublinear_tf=%s)...",
        b_cfg["ngram_range"],
        b_cfg["max_features"],
        b_cfg["sublinear_tf"],
    )

    vectorizer = build_vectorizer(b_cfg)

    # Fit vectorizer strictly on training set
    start_time = time.time()
    X_train = vectorizer.fit_transform(corpus.pop("train_text"))
    y_train = np.array(corpus.pop("train_label"))

    classifier = LogisticRegression(
        C=b_cfg.get("c_param", 1.0),
        max_iter=b_cfg.get("max_iter", 1000),
        class_weight=b_cfg.get("class_weight", "balanced"),
        random_state=cfg.get("random_seed", 42),
    )

    logger.info("Fitting Logistic Regression classifier on %d training samples...", X_train.shape[0])
    classifier.fit(X_train, y_train)
    del X_train, y_train
    training_duration_seconds = round(time.time() - start_time, 4)
    logger.info("Baseline training completed in %.3f seconds.", training_duration_seconds)

    # Validation evaluation
    X_val = vectorizer.transform(val_texts)
    y_val = np.array(corpus["val_label"])
    val_preds = classifier.predict(X_val)
    val_probs = classifier.predict_proba(X_val)

    val_acc = float(accuracy_score(y_val, val_preds))
    val_macro_f1 = float(f1_score(y_val, val_preds, average="macro", zero_division=0))
    val_weighted_f1 = float(f1_score(y_val, val_preds, average="weighted", zero_division=0))

    target_names = [id2label[i] for i in range(len(label2id))]
    report_dict = classification_report(
        y_val,
        val_preds,
        labels=list(range(len(label2id))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # Benchmark validation latency
    latency_trials = []
    sample_texts = val_texts[:50]
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
        "train_samples": corpus["n_train_tickets"],
        "augmented_samples": corpus["n_augmented"],
        "val_samples": len(val_texts),
        "test_samples": corpus["test_size"],
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
    model_dir = Path(cfg["paths"]["model_dir"]) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    # `stop_words_` only lists pruned terms (introspection only); dropping it shrinks the artifact.
    vectorizer.stop_words_ = None
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib", compress=3)
    joblib.dump(classifier, model_dir / "logistic_regression.joblib", compress=3)
    with open(model_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / f"metrics_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Baseline validation: Macro-F1=%.4f, Acc=%.4f, P50 Latency=%.2f ms", val_macro_f1, val_acc, p50_latency_ms)
    logger.info("Saved baseline artifacts to %s", model_dir)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--mode", choices=["quick", "full", "all"], default="quick", help="Execution mode ('all' trains on every row)")
    parser.add_argument("--fixture", action="store_true", help="Force synthetic fixture usage")
    parser.add_argument(
        "--require-dataset",
        action="store_true",
        help="Fail instead of falling back to the synthetic fixture when the real dataset can't be loaded (production builds)",
    )
    parser.add_argument(
        "--model-name",
        default="baseline",
        help="Artifact sub-directory under artifacts/models ('baseline' is the served model; e.g. 'tfidf_benchmark' for benchmarking)",
    )
    parser.add_argument("--no-augment", action="store_true", help="Disable short-text augmentation (plain TF-IDF benchmark)")
    parser.add_argument(
        "--from-corpus",
        action="store_true",
        help="Train from the corpus written by 'python -m ticketlens.data --prepare-corpus' (low-memory production build)",
    )
    args = parser.parse_args()

    train_baseline_model(
        mode=args.mode,
        use_fixture=args.fixture,
        require_dataset=args.require_dataset,
        from_corpus=args.from_corpus,
        model_name=args.model_name,
        augment=False if args.no_augment else None,
    )
