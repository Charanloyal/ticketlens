"""Benchmark every trained model on the identical held-out test split, with precision/recall error analysis.

Models (each evaluated only if its artifacts and framework are available):
  - tfidf_benchmark : TF-IDF + Logistic Regression trained on the same subset as Keras and DistilBERT
  - keras           : Keras (TensorFlow) CNN baseline
  - transformer     : DistilBERT fine-tuned with LoRA (PyTorch + PEFT)
  - baseline        : the served TF-IDF model (all training data + short-text augmentation), with and
                      without the ITIL cue correction used in production

Outputs (reports/): model_comparison.json, metrics_test_<model>.json, evaluation_summary.md,
error_analysis.md and figures/confusion_matrix_<model>.png, model_comparison_f1.png, precision_coverage.png.

Usage: python -m ticketlens.evaluate
"""

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from ticketlens.config import get_device, get_project_root, load_config
from ticketlens.data import clean_and_redact_text, get_dataset_splits
from ticketlens.itil_rules import apply_itil_cues

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ticketlens.evaluate")

PredictProba = Callable[[List[str]], np.ndarray]

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
LENGTH_BUCKETS = [(0, 50, "< 50 words"), (50, 100, "50-99 words"), (100, 200, "100-199 words"), (200, 10**9, ">= 200 words")]

MODEL_TITLES = {
    "tfidf_benchmark": "TF-IDF + Logistic Regression",
    "keras": "Keras CNN (TensorFlow)",
    "transformer": "DistilBERT + LoRA (PyTorch/PEFT)",
    "baseline": "Served TF-IDF (all data, model only)",
    "baseline_itil": "Served TF-IDF + ITIL cues (production)",
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], title: str, save_path: Path) -> None:
    """Plot and save confusion matrix figure."""
    import seaborn as sns

    plt = _plt()
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=13, pad=12)
    plt.ylabel("Actual True Category", fontsize=11)
    plt.xlabel("Predicted Category", fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info("Saved confusion matrix plot to %s", save_path)


def plot_comparison_chart(rows: List[Dict[str, Any]], save_path: Path) -> None:
    """Grouped bar chart of Accuracy / Macro-F1 / Weighted-F1 per model."""
    import seaborn as sns

    plt = _plt()
    df = pd.DataFrame(rows)
    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(data=df, x="Metric", y="Score", hue="Model")
    plt.title("Model Comparison on the Held-Out Test Set", fontsize=14, pad=12)
    plt.ylim(0, 1.05)
    plt.ylabel("Score", fontsize=11)
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2.0, height), ha="center", va="bottom",
                        fontsize=8, xytext=(0, 2), textcoords="offset points")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info("Saved comparison bar chart to %s", save_path)


def plot_precision_coverage(results: Dict[str, Dict[str, Any]], save_path: Path) -> None:
    """Precision of auto-accepted tickets vs. share of tickets auto-accepted, per confidence threshold."""
    plt = _plt()
    plt.figure(figsize=(7, 5))
    for key, res in results.items():
        pts = res["threshold_analysis"]
        plt.plot([p["coverage"] for p in pts], [p["precision"] for p in pts], marker="o", label=MODEL_TITLES.get(key, key))
        for p in pts:
            if p["threshold"] == 0.8:
                plt.annotate("0.8", (p["coverage"], p["precision"]), fontsize=7, xytext=(3, -10), textcoords="offset points")
    plt.xlabel("Coverage (share of tickets auto-accepted)")
    plt.ylabel("Precision of auto-accepted tickets")
    plt.title("Confidence Gate Trade-off (thresholds 0.5-0.9)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Model loaders: each returns predict_proba(texts) -> (n, num_classes) in the test label order
# ---------------------------------------------------------------------------
def _reorder(probs: np.ndarray, model_id2label: Dict[int, str], label2id: Dict[str, int]) -> np.ndarray:
    out = np.zeros((probs.shape[0], len(label2id)), dtype=np.float64)
    for model_idx, label in model_id2label.items():
        out[:, label2id[label]] = probs[:, model_idx]
    return out


def _read_id2label(model_dir: Path) -> Dict[int, str]:
    with open(model_dir / "labels.json", "r", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f)["id2label"].items()}


def load_sklearn_predictor(model_dir: Path, label2id: Dict[str, int]) -> Optional[PredictProba]:
    if not (model_dir / "logistic_regression.joblib").exists():
        return None
    vec = joblib.load(model_dir / "tfidf_vectorizer.joblib")
    clf = joblib.load(model_dir / "logistic_regression.joblib")
    id2label = _read_id2label(model_dir)
    class_map = {i: id2label[int(c)] for i, c in enumerate(clf.classes_)}
    return lambda texts: _reorder(clf.predict_proba(vec.transform(texts)), class_map, label2id)


def load_keras_predictor(model_dir: Path, label2id: Dict[str, int]) -> Optional[PredictProba]:
    if not (model_dir / "model.keras").exists():
        return None
    try:
        from ticketlens.train_keras import keras_predict_proba, load_keras_model

        model = load_keras_model(model_dir)
    except ImportError as e:
        logger.warning("Skipping Keras model (TensorFlow not installed: %s)", e)
        return None
    id2label = _read_id2label(model_dir)
    return lambda texts: _reorder(keras_predict_proba(model, texts), id2label, label2id)


def load_transformer_predictor(model_dir: Path, label2id: Dict[str, int], cfg: Dict[str, Any]) -> Optional[PredictProba]:
    if not model_dir.exists() or not any(model_dir.iterdir()):
        return None
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        logger.warning("Skipping transformer (PyTorch/Transformers not installed: %s)", e)
        return None

    device = get_device()
    id2label = _read_id2label(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if (model_dir / "adapter_config.json").exists():
        base_name = cfg["transformer"].get("pretrained_model", "distilbert/distilbert-base-uncased")
        base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=len(id2label))
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device).eval()
    max_length = int(cfg["transformer"].get("max_length", 128))

    def predict(texts: List[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                enc = tokenizer(texts[i: i + 64], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
                out.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
        return _reorder(np.concatenate(out), id2label, label2id)

    return predict


# ---------------------------------------------------------------------------
# Metrics & error analysis
# ---------------------------------------------------------------------------
def benchmark_latency(predict_fn: Callable[[str], Any], sample_texts: List[str], warmup: int = 5) -> Dict[str, float]:
    """Single-ticket (batch size 1) latency after warm-up: mean, P50, P95, P99 and throughput."""
    for text in sample_texts[: min(warmup, len(sample_texts))]:
        predict_fn(text)
    latencies = []
    for text in sample_texts:
        t0 = time.perf_counter()
        predict_fn(text)
        latencies.append((time.perf_counter() - t0) * 1000)
    arr = np.array(latencies)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "throughput_samples_per_sec": round(float(1000.0 / arr.mean()), 2) if arr.mean() > 0 else 0.0,
    }


def threshold_analysis(y_true: np.ndarray, probs: np.ndarray, thresholds: List[float] = THRESHOLDS) -> List[Dict[str, float]]:
    """For each confidence gate: share auto-accepted (coverage) and precision of those auto-accepted predictions."""
    preds, conf = probs.argmax(axis=1), probs.max(axis=1)
    rows = []
    for t in thresholds:
        mask = conf >= t
        rows.append({
            "threshold": t,
            "coverage": round(float(mask.mean()), 4),
            "precision": round(float((preds[mask] == y_true[mask]).mean()), 4) if mask.any() else None,
            "auto_accepted": int(mask.sum()),
        })
    return rows


def length_bucket_analysis(texts: List[str], y_true: np.ndarray, preds: np.ndarray) -> List[Dict[str, Any]]:
    """Accuracy and macro-F1 by ticket length: checks the model is not biased towards long or short tickets."""
    words = np.array([len(t.split()) for t in texts])
    rows = []
    for lo, hi, name in LENGTH_BUCKETS:
        mask = (words >= lo) & (words < hi)
        if not mask.any():
            continue
        rows.append({
            "bucket": name,
            "tickets": int(mask.sum()),
            "accuracy": round(float(accuracy_score(y_true[mask], preds[mask])), 4),
            "macro_f1": round(float(f1_score(y_true[mask], preds[mask], average="macro", zero_division=0)), 4),
        })
    return rows


def error_analysis(
    texts: List[str], y_true: np.ndarray, probs: np.ndarray, class_names: List[str], n_examples: int = 6
) -> Dict[str, Any]:
    preds, conf = probs.argmax(axis=1), probs.max(axis=1)
    wrong = np.where(preds != y_true)[0]
    pairs = Counter((class_names[y_true[i]], class_names[preds[i]]) for i in wrong)
    confident_errors = sorted(wrong, key=lambda i: -conf[i])[:n_examples]
    return {
        "errors": int(len(wrong)),
        "error_rate": round(float(len(wrong) / max(1, len(y_true))), 4),
        "top_confusions": [
            {"true": t, "predicted": p, "count": c, "share_of_errors": round(c / max(1, len(wrong)), 4)}
            for (t, p), c in pairs.most_common(6)
        ],
        "most_confident_errors": [
            {
                "true": class_names[y_true[i]],
                "predicted": class_names[preds[i]],
                "confidence": round(float(conf[i]), 4),
                "text": texts[i][:220].replace("\n", " ") + ("..." if len(texts[i]) > 220 else ""),
            }
            for i in confident_errors
        ],
    }


def handwritten_accuracy(predict: PredictProba, label2id: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """Accuracy on hand-written, real-world-style tickets (dev + holdout fixtures), model output only."""
    fixtures = get_project_root() / "data" / "fixtures"
    tickets = []
    for name in ("handwritten_eval.json", "handwritten_holdout.json"):
        path = fixtures / name
        if path.exists():
            tickets += json.loads(path.read_text(encoding="utf-8"))
    if not tickets:
        return None
    probs = predict([clean_and_redact_text(t["text"]) for t in tickets])
    y = np.array([label2id[t["label"]] for t in tickets])
    preds = probs.argmax(axis=1)
    return {
        "tickets": len(tickets),
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "macro_f1": round(float(f1_score(y, preds, average="macro", zero_division=0)), 4),
    }


def evaluate_predictions(
    key: str,
    texts: List[str],
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    figures_dir: Path,
) -> Dict[str, Any]:
    preds = probs.argmax(axis=1)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, preds, labels=labels)
    plot_confusion_matrix(cm, class_names, f"{MODEL_TITLES.get(key, key)}\n({len(y_true):,} test tickets)",
                          figures_dir / f"confusion_matrix_{key}.png")
    return {
        "model_name": MODEL_TITLES.get(key, key),
        "test_samples": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "macro_f1": round(float(f1_score(y_true, preds, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, preds, average="weighted", zero_division=0)), 4),
        "classification_report": classification_report(
            y_true, preds, labels=labels, target_names=class_names, output_dict=True, zero_division=0
        ),
        "confusion_matrix": cm.tolist(),
        "threshold_analysis": threshold_analysis(y_true, probs),
        "length_buckets": length_bucket_analysis(texts, y_true, preds),
        "error_analysis": error_analysis(texts, y_true, probs, class_names),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _load_test_split(cfg: Dict[str, Any], use_fixture: bool) -> Tuple[pd.DataFrame, Dict[str, int]]:
    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    test_path, label_path = processed_dir / "test.parquet", processed_dir / "label_mapping.json"
    if test_path.exists() and label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            return pd.read_parquet(test_path), json.load(f)["label2id"]
    logger.info("Processed test split not found. Generating splits (the test split is identical in every mode)...")
    _, _, test_df, label2id = get_dataset_splits(cfg, mode="all", use_fixture=use_fixture)
    return test_df, label2id


def evaluate_models(cfg: Optional[Dict[str, Any]] = None, use_fixture: bool = False) -> Dict[str, Any]:
    """Evaluate every available model on the identical held-out test split and write reports/figures."""
    if cfg is None:
        cfg = load_config()

    test_df, label2id = _load_test_split(cfg, use_fixture)
    id2label = {int(v): k for k, v in label2id.items()}
    class_names = [id2label[i] for i in range(len(label2id))]
    texts = test_df["text"].tolist()
    y_true = test_df["label"].to_numpy()
    logger.info("Evaluating on held-out test split: %d tickets across %d classes", len(texts), len(class_names))

    reports_dir = Path(cfg["paths"]["reports_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(cfg["paths"]["model_dir"])

    loaders: List[Tuple[str, Callable[[], Optional[PredictProba]]]] = [
        ("tfidf_benchmark", lambda: load_sklearn_predictor(model_dir / "tfidf_benchmark", label2id)),
        ("keras", lambda: load_keras_predictor(model_dir / "keras", label2id)),
        ("transformer", lambda: load_transformer_predictor(model_dir / "transformer", label2id, cfg)),
        ("baseline", lambda: load_sklearn_predictor(model_dir / "baseline", label2id)),
    ]

    train_info = {}
    for key in ("tfidf_benchmark", "keras", "transformer", "baseline"):
        path = reports_dir / f"metrics_{key}.json"
        if path.exists():
            info = json.loads(path.read_text(encoding="utf-8"))
            train_info[key] = {"train_samples": info.get("train_samples"), "mode": info.get("mode")}

    results: Dict[str, Dict[str, Any]] = {}
    cue_weight = float(cfg.get("inference", {}).get("itil_cue_weight", 1.5))
    for key, loader in loaders:
        predict = loader()
        if predict is None:
            logger.info("Model '%s' not available - skipped.", key)
            continue
        logger.info("Evaluating %s ...", MODEL_TITLES[key])
        probs = predict(texts)
        res = evaluate_predictions(key, texts, y_true, probs, class_names, figures_dir)
        res["latency"] = benchmark_latency(lambda t: predict([t]), texts[:50])
        res["handwritten"] = handwritten_accuracy(predict, label2id)
        res["train_samples"] = train_info.get(key, {}).get("train_samples")
        results[key] = res

        if key == "baseline":
            # Production pipeline: same model + ITIL cue correction.
            cued = np.array([
                [apply_itil_cues(dict(zip(class_names, row)), text, weight=cue_weight)[0][c] for c in class_names]
                for row, text in zip(probs, texts)
            ])
            res_itil = evaluate_predictions("baseline_itil", texts, y_true, cued, class_names, figures_dir)
            res_itil["latency"] = res["latency"]
            res_itil["train_samples"] = res["train_samples"]

            def predict_itil(batch: List[str], _predict: PredictProba = predict) -> np.ndarray:
                raw = _predict(batch)
                return np.array([
                    [apply_itil_cues(dict(zip(class_names, row)), t, weight=cue_weight)[0][c] for c in class_names]
                    for row, t in zip(raw, batch)
                ])

            res_itil["handwritten"] = handwritten_accuracy(predict_itil, label2id)
            results["baseline_itil"] = res_itil

        with open(reports_dir / f"metrics_test_{key}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

    if not results:
        logger.warning("No trained models found - nothing to evaluate.")
        return {}

    summary = {
        "dataset_name": cfg["dataset"]["name"],
        "test_samples_count": len(texts),
        "classes": class_names,
        "models": results,
    }
    with open(reports_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows = []
    for key, res in results.items():
        for metric, field in (("Accuracy", "accuracy"), ("Macro-F1", "macro_f1"), ("Weighted-F1", "weighted_f1")):
            rows.append({"Model": MODEL_TITLES.get(key, key), "Metric": metric, "Score": res[field]})
    plot_comparison_chart(rows, figures_dir / "model_comparison_f1.png")
    plot_precision_coverage(results, figures_dir / "precision_coverage.png")

    (reports_dir / "evaluation_summary.md").write_text(generate_summary_markdown(summary), encoding="utf-8")
    (reports_dir / "error_analysis.md").write_text(generate_error_analysis_markdown(summary), encoding="utf-8")
    logger.info("Evaluation complete! Reports and figures saved under %s", reports_dir)
    return summary


# ---------------------------------------------------------------------------
# Markdown reports
# ---------------------------------------------------------------------------
def generate_summary_markdown(summary: Dict[str, Any]) -> str:
    models = summary["models"]
    md = [
        "# TicketLens: Model Benchmark Summary\n",
        f"- **Dataset:** {summary['dataset_name']}",
        f"- **Held-out test set:** {summary['test_samples_count']:,} tickets (identical for every model)",
        f"- **Classes:** {', '.join(summary['classes'])}\n",
        "## Macro-F1 Benchmark\n",
        "| Model | Train tickets | Accuracy | Macro-F1 | Weighted-F1 | Hand-written acc. | P50 latency |",
        "| :--- | ---: | :---: | :---: | :---: | :---: | ---: |",
    ]
    for key, res in models.items():
        hw = res.get("handwritten") or {}
        train = res.get("train_samples")
        md.append(
            f"| **{res['model_name']}** | {train:,} | {res['accuracy']:.4f} | **{res['macro_f1']:.4f}** | "
            f"{res['weighted_f1']:.4f} | {hw.get('accuracy', float('nan')):.0%} | {res['latency']['p50_ms']} ms |"
            if train else
            f"| **{res['model_name']}** | - | {res['accuracy']:.4f} | **{res['macro_f1']:.4f}** | "
            f"{res['weighted_f1']:.4f} | {hw.get('accuracy', float('nan')):.0%} | {res['latency']['p50_ms']} ms |"
        )

    md.append("\n## Per-Class Precision / Recall\n")
    classes = summary["classes"]
    md.append("| Model | " + " | ".join(f"{c} P / R" for c in classes) + " |")
    md.append("| :--- | " + " | ".join(":---:" for _ in classes) + " |")
    for res in models.values():
        rep = res["classification_report"]
        md.append(f"| {res['model_name']} | " + " | ".join(
            f"{rep[c]['precision']:.2f} / {rep[c]['recall']:.2f}" for c in classes) + " |")

    md.extend([
        "\n## Notes\n",
        "- TF-IDF, Keras and DistilBERT are trained on the **same** training subset and scored on the **same** test tickets.",
        "- The served TF-IDF model uses all training data plus short-text augmentation; the production row adds the ITIL cue correction.",
        "- Hand-written accuracy uses the 72 hand-labelled tickets in `data/fixtures/handwritten_*.json` (model output only for non-production rows).",
        "- See `error_analysis.md` for confusion pairs, confidence-gate precision/coverage, length-bucket parity and error examples.\n",
        "## Figures\n",
        "- `reports/figures/model_comparison_f1.png`",
        "- `reports/figures/precision_coverage.png`",
    ] + [f"- `reports/figures/confusion_matrix_{key}.png`" for key in models])
    return "\n".join(md) + "\n"


def generate_error_analysis_markdown(summary: Dict[str, Any]) -> str:
    md = ["# TicketLens: Precision / Recall Error Analysis\n",
          f"All numbers are on the identical held-out test set ({summary['test_samples_count']:,} tickets).\n"]
    for res in summary["models"].values():
        ea = res["error_analysis"]
        md.extend([
            f"## {res['model_name']}\n",
            f"Errors: **{ea['errors']:,}** ({ea['error_rate']:.1%} of test tickets). Macro-F1 **{res['macro_f1']:.4f}**.\n",
            "**Most frequent confusions (true -> predicted):**\n",
            "| True | Predicted | Count | Share of errors |",
            "| :--- | :--- | ---: | ---: |",
        ])
        md += [f"| {c['true']} | {c['predicted']} | {c['count']} | {c['share_of_errors']:.1%} |" for c in ea["top_confusions"]]
        md.extend(["\n**Confidence gate (auto-accept) precision vs. coverage:**\n",
                   "| Threshold | Auto-accepted | Precision of auto-accepted |", "| :---: | :---: | :---: |"])
        md += [
            f"| {t['threshold']:.1f} | {t['coverage']:.1%} | {t['precision']:.1%} |" if t["precision"] is not None
            else f"| {t['threshold']:.1f} | 0.0% | - |"
            for t in res["threshold_analysis"]
        ]
        md.extend(["\n**Performance parity by ticket length (bias check):**\n",
                   "| Length | Tickets | Accuracy | Macro-F1 |", "| :--- | ---: | :---: | :---: |"])
        md += [f"| {b['bucket']} | {b['tickets']:,} | {b['accuracy']:.4f} | {b['macro_f1']:.4f} |" for b in res["length_buckets"]]
        md.extend(["\n**Most confident errors:**\n", "| True | Predicted | Confidence | Ticket (truncated) |",
                   "| :--- | :--- | :---: | :--- |"])
        md += [f"| {e['true']} | {e['predicted']} | {e['confidence']:.2f} | {e['text'].replace('|', '/')} |"
               for e in ea["most_confident_errors"]]
        md.append("")
    md.extend([
        "## Takeaways\n",
        "- Most errors across all models are **Incident <-> Problem** confusions: both describe something broken; ITIL separates "
        "them by recurrence / root-cause intent, which the synthetic labels do not apply consistently.",
        "- Request and Change are separated with high precision and recall by every model.",
        "- The confidence gate trades coverage for precision; at 0.8 the auto-accepted tickets are far more accurate than the overall "
        "accuracy, which is why lower-confidence tickets are routed to human review.\n",
    ])
    return "\n".join(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark all trained TicketLens models on the held-out test set")
    parser.add_argument("--fixture", action="store_true", help="Force synthetic fixture usage")
    args = parser.parse_args()

    evaluate_models(use_fixture=args.fixture)
