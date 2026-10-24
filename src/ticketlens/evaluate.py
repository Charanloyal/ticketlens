"""Comprehensive model evaluation and comparison on held-out test set."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ticketlens.config import get_device, load_config
from ticketlens.data import get_dataset_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ticketlens.evaluate")


def benchmark_latency(predict_fn, sample_texts: List[str], warmup: int = 5) -> Dict[str, float]:
    """
    Measure single-sample inference latency with P50, P95, P99 and throughput.
    Methodology: Warm-up passes followed by single-item (batch_size=1) timed executions using time.perf_counter.
    """
    # Warmup
    for text in sample_texts[: min(warmup, len(sample_texts))]:
        _ = predict_fn(text)

    latencies_ms = []
    for text in sample_texts:
        t0 = time.perf_counter()
        _ = predict_fn(text)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies_ms)
    return {
        "mean_ms": round(float(np.mean(latencies)), 2),
        "std_ms": round(float(np.std(latencies)), 2),
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "throughput_samples_per_sec": round(float(1000.0 / np.mean(latencies)), 2) if np.mean(latencies) > 0 else 0.0,
    }


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], title: str, save_path: Path) -> None:
    """Plot and save confusion matrix figure."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=14, pad=12)
    plt.ylabel("Actual True Category", fontsize=11)
    plt.xlabel("Predicted Category", fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved confusion matrix plot to %s", save_path)


def plot_comparison_chart(comparison_df: pd.DataFrame, save_path: Path) -> None:
    """Generate bar chart comparing baseline vs transformer."""
    plt.figure(figsize=(8, 5))
    metrics_to_plot = ["Accuracy", "Macro-F1", "Weighted-F1"]
    plot_df = comparison_df[comparison_df["Metric"].isin(metrics_to_plot)]

    ax = sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette=["#3b82f6", "#10b981"])
    plt.title("Model Performance Comparison on Test Set", fontsize=14, pad=12)
    plt.ylim(0, 1.05)
    plt.ylabel("Score", fontsize=11)

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved comparison bar chart to %s", save_path)


def evaluate_models(cfg: Optional[Dict[str, Any]] = None, use_fixture: bool = False) -> Dict[str, Any]:
    """Evaluate both baseline and transformer on identical held-out test split."""
    if cfg is None:
        cfg = load_config()

    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    test_path = processed_dir / "test.parquet"
    label_path = processed_dir / "label_mapping.json"

    if not test_path.exists() or not label_path.exists():
        logger.info("Processed test data not found. Generating splits...")
        _, _, test_df, label2id = get_dataset_splits(cfg, mode="quick", use_fixture=use_fixture)
    else:
        test_df = pd.read_parquet(test_path)
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
            label2id = label_data["label2id"]

    id2label = {int(v): k for k, v in label2id.items()}
    num_classes = len(label2id)
    class_names = [id2label[i] for i in range(num_classes)]
    y_test = test_df["label"].values
    test_texts = test_df["text"].tolist()

    logger.info("Evaluating on held-out test split: %d samples across %d classes", len(test_df), num_classes)
    reports_dir = Path(cfg["paths"]["reports_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "dataset_name": cfg["dataset"]["name"],
        "test_samples_count": len(test_df),
        "classes": class_names,
        "models": {},
    }

    # ==========================================
    # 1. EVALUATE BASELINE (TF-IDF + Logistic Regression)
    # ==========================================
    baseline_dir = Path(cfg["paths"]["model_dir"]) / "baseline"
    has_baseline = (baseline_dir / "logistic_regression.joblib").exists()

    if has_baseline:
        logger.info("Loading baseline model from %s...", baseline_dir)
        vec = joblib.load(baseline_dir / "tfidf_vectorizer.joblib")
        clf = joblib.load(baseline_dir / "logistic_regression.joblib")

        X_test = vec.transform(test_texts)
        b_preds = clf.predict(X_test)
        b_probs = clf.predict_proba(X_test)

        b_acc = float(accuracy_score(y_test, b_preds))
        b_macro_f1 = float(f1_score(y_test, b_preds, average="macro", zero_division=0))
        b_weighted_f1 = float(f1_score(y_test, b_preds, average="weighted", zero_division=0))
        b_cm = confusion_matrix(y_test, b_preds)

        b_rep = classification_report(y_test, b_preds, target_names=class_names, output_dict=True, zero_division=0)
        plot_confusion_matrix(b_cm, class_names, "Baseline: TF-IDF + Logistic Regression", figures_dir / "confusion_matrix_baseline.png")

        # Latency benchmark
        def baseline_predict_single(text: str):
            v = vec.transform([text])
            return clf.predict_proba(v)

        b_latency = benchmark_latency(baseline_predict_single, test_texts[: min(50, len(test_texts))])

        results["models"]["baseline"] = {
            "model_name": "TF-IDF + Logistic Regression",
            "accuracy": round(b_acc, 4),
            "macro_f1": round(b_macro_f1, 4),
            "weighted_f1": round(b_weighted_f1, 4),
            "confusion_matrix": b_cm.tolist(),
            "classification_report": b_rep,
            "latency": b_latency,
        }
        with open(reports_dir / "metrics_test_baseline.json", "w", encoding="utf-8") as f:
            json.dump(results["models"]["baseline"], f, indent=2)

    # ==========================================
    # 2. EVALUATE TRANSFORMER (DistilBERT + LoRA)
    # ==========================================
    transformer_dir = Path(cfg["paths"]["model_dir"]) / "transformer"
    has_transformer = transformer_dir.exists() and any(transformer_dir.iterdir())

    if has_transformer:
        device = get_device()
        logger.info("Loading transformer model from %s onto %s...", transformer_dir, device)
        tokenizer = AutoTokenizer.from_pretrained(transformer_dir)

        # Check if saved with PEFT adapter or standard model
        adapter_config = transformer_dir / "adapter_config.json"
        if adapter_config.exists():
            base_model_name = cfg["transformer"].get("pretrained_model", "distilbert/distilbert-base-uncased")
            base = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=num_classes)
            t_model = PeftModel.from_pretrained(base, transformer_dir)
        else:
            t_model = AutoModelForSequenceClassification.from_pretrained(transformer_dir)

        t_model.to(device)
        t_model.eval()

        t_preds = []
        t_probs = []

        with torch.no_grad():
            batch_size = 32
            for i in range(0, len(test_texts), batch_size):
                batch_text = test_texts[i: i + batch_size]
                encoded = tokenizer(batch_text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                logits = t_model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probs, axis=-1)
                t_preds.extend(preds)
                t_probs.extend(probs)

        t_acc = float(accuracy_score(y_test, t_preds))
        t_macro_f1 = float(f1_score(y_test, t_preds, average="macro", zero_division=0))
        t_weighted_f1 = float(f1_score(y_test, t_preds, average="weighted", zero_division=0))
        t_cm = confusion_matrix(y_test, t_preds)

        t_rep = classification_report(y_test, t_preds, target_names=class_names, output_dict=True, zero_division=0)
        plot_confusion_matrix(t_cm, class_names, "Transformer: DistilBERT (LoRA)", figures_dir / "confusion_matrix_transformer.png")

        # Latency benchmark
        def transformer_predict_single(text: str):
            enc = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.no_grad():
                return t_model(**enc).logits

        t_latency = benchmark_latency(transformer_predict_single, test_texts[: min(50, len(test_texts))])

        results["models"]["transformer"] = {
            "model_name": "DistilBERT (LoRA / PEFT)",
            "accuracy": round(t_acc, 4),
            "macro_f1": round(t_macro_f1, 4),
            "weighted_f1": round(t_weighted_f1, 4),
            "confusion_matrix": t_cm.tolist(),
            "classification_report": t_rep,
            "latency": t_latency,
        }
        with open(reports_dir / "metrics_test_transformer.json", "w", encoding="utf-8") as f:
            json.dump(results["models"]["transformer"], f, indent=2)

    # ==========================================
    # 3. COMPARISON & SUMMARY REPORT
    # ==========================================
    comparison_rows = []
    if has_baseline and "baseline" in results["models"]:
        b = results["models"]["baseline"]
        comparison_rows.extend([
            {"Model": "Baseline (TF-IDF + LR)", "Metric": "Accuracy", "Score": b["accuracy"]},
            {"Model": "Baseline (TF-IDF + LR)", "Metric": "Macro-F1", "Score": b["macro_f1"]},
            {"Model": "Baseline (TF-IDF + LR)", "Metric": "Weighted-F1", "Score": b["weighted_f1"]},
        ])
    if has_transformer and "transformer" in results["models"]:
        t = results["models"]["transformer"]
        comparison_rows.extend([
            {"Model": "Transformer (DistilBERT LoRA)", "Metric": "Accuracy", "Score": t["accuracy"]},
            {"Model": "Transformer (DistilBERT LoRA)", "Metric": "Macro-F1", "Score": t["macro_f1"]},
            {"Model": "Transformer (DistilBERT LoRA)", "Metric": "Weighted-F1", "Score": t["weighted_f1"]},
        ])

    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        plot_comparison_chart(comp_df, figures_dir / "model_comparison_f1.png")

    with open(reports_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Summary
    summary_md = generate_summary_markdown(results)
    with open(reports_dir / "evaluation_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    logger.info("Evaluation complete! Reports and figures saved under %s", reports_dir)
    return results


def generate_summary_markdown(results: Dict[str, Any]) -> str:
    """Generate professional Markdown summary table and discussion."""
    models = results.get("models", {})
    md = [
        "# TicketLens: Model Evaluation & Benchmark Summary\n",
        f"- **Dataset Source:** {results.get('dataset_name', 'Unknown')}",
        f"- **Held-Out Test Set Size:** {results.get('test_samples_count', 0)} samples",
        f"- **Evaluated Classes:** {', '.join(results.get('classes', []))}\n",
        "## Performance Comparison\n",
        "| Model Architecture | Accuracy | Macro-F1 | Weighted-F1 | P50 Latency (ms) | P95 Latency (ms) | Throughput (req/s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model_key, data in models.items():
        name = data.get("model_name", model_key)
        acc = data.get("accuracy", 0.0)
        macro = data.get("macro_f1", 0.0)
        weighted = data.get("weighted_f1", 0.0)
        lat = data.get("latency", {})
        p50 = lat.get("p50_ms", 0.0)
        p95 = lat.get("p95_ms", 0.0)
        thr = lat.get("throughput_samples_per_sec", 0.0)
        md.append(f"| **{name}** | {acc:.4f} | {macro:.4f} | {weighted:.4f} | {p50} ms | {p95} ms | {thr} req/s |")

    md.extend([
        "\n## Operational Trade-Offs & Key Takeaways\n",
        "1. **Inference Latency vs Semantic Generalization:**",
        "   - The **TF-IDF + Logistic Regression** baseline delivers sub-millisecond to low-millisecond CPU latency, making it extremely cost-efficient for high-volume ticket routing.",
        "   - The **DistilBERT (LoRA)** transformer captures deep contextual subtleties (e.g. nuanced phrases distinguishing a recurring *Problem* from an immediate *Incident*), but exhibits higher inference latency on CPU architectures.",
        "2. **Production Recommendation:**",
        "   - Use the **DistilBERT (LoRA)** model for tickets with complex descriptions or ambiguous phrasing, while leveraging the **TF-IDF Baseline** as a high-speed fallback or lightweight edge filter.",
        "\n## Generated Figures\n",
        "- `reports/figures/confusion_matrix_baseline.png`",
        "- `reports/figures/confusion_matrix_transformer.png`",
        "- `reports/figures/model_comparison_f1.png`\n",
    ])
    return "\n".join(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TicketLens models on held-out test set")
    parser.add_argument("--fixture", action="store_true", help="Force synthetic fixture usage")
    args = parser.parse_args()

    evaluate_models(use_fixture=args.fixture)
