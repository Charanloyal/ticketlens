"""Explainability analysis (SHAP & n-gram attributions) and dataset bias/fairness auditing."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import shap
from ticketlens.config import load_config
from ticketlens.data import EMAIL_REGEX, IP_REGEX, PHONE_REGEX, get_dataset_splits, load_raw_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ticketlens.explain")


def extract_top_features_per_class(vectorizer, classifier, id2label: Dict[int, str], top_n: int = 10) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Extract top positive and negative n-grams for each category based on Logistic Regression coefficients."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for class_idx, class_name in id2label.items():
        coefs = classifier.coef_[class_idx]
        top_pos_indices = np.argsort(coefs)[-top_n:][::-1]
        top_neg_indices = np.argsort(coefs)[:top_n]

        results[class_name] = {
            "top_positive_features": [
                {"feature": str(feature_names[i]), "weight": round(float(coefs[i]), 4)}
                for i in top_pos_indices
            ],
            "top_negative_features": [
                {"feature": str(feature_names[i]), "weight": round(float(coefs[i]), 4)}
                for i in top_neg_indices
            ],
        }

    return results


def run_shap_analysis(vectorizer, classifier, sample_texts: List[str], id2label: Dict[int, str]) -> Dict[str, Any]:
    """Run SHAP LinearExplainer on TF-IDF features for sample tickets."""
    logger.info("Computing SHAP values for %d sample tickets...", len(sample_texts))
    X_samples = vectorizer.transform(sample_texts).toarray()
    explainer = shap.LinearExplainer(classifier, X_samples)
    shap_values = explainer.shap_values(X_samples)

    feature_names = np.array(vectorizer.get_feature_names_out())
    # Handle multi-class SHAP outputs (shape: samples x features x classes or list of classes)
    shap_summary: Dict[str, List[Dict[str, Any]]] = {}

    for class_idx, class_name in id2label.items():
        if isinstance(shap_values, list):
            class_shap = np.abs(shap_values[class_idx]).mean(axis=0)
        elif len(shap_values.shape) == 3:
            class_shap = np.abs(shap_values[:, :, class_idx]).mean(axis=0)
        else:
            class_shap = np.abs(shap_values).mean(axis=0)

        top_indices = np.argsort(class_shap)[-10:][::-1]
        shap_summary[class_name] = [
            {"feature": str(feature_names[i]), "mean_abs_shap": round(float(class_shap[i]), 5)}
            for i in top_indices if class_shap[i] > 0
        ]

    return shap_summary


def audit_dataset_bias(df: pd.DataFrame, target_col: str = "type", raw_texts: Optional[List[str]] = None) -> Dict[str, Any]:
    """Audit dataset label distribution, class imbalance, text lengths, and PII exposure."""
    total_count = len(df)
    class_counts = df[target_col].value_counts().to_dict()
    majority_count = max(class_counts.values())
    minority_count = min(class_counts.values())
    imbalance_ratio = round(majority_count / max(1, minority_count), 2)

    # Length statistics
    lengths_by_class = {}
    for cls in class_counts.keys():
        subset = df[df[target_col] == cls]
        lengths = subset["text"].apply(lambda t: len(str(t).split())).tolist()
        lengths_by_class[cls] = {
            "mean_word_count": round(float(np.mean(lengths)), 1),
            "median_word_count": int(np.median(lengths)),
            "min_words": int(np.min(lengths)),
            "max_words": int(np.max(lengths)),
        }

    # PII presence scan on the raw (pre-redaction) text; the cleaned "text" column is already redacted.
    pool = pd.Series(raw_texts if raw_texts is not None else df["text"].tolist()).dropna()
    sample_text = " ".join(pool.sample(min(500, len(pool)), random_state=42).tolist())
    email_hits = len(EMAIL_REGEX.findall(sample_text))
    phone_hits = len(PHONE_REGEX.findall(sample_text))
    ip_hits = len(IP_REGEX.findall(sample_text))

    return {
        "total_tickets_audited": total_count,
        "class_distribution": class_counts,
        "class_imbalance_ratio": imbalance_ratio,
        "text_length_by_class": lengths_by_class,
        "pii_risk_audit": {
            "raw_sample_size": int(min(500, len(pool))),
            "emails_detected_pre_sanitization": email_hits,
            "phone_numbers_detected_pre_sanitization": phone_hits,
            "ip_addresses_detected_pre_sanitization": ip_hits,
            "sanitization_status": "PII scrubbing enabled (regex redaction)",
        },
    }


def generate_explainability_report(
    top_features: Dict[str, Any],
    shap_summary: Dict[str, Any],
    bias_audit: Dict[str, Any],
    save_path: Path,
) -> None:
    """Write comprehensive Markdown report covering explainability, bias, and privacy."""
    md = [
        "# TicketLens: Model Explainability, Bias & Responsible AI Audit\n",
        "## 1. Executive Summary\n",
        "This report details feature attribution, SHAP explainability, dataset label representation, and privacy/PII risk checks conducted on the TicketLens IT support ticket classification system.\n",
        "## 2. Dataset Distribution & Bias Audit\n",
        f"- **Total Tickets Audited:** {bias_audit['total_tickets_audited']}",
        f"- **Class Imbalance Ratio (Majority / Minority):** {bias_audit['class_imbalance_ratio']}:1",
        "- **Class Distribution:**",
    ]

    for cls, count in bias_audit["class_distribution"].items():
        pct = (count / bias_audit['total_tickets_audited']) * 100
        md.append(f"  - **{cls}:** {count} samples ({pct:.1f}%)")

    md.extend([
        "\n### Ticket Length Profile by Category (Words)\n",
        "| Category | Mean Word Count | Median Word Count | Min Words | Max Words |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for cls, stats in bias_audit["text_length_by_class"].items():
        md.append(f"| {cls} | {stats['mean_word_count']} | {stats['median_word_count']} | {stats['min_words']} | {stats['max_words']} |")

    md.extend([
        "\n## 3. Feature Importance & SHAP Attribution\n",
        "Top driving n-gram features identified for each IT service category via linear attribution and SHAP values:\n",
    ])

    for cls, feat_data in top_features.items():
        md.append(f"### Category: `{cls}`")
        md.append("**Top Positive Driving Features (TF-IDF Weights):**")
        pos_list = ", ".join([f"`{x['feature']}` (+{x['weight']})" for x in feat_data["top_positive_features"][:6]])
        md.append(pos_list)
        if cls in shap_summary:
            md.append("\n**Top SHAP Attributions (|SHAP| impact):**")
            shap_list = ", ".join([f"`{x['feature']}` ({x['mean_abs_shap']})" for x in shap_summary[cls][:6]])
            md.append(shap_list)
        md.append("")

    md.extend([
        "## 4. Privacy, PII & Responsible AI Considerations\n",
        "### PII Audit Findings\n",
        f"- **Emails detected in raw sample:** {bias_audit['pii_risk_audit']['emails_detected_pre_sanitization']}",
        f"- **Phone numbers detected in raw sample:** {bias_audit['pii_risk_audit']['phone_numbers_detected_pre_sanitization']}",
        f"- **IP addresses detected in raw sample:** {bias_audit['pii_risk_audit']['ip_addresses_detected_pre_sanitization']}",
        f"- **Sanitization Pipeline:** `{bias_audit['pii_risk_audit']['sanitization_status']}`.\n",
        "### Limitations & Responsible Deployment\n",
        "1. **Audit Scope:** A balanced class distribution does not imply absence of demographic or organizational bias. Tickets may exhibit subtle stylistic variance across different regional departments.",
        "2. **Explainability Caveat:** Linear coefficients and SHAP values highlight lexical correlations (e.g. keywords like 'outage', 'access', 'invoice'). They do not capture complex cross-sentence dependencies or emotional sentiment.",
        "3. **Human-in-the-Loop Routing:** In high-stakes IT incidents (such as security breaches or enterprise database dropouts), automated routing decisions with confidence < 0.80 MUST trigger human triage rather than silent auto-assignment.\n",
    ])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info("Saved explainability and bias report to %s", save_path)


def run_explainability_pipeline(cfg: Optional[Dict[str, Any]] = None, use_fixture: bool = False) -> Dict[str, Any]:
    """Execute complete explainability and bias analysis."""
    if cfg is None:
        cfg = load_config()

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset audit on the full cleaned dataset (temporary processed dir: no side effects on data/processed)
    import copy
    import tempfile

    raw_df = load_raw_dataset(cfg, use_fixture=use_fixture)
    text_cols = [c for c in cfg["dataset"]["text_columns"] if c in raw_df.columns]
    raw_texts = raw_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
    tmp_cfg = copy.deepcopy(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_cfg["paths"]["processed_data_dir"] = tmp
        train_df, val_df, test_df, label2id = get_dataset_splits(tmp_cfg, mode="all", use_fixture=use_fixture)
    id2label = {v: k for k, v in label2id.items()}
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    bias_audit = audit_dataset_bias(full_df, target_col="label", raw_texts=raw_texts)
    bias_audit["class_distribution"] = {id2label[k]: v for k, v in bias_audit["class_distribution"].items()}
    bias_audit["text_length_by_class"] = {id2label[k]: v for k, v in bias_audit["text_length_by_class"].items()}

    with open(reports_dir / "bias_and_fairness_audit.json", "w", encoding="utf-8") as f:
        json.dump(bias_audit, f, indent=2)

    # 2. Baseline Explainability
    baseline_dir = Path(cfg["paths"]["model_dir"]) / "baseline"
    top_features = {}
    shap_summary = {}

    if (baseline_dir / "logistic_regression.joblib").exists():
        vec = joblib.load(baseline_dir / "tfidf_vectorizer.joblib")
        clf = joblib.load(baseline_dir / "logistic_regression.joblib")
        top_features = extract_top_features_per_class(vec, clf, id2label)
        with open(reports_dir / "top_features_per_class.json", "w", encoding="utf-8") as f:
            json.dump(top_features, f, indent=2)

        sample_texts = test_df["text"].iloc[: min(40, len(test_df))].tolist()
        shap_summary = run_shap_analysis(vec, clf, sample_texts, id2label)
        with open(reports_dir / "shap_summary.json", "w", encoding="utf-8") as f:
            json.dump(shap_summary, f, indent=2)

    # 3. Generate Report
    generate_explainability_report(top_features, shap_summary, bias_audit, reports_dir / "explainability_and_bias_report.md")

    return {
        "bias_audit": bias_audit,
        "top_features": top_features,
        "shap_summary": shap_summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run explainability and bias analysis")
    parser.add_argument("--fixture", action="store_true", help="Force synthetic fixture usage")
    args = parser.parse_args()

    run_explainability_pipeline(use_fixture=args.fixture)
