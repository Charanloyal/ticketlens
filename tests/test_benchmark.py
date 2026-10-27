"""Tests for the benchmark pipeline: fixed test split, Keras baseline, evaluation and error analysis."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
from ticketlens.data import preprocess_data
from ticketlens.evaluate import error_analysis, length_bucket_analysis, threshold_analysis


def _tmp_cfg(base_config, tmp_path):
    cfg = copy.deepcopy(base_config)
    for key in ("raw_data_dir", "processed_data_dir", "model_dir", "reports_dir"):
        cfg["paths"][key] = str(tmp_path / key)
    cfg["paths"]["figures_dir"] = str(tmp_path / "reports_dir" / "figures")
    return cfg


def _synthetic_df(n_per_class: int = 60) -> pd.DataFrame:
    rows = []
    words = {
        "Incident": "server outage down error crash",
        "Request": "please provide access license invoice",
        "Problem": "recurring root cause investigate repeated",
        "Change": "upgrade migrate schedule deploy configure",
    }
    for label, vocab in words.items():
        for i in range(n_per_class):
            rows.append({"subject": f"{label} ticket {i}", "body": f"{vocab} number {i} for team {i % 7}", "type": label})
    return pd.DataFrame(rows)


def test_test_split_is_identical_across_modes(base_config, tmp_path):
    cfg = _tmp_cfg(base_config, tmp_path)
    cfg["dataset"]["quick_mode_samples"] = 40
    cfg["dataset"]["full_mode_samples"] = 120
    df = _synthetic_df()
    test_sets = {}
    for mode in ("quick", "full", "all"):
        train_df, val_df, test_df, _ = preprocess_data(df.copy(), cfg, mode=mode)
        test_sets[mode] = set(test_df["text"])
        assert not set(train_df["text"]) & test_sets[mode]
        assert not set(val_df["text"]) & test_sets[mode]
    assert test_sets["quick"] == test_sets["full"] == test_sets["all"]
    assert len(preprocess_data(df.copy(), cfg, mode="quick")[0]) == 28  # 70% of the 40-ticket budget


def test_threshold_analysis_precision_and_coverage():
    y = np.array([0, 1, 1, 0])
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.55, 0.45]])
    rows = {r["threshold"]: r for r in threshold_analysis(y, probs, thresholds=[0.5, 0.8])}
    assert rows[0.5]["coverage"] == 1.0 and rows[0.5]["precision"] == 0.75
    assert rows[0.8]["coverage"] == 0.5 and rows[0.8]["precision"] == 1.0


def test_error_analysis_and_length_buckets():
    texts = ["short text", "another short one", "word " * 120, "word " * 60]
    y = np.array([0, 1, 1, 0])
    probs = np.array([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.4, 0.6]])
    ea = error_analysis(texts, y, probs, ["A", "B"])
    assert ea["errors"] == 2
    assert ea["most_confident_errors"][0]["confidence"] == 0.7
    assert {c["true"] + c["predicted"] for c in ea["top_confusions"]} == {"BA", "AB"}
    buckets = {b["bucket"]: b for b in length_bucket_analysis(texts, y, probs.argmax(axis=1))}
    assert buckets["< 50 words"]["tickets"] == 2 and buckets["100-199 words"]["accuracy"] == 1.0


def test_evaluate_models_end_to_end_with_tfidf(base_config, tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    from ticketlens.evaluate import evaluate_models
    from ticketlens.train_baseline import train_baseline_model

    cfg = _tmp_cfg(base_config, tmp_path)
    cfg["dataset"]["full_mode_samples"] = 160
    raw_dir = tmp_path / "raw_data_dir"
    raw_dir.mkdir()
    df = _synthetic_df()
    df["language"] = "en"
    df.to_parquet(raw_dir / "customer_support_tickets_en.parquet", index=False)

    train_baseline_model(cfg, mode="full", model_name="tfidf_benchmark", augment=False)
    summary = evaluate_models(cfg)
    res = summary["models"]["tfidf_benchmark"]
    assert res["test_samples"] == summary["test_samples_count"] == 36
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert set(res["classification_report"]) >= {"Incident", "Request", "Problem", "Change"}
    # Headline hand-written score is the untouched 24-ticket holdout; the tuning set is reported separately.
    assert res["handwritten"]["holdout"]["tickets"] == 24
    assert res["handwritten"]["accuracy"] == res["handwritten"]["holdout"]["accuracy"]
    assert res["handwritten"]["development"]["tickets"] == 48
    reports = tmp_path / "reports_dir"
    for name in ("model_comparison.json", "evaluation_summary.md", "error_analysis.md"):
        assert (reports / name).exists()
    assert (reports / "figures" / "confusion_matrix_tfidf_benchmark.png").exists()
    assert json.loads((reports / "model_comparison.json").read_text())["models"]


def test_keras_baseline_trains_saves_and_reloads(base_config, tmp_path):
    pytest.importorskip("tensorflow")
    from ticketlens.train_keras import keras_predict_proba, load_keras_model, train_keras_model

    cfg = _tmp_cfg(base_config, tmp_path)
    cfg["keras"] = {**cfg.get("keras", {}), "epochs": 2, "max_tokens": 500, "sequence_length": 16, "batch_size": 16}
    raw_dir = tmp_path / "raw_data_dir"
    raw_dir.mkdir()
    df = _synthetic_df()
    df["language"] = "en"
    df.to_parquet(raw_dir / "customer_support_tickets_en.parquet", index=False)

    metrics = train_keras_model(cfg, mode="all")
    assert metrics["framework"] == "TensorFlow / Keras"
    model = load_keras_model(tmp_path / "model_dir" / "keras")
    probs = keras_predict_proba(model, ["server outage down", "please provide a license"])
    assert probs.shape == (2, 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
