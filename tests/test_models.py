"""Tests for model training, tokenization, forward pass, and evaluation metrics."""

import numpy as np
import pytest
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer
from ticketlens.train_transformer import TicketDataset


def test_baseline_fit_and_predict():
    corpus = [
        "VPN connection timed out with error",
        "WiFi network is disconnected on 5th floor",
        "Requesting monitor and docking station for new desk",
        "Please provide database access for analytics team",
    ]
    labels = [0, 0, 1, 1]

    vec = TfidfVectorizer(ngram_range=(1, 2))
    X = vec.fit_transform(corpus)
    clf = LogisticRegression()
    clf.fit(X, labels)

    test_sample = vec.transform(["VPN connection is not working"])
    pred = clf.predict(test_sample)[0]
    assert pred in [0, 1]

    probs = clf.predict_proba(test_sample)[0]
    assert len(probs) == 2
    assert np.isclose(np.sum(probs), 1.0)


def test_metric_calculations():
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 0]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    assert acc == 0.75
    assert 0.0 <= macro_f1 <= 1.0


def test_ticket_dataset_tokenization():
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    texts = ["VPN issue", "Need access"]
    labels = [0, 1]

    ds = TicketDataset(texts, labels, tokenizer, max_length=16)
    assert len(ds) == 2

    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["input_ids"].shape[0] == 16
    assert item["labels"].item() == 0
