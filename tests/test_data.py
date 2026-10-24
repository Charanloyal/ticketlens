"""Tests for data preprocessing, validation, PII redaction, and splitting."""

import pandas as pd
import pytest
from ticketlens.data import (
    clean_and_redact_text,
    preprocess_data,
    validate_schema,
)


def test_pii_redaction():
    text = "User bob.smith@example.com called from 555-123-4567 reporting error on server 192.168.1.50."
    sanitized = clean_and_redact_text(text, redact_pii=True)

    assert "[EMAIL]" in sanitized
    assert "bob.smith@example.com" not in sanitized
    assert "[IP]" in sanitized
    assert "192.168.1.50" not in sanitized
    assert "[PHONE]" in sanitized


def test_text_normalization():
    messy_text = "   VPN error    \n\n\n\n\n  Connection dropped!   "
    cleaned = clean_and_redact_text(messy_text)
    assert cleaned == "VPN error\n\nConnection dropped!"


def test_validate_schema_passes():
    df = pd.DataFrame({
        "subject": ["A"],
        "body": ["B"],
        "type": ["Incident"],
    })
    validate_schema(df, ["subject", "body", "type"])


def test_validate_schema_raises_missing_column():
    df = pd.DataFrame({"subject": ["A"]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(df, ["subject", "type"])


def test_preprocess_data_splits_and_leakage(sample_ticket_fixture, base_config):
    df = pd.DataFrame(sample_ticket_fixture)
    # Duplicate some rows to verify deduplication
    df_with_duplicates = pd.concat([df, df.iloc[:2]], ignore_index=True)

    train_df, val_df, test_df, label2id = preprocess_data(df_with_duplicates, base_config, mode="quick")

    # Verify no overlap between train and test texts
    train_texts = set(train_df["text"])
    test_texts = set(test_df["text"])
    overlap = train_texts.intersection(test_texts)
    assert len(overlap) == 0, f"Data leakage detected! Overlapping texts: {overlap}"

    # Verify label mappings are valid
    assert len(label2id) >= 2
    assert "Incident" in label2id or "Request" in label2id
