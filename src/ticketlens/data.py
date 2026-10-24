"""Data loading, validation, preprocessing, and splitting pipeline for TicketLens."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from ticketlens.config import get_project_root, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ticketlens.data")

# Regex patterns for basic PII redaction
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
IP_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def clean_and_redact_text(text: str, redact_pii: bool = True) -> str:
    """Clean text, normalize whitespace, and optionally redact common PII."""
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()
    if redact_pii:
        cleaned = EMAIL_REGEX.sub("[EMAIL]", cleaned)
        cleaned = PHONE_REGEX.sub("[PHONE]", cleaned)
        cleaned = IP_REGEX.sub("[IP]", cleaned)

    # Normalize whitespace per line and remove redundant blank lines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
    non_empty_lines = []
    prev_blank = False
    for line in lines:
        if line:
            non_empty_lines.append(line)
            prev_blank = False
        elif not prev_blank:
            non_empty_lines.append("")
            prev_blank = True

    return "\n\n".join([line for line in non_empty_lines if line])


def validate_schema(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Ensure required columns exist in the DataFrame."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. Present: {list(df.columns)}")


def load_raw_dataset(cfg: Dict[str, Any], use_fixture: bool = False) -> pd.DataFrame:
    """
    Load dataset from Hugging Face or local parquet cache.
    Falls back to fixture file if internet is unavailable or use_fixture is True.
    """
    root = get_project_root()
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_cache_file = raw_dir / "customer_support_tickets_en.parquet"
    fixtures_file = Path(cfg["paths"]["fixtures_file"])

    if use_fixture:
        logger.info("Explicitly using synthetic test fixture: %s", fixtures_file)
        if not fixtures_file.exists():
            raise FileNotFoundError(f"Fixture not found at {fixtures_file}")
        with open(fixtures_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    # 1. Check local cache
    if local_cache_file.exists():
        logger.info("Loading cached raw dataset from %s", local_cache_file)
        return pd.read_parquet(local_cache_file)

    # 2. Try loading from Hugging Face
    dataset_name = cfg["dataset"]["name"]
    logger.info("Downloading dataset '%s' from Hugging Face Hub...", dataset_name)
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train")
        df = ds.to_pandas()

        # Filter by language if specified
        target_lang = cfg["dataset"].get("language", "en")
        if "language" in df.columns and target_lang:
            df = df[df["language"] == target_lang].copy()

        # Cache locally
        df.to_parquet(local_cache_file, index=False)
        logger.info("Cached %d rows to %s", len(df), local_cache_file)
        return df
    except Exception as e:
        logger.warning("Failed to load dataset from Hugging Face (%s). Falling back to fixture.", e)
        if fixtures_file.exists():
            with open(fixtures_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return pd.DataFrame(data)
        raise RuntimeError("Could not load Hugging Face dataset and no fixture file exists.") from e


def preprocess_data(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    mode: str = "quick",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Validate, clean, deduplicate, label-encode, and split dataset into train/val/test.
    Returns: (train_df, val_df, test_df, label2id)
    """
    target_col = cfg["dataset"]["target_column"]
    text_cols = cfg["dataset"]["text_columns"]
    validate_schema(df, [target_col] + text_cols)

    # Filter nulls in target or text
    df = df.dropna(subset=[target_col] + text_cols).copy()

    # Drop unsupported / rare classes if necessary (we focus on standard IT ticket categories)
    # Target column 'type' standard categories: Incident, Request, Problem, Change
    valid_classes = ["Incident", "Request", "Problem", "Change"]
    df = df[df[target_col].isin(valid_classes)].copy()

    # Construct combined text
    redact = cfg["dataset"].get("redact_pii", True)
    logger.info("Preprocessing and cleaning text (PII redaction=%s)...", redact)

    def combine_and_clean(row: pd.Series) -> str:
        parts = [str(row[col]) for col in text_cols if pd.notna(row[col])]
        combined = "\n".join(parts)
        return clean_and_redact_text(combined, redact_pii=redact)

    df["text"] = df.apply(combine_and_clean, axis=1)
    df = df[df["text"].str.len() > 10].copy()

    # Deduplication to avoid train/test leakage
    if cfg["dataset"].get("deduplicate", True):
        initial_count = len(df)
        df = df.drop_duplicates(subset=["text"]).copy()
        logger.info("Deduplication: removed %d duplicate tickets, %d remaining", initial_count - len(df), len(df))

    # Sample subset if requested or in quick/full mode
    seed = cfg.get("random_seed", 42)
    if mode == "quick":
        quick_samples = cfg["dataset"].get("quick_mode_samples", 300)
        if quick_samples < len(df):
            logger.info("Quick mode: sampling %d rows (stratified)", quick_samples)
            df, _ = train_test_split(df, train_size=quick_samples, stratify=df[target_col], random_state=seed)
    elif mode == "full":
        full_samples = cfg["dataset"].get("full_mode_samples", None)
        if full_samples and full_samples < len(df):
            logger.info("Full mode: sampling %d rows (stratified) for CPU efficiency", full_samples)
            df, _ = train_test_split(df, train_size=full_samples, stratify=df[target_col], random_state=seed)

    # Deterministic label mapping
    unique_labels = sorted(df[target_col].unique().tolist())
    label2id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    df["label"] = df[target_col].map(label2id)

    # Train / Val / Test split (70 / 15 / 15)
    train_ratio = cfg["dataset"].get("train_ratio", 0.70)
    val_ratio = cfg["dataset"].get("val_ratio", 0.15)
    test_ratio = cfg["dataset"].get("test_ratio", 0.15)

    # Stratification check: require each class count >= 3 and dataset >= 20
    min_class_count = df["label"].value_counts().min()
    stratify_col = df["label"] if (min_class_count >= 3 and len(df) >= 20) else None

    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=stratify_col,
        random_state=seed,
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    temp_stratify = temp_df["label"] if (stratify_col is not None and temp_df["label"].value_counts().min() >= 2) else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_stratify,
        random_state=seed,
    )

    logger.info(
        "Split sizes: Train=%d, Val=%d, Test=%d (Total=%d across %d classes)",
        len(train_df),
        len(val_df),
        len(test_df),
        len(df),
        len(label2id),
    )

    # Save processed splits
    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)

    with open(processed_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": {v: k for k, v in label2id.items()}}, f, indent=2)

    return train_df, val_df, test_df, label2id


def get_dataset_splits(cfg: Optional[Dict[str, Any]] = None, mode: str = "quick", use_fixture: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Convenience helper to load raw and return preprocessed splits."""
    if cfg is None:
        cfg = load_config()
    raw_df = load_raw_dataset(cfg, use_fixture=use_fixture)
    return preprocess_data(raw_df, cfg, mode=mode)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare dataset for TicketLens")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick", help="Execution mode")
    parser.add_argument("--fixture", action="store_true", help="Force synthetic fixture usage")
    args = parser.parse_args()

    config = load_config()
    get_dataset_splits(config, mode=args.mode, use_fixture=args.fixture)
