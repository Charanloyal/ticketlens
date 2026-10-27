"""Data loading, validation, preprocessing, and splitting pipeline for TicketLens."""

import gzip
import json
import logging
import re
import time
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


def _required_columns(cfg: Dict[str, Any]) -> List[str]:
    """Columns needed for training; loading only these keeps memory low (e.g. on Render build machines)."""
    return [cfg["dataset"]["target_column"], *cfg["dataset"]["text_columns"], "language"]


def load_raw_dataset(cfg: Dict[str, Any], use_fixture: bool = False, require_dataset: bool = False) -> pd.DataFrame:
    """
    Load dataset from Hugging Face or local parquet cache.
    Falls back to fixture file if internet is unavailable or use_fixture is True.
    With require_dataset=True (production builds), a failed download raises instead of
    silently training on the tiny fixture.
    """
    root = get_project_root()
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_cache_file = raw_dir / "customer_support_tickets_en.parquet"
    fixtures_file = Path(cfg["paths"]["fixtures_file"])
    required = _required_columns(cfg)

    if use_fixture:
        if require_dataset:
            raise ValueError("use_fixture and require_dataset are mutually exclusive.")
        logger.info("Explicitly using synthetic test fixture: %s", fixtures_file)
        if not fixtures_file.exists():
            raise FileNotFoundError(f"Fixture not found at {fixtures_file}")
        with open(fixtures_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    # 1. Check local cache
    if local_cache_file.exists():
        logger.info("Loading cached raw dataset from %s", local_cache_file)
        import pyarrow.parquet as pq

        available = set(pq.read_schema(local_cache_file).names)
        return pd.read_parquet(local_cache_file, columns=[c for c in required if c in available])

    # 2. Try loading from Hugging Face (with retries for transient network / rate-limit errors)
    dataset_name = cfg["dataset"]["name"]
    attempts = 3 if require_dataset else 1
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        logger.info("Downloading dataset '%s' from Hugging Face Hub (attempt %d/%d)...", dataset_name, attempt, attempts)
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split="train")
            ds = ds.select_columns([c for c in required if c in ds.column_names])

            # Filter by language inside Arrow (memory-mapped) before converting to pandas
            target_lang = cfg["dataset"].get("language", "en")
            if "language" in ds.column_names and target_lang:
                ds = ds.filter(
                    lambda langs: [lang == target_lang for lang in langs],
                    batched=True,
                    input_columns=["language"],
                )
            df = ds.to_pandas()
            del ds

            # Cache locally
            df.to_parquet(local_cache_file, index=False)
            logger.info("Cached %d rows to %s", len(df), local_cache_file)
            return df
        except Exception as e:  # noqa: BLE001 - any failure triggers retry / fallback
            last_error = e
            logger.warning("Failed to load dataset from Hugging Face: %s", e)
            if attempt < attempts:
                time.sleep(5 * attempt)

    if require_dataset:
        raise RuntimeError(
            f"Could not download dataset '{dataset_name}' from Hugging Face and require_dataset is set. "
            "Refusing to fall back to the tiny synthetic fixture."
        ) from last_error

    logger.warning("Falling back to synthetic fixture %s.", fixtures_file)
    if fixtures_file.exists():
        with open(fixtures_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    raise RuntimeError("Could not load Hugging Face dataset and no fixture file exists.") from last_error


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

    # Filter nulls in target; a ticket only needs at least one non-empty text field
    # (many short Request tickets have no subject and were previously dropped).
    df = df.dropna(subset=[target_col]).copy()
    df = df[df[text_cols].notna().any(axis=1)].copy()

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
    # mode == "all": use every cleaned, deduplicated row (cheap for the TF-IDF baseline).

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


MIN_REAL_DATASET_ROWS = 1000
CORPUS_FILENAME = "baseline_corpus.json.gz"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HTML_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)


def short_text_variants(train_df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    """Create subject-only and first-sentence samples so the model also learns from short, hand-written-style tickets.

    Only derived from the training split, so validation/test data never leaks into training.
    """
    text_cols = cfg["dataset"]["text_columns"]
    redact = cfg["dataset"].get("redact_pii", True)
    texts: List[str] = []
    labels: List[int] = []
    subject_col, body_col = (text_cols + [None, None])[:2]

    for _, row in train_df.iterrows():
        candidates = []
        if subject_col and isinstance(row.get(subject_col), str):
            candidates.append(row[subject_col])
        if body_col and isinstance(row.get(body_col), str):
            body = _HTML_BREAK.sub(" ", row[body_col])
            candidates.append(_SENTENCE_SPLIT.split(body.strip(), maxsplit=1)[0])
        for candidate in candidates:
            cleaned = clean_and_redact_text(candidate, redact_pii=redact)
            if len(cleaned) > 8:
                texts.append(cleaned)
                labels.append(int(row["label"]))
    return texts, labels


def build_baseline_corpus(
    cfg: Dict[str, Any],
    mode: str = "quick",
    use_fixture: bool = False,
    require_dataset: bool = False,
) -> Dict[str, Any]:
    """Load, clean, split and (optionally) augment the data into plain text/label lists for baseline training."""
    train_df, val_df, test_df, label2id = get_dataset_splits(
        cfg, mode=mode, use_fixture=use_fixture, require_dataset=require_dataset
    )
    train_text = train_df["text"].tolist()
    train_label = [int(v) for v in train_df["label"]]
    n_augmented = 0
    if cfg["baseline"].get("augment_short_text", False):
        extra_text, extra_label = short_text_variants(train_df, cfg)
        n_augmented = len(extra_text)
        logger.info("Short-text augmentation: added %d subject/first-sentence samples.", n_augmented)
        train_text += extra_text
        train_label += extra_label
    return {
        "mode": mode,
        "label2id": label2id,
        "train_text": train_text,
        "train_label": train_label,
        "n_train_tickets": len(train_df),
        "n_augmented": n_augmented,
        "val_text": val_df["text"].tolist(),
        "val_label": [int(v) for v in val_df["label"]],
        "test_size": len(test_df),
    }


def corpus_path(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["paths"]["processed_data_dir"]) / CORPUS_FILENAME


def save_baseline_corpus(cfg: Dict[str, Any], corpus: Dict[str, Any]) -> Path:
    path = corpus_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(corpus, f)
    logger.info("Saved baseline training corpus (%d train / %d val texts) to %s", len(corpus["train_text"]), len(corpus["val_text"]), path)
    return path


def load_baseline_corpus(cfg: Dict[str, Any], expected_mode: Optional[str] = None) -> Dict[str, Any]:
    path = corpus_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Training corpus not found at {path}. Run 'python -m ticketlens.data --prepare-corpus' first."
        )
    with gzip.open(path, "rt", encoding="utf-8") as f:
        corpus = json.load(f)
    if expected_mode and corpus.get("mode") != expected_mode:
        raise ValueError(f"Corpus at {path} was prepared with mode={corpus.get('mode')!r}, expected {expected_mode!r}.")
    return corpus


def get_dataset_splits(
    cfg: Optional[Dict[str, Any]] = None,
    mode: str = "quick",
    use_fixture: bool = False,
    require_dataset: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Convenience helper to load raw and return preprocessed splits."""
    if cfg is None:
        cfg = load_config()
    raw_df = load_raw_dataset(cfg, use_fixture=use_fixture, require_dataset=require_dataset)
    if require_dataset and len(raw_df) < MIN_REAL_DATASET_ROWS:
        raise RuntimeError(
            f"Loaded only {len(raw_df)} rows (< {MIN_REAL_DATASET_ROWS}); refusing to train a production model on it."
        )
    return preprocess_data(raw_df, cfg, mode=mode)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare dataset for TicketLens")
    parser.add_argument("--mode", choices=["quick", "full", "all"], default="quick", help="Execution mode")
    parser.add_argument("--fixture", action="store_true", help="Force synthetic fixture usage")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download and cache the raw dataset (run as a separate build step to keep peak memory low)",
    )
    parser.add_argument("--require-dataset", action="store_true", help="Fail instead of falling back to the fixture")
    parser.add_argument(
        "--prepare-corpus",
        action="store_true",
        help=f"Clean, split and augment the data and write data/processed/{CORPUS_FILENAME} for 'train_baseline --from-corpus'",
    )
    args = parser.parse_args()

    config = load_config()
    if args.download_only:
        raw = load_raw_dataset(config, use_fixture=args.fixture, require_dataset=args.require_dataset)
        if args.require_dataset and len(raw) < MIN_REAL_DATASET_ROWS:
            raise SystemExit(f"Downloaded only {len(raw)} rows (< {MIN_REAL_DATASET_ROWS}).")
        logger.info("Raw dataset ready: %d rows.", len(raw))
    elif args.prepare_corpus:
        save_baseline_corpus(
            config,
            build_baseline_corpus(config, mode=args.mode, use_fixture=args.fixture, require_dataset=args.require_dataset),
        )
    else:
        get_dataset_splits(config, mode=args.mode, use_fixture=args.fixture, require_dataset=args.require_dataset)
