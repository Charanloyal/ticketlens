"""Tests for ITIL cue correction, language detection, config overrides and short-ticket handling."""

import copy

import pandas as pd
import pytest
from ticketlens.config import load_config
from ticketlens.data import preprocess_data
from ticketlens.itil_rules import apply_itil_cues, match_itil_cues
from ticketlens.language import detect_non_english
from ticketlens.train_baseline import short_text_variants


@pytest.mark.parametrize(
    "text, category",
    [
        ("send me last month's invoice", "Request"),
        ("I need access to the shared finance drive", "Request"),
        ("The server crashed three times this week, please find the root cause", "Problem"),
        ("Please upgrade our database server to PostgreSQL 16", "Change"),
        ("The website is down right now", "Incident"),
    ],
)
def test_itil_cues_match_expected_category(text, category):
    assert category in match_itil_cues(text)


def test_itil_cues_leave_probabilities_unchanged_without_match():
    probs = {"Incident": 0.4, "Request": 0.3, "Problem": 0.2, "Change": 0.1}
    adjusted, matches = apply_itil_cues(probs, "lorem ipsum dolor")
    assert matches == {}
    assert adjusted == probs


def test_itil_cues_shift_low_confidence_prediction():
    probs = {"Incident": 0.2, "Request": 0.3, "Problem": 0.45, "Change": 0.05}
    adjusted, matches = apply_itil_cues(probs, "send me last month's invoice")
    assert "Request" in matches
    assert max(adjusted, key=adjusted.get) == "Request"
    assert sum(adjusted.values()) == pytest.approx(1.0)


def test_itil_cues_do_not_override_very_confident_model():
    probs = {"Incident": 0.005, "Request": 0.005, "Problem": 0.98, "Change": 0.01}
    adjusted, _ = apply_itil_cues(probs, "please share the invoice")
    assert max(adjusted, key=adjusted.get) == "Problem"


def test_itil_cues_disabled_with_zero_weight():
    probs = {"Incident": 0.2, "Request": 0.3, "Problem": 0.45, "Change": 0.05}
    adjusted, _ = apply_itil_cues(probs, "send me last month's invoice", weight=0)
    assert adjusted == probs


@pytest.mark.parametrize(
    "text",
    [
        "VPN TLS error 504",
        "Outlook crashes every time I open it",
        "Please reset my password, I forgot it",
        "Der Drucker",  # too short to judge
        "Same e-mail issue as yesterday",
        "Outlook e-mail sync fails as usual",
        "Send invoice as PDF e.g. via e-mail",
        "VPN o365 login as admin e-mail",
        "Please die hard reset den server",
        "Install the MIT licensed package on the OS image from example.com",
    ],
)
def test_english_text_not_flagged(text):
    assert detect_non_english(text) is None


@pytest.mark.parametrize(
    "text, language",
    [
        ("Der Drucker im Büro funktioniert nicht und ich kann nicht drucken", "German"),
        ("No puedo acceder al correo y el servidor no responde, por favor ayuda", "Spanish"),
        ("Bonjour, je ne peux pas me connecter au VPN depuis ce matin, merci", "French"),
    ],
)
def test_foreign_language_flagged(text, language):
    warning = detect_non_english(text)
    assert warning is not None and language in warning


def test_non_latin_script_flagged():
    assert "non-Latin" in detect_non_english("打印机无法连接到网络，请帮忙解决")


def test_confidence_threshold_env_override(monkeypatch):
    monkeypatch.setenv("TICKETLENS_CONFIDENCE_THRESHOLD", "0.65")
    assert load_config()["inference"]["confidence_threshold"] == 0.65


@pytest.mark.parametrize("value", ["abc", "1.5"])
def test_confidence_threshold_env_override_rejects_invalid(monkeypatch, value):
    monkeypatch.setenv("TICKETLENS_CONFIDENCE_THRESHOLD", value)
    with pytest.raises(ValueError):
        load_config()


def test_preprocess_keeps_tickets_without_subject(base_config, tmp_path):
    cfg = copy.deepcopy(base_config)
    cfg["paths"]["processed_data_dir"] = str(tmp_path)
    rows = []
    for i, label in enumerate(["Incident", "Request", "Problem", "Change"] * 8):
        rows.append({
            "subject": None if i % 2 else f"Subject {i}",
            "body": f"Ticket body number {i} describing the {label.lower()} in enough detail.",
            "type": label,
        })
    train_df, val_df, test_df, _ = preprocess_data(pd.DataFrame(rows), cfg, mode="all")
    assert len(train_df) + len(val_df) + len(test_df) == len(rows)


def test_short_text_variants_use_subject_and_first_sentence(base_config):
    train_df = pd.DataFrame({
        "subject": ["Invoice copy needed", None],
        "body": ["Please send the March invoice. Thanks a lot!", "VPN is down for everyone.<br>Since 9am."],
        "label": [1, 0],
    })
    texts, labels = short_text_variants(train_df, base_config)
    assert "Invoice copy needed" in texts
    assert "Please send the March invoice." in texts
    assert "VPN is down for everyone." in texts
    assert len(texts) == len(labels)


def _offline_cfg(base_config, tmp_path):
    cfg = copy.deepcopy(base_config)
    cfg["paths"]["raw_data_dir"] = str(tmp_path / "raw")
    cfg["paths"]["processed_data_dir"] = str(tmp_path / "processed")
    return cfg


def _failing_hf(monkeypatch):
    import datasets
    import ticketlens.data as data_module

    def boom(*args, **kwargs):
        raise ConnectionError("simulated Hugging Face outage")

    monkeypatch.setattr(datasets, "load_dataset", boom)
    monkeypatch.setattr(data_module.time, "sleep", lambda s: None)


def test_require_dataset_refuses_fixture_fallback(monkeypatch, base_config, tmp_path):
    from ticketlens.data import get_dataset_splits

    _failing_hf(monkeypatch)
    with pytest.raises(RuntimeError, match="Refusing to fall back"):
        get_dataset_splits(_offline_cfg(base_config, tmp_path), mode="all", require_dataset=True)


def test_without_require_dataset_falls_back_to_fixture(monkeypatch, base_config, tmp_path):
    from ticketlens.data import load_raw_dataset

    _failing_hf(monkeypatch)
    df = load_raw_dataset(_offline_cfg(base_config, tmp_path))
    assert 0 < len(df) < 100


def test_require_dataset_rejects_tiny_dataset(base_config, tmp_path):
    from ticketlens.data import get_dataset_splits

    cfg = _offline_cfg(base_config, tmp_path)
    (tmp_path / "raw").mkdir()
    tiny = pd.DataFrame({"type": ["Incident"] * 10, "subject": ["s"] * 10, "body": ["body text here"] * 10, "language": ["en"] * 10})
    tiny.to_parquet(tmp_path / "raw" / "customer_support_tickets_en.parquet", index=False)
    with pytest.raises(RuntimeError, match="refusing to train"):
        get_dataset_splits(cfg, mode="all", require_dataset=True)


def test_production_never_trains_at_startup(monkeypatch, base_config, tmp_path):
    import ticketlens.train_baseline as tb
    from ticketlens.predict import TicketPredictor

    cfg = copy.deepcopy(base_config)
    cfg["paths"]["model_dir"] = str(tmp_path / "models")
    monkeypatch.setenv("TICKETLENS_ENV", "production")
    monkeypatch.setattr(tb, "train_baseline_model", lambda *a, **k: pytest.fail("must not train in production"))
    predictor = TicketPredictor(cfg=cfg)
    assert not predictor.is_ready
