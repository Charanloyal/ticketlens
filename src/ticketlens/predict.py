"""Reproducible single and batch ticket inference engine."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ticketlens.config import get_device, load_config
from ticketlens.data import clean_and_redact_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ticketlens.predict")


class TicketPredictor:
    """Inference engine supporting both TF-IDF baseline and DistilBERT (LoRA) models."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, model_type: Optional[str] = None):
        self.cfg = cfg or load_config()
        self.device = get_device()
        self.default_model_type = model_type or self.cfg.get("inference", {}).get("default_model", "transformer")
        self.confidence_threshold = float(self.cfg.get("inference", {}).get("confidence_threshold", 0.80))

        self.baseline_vectorizer = None
        self.baseline_classifier = None
        self.baseline_labels = None

        self.transformer_model = None
        self.transformer_tokenizer = None
        self.transformer_labels = None

        self._load_available_models()

    def _load_available_models(self) -> None:
        """Load baseline and/or transformer models if artifacts exist."""
        # Load Baseline
        baseline_dir = Path(self.cfg["paths"]["model_dir"]) / "baseline"
        if (baseline_dir / "logistic_regression.joblib").exists():
            try:
                self.baseline_vectorizer = joblib.load(baseline_dir / "tfidf_vectorizer.joblib")
                self.baseline_classifier = joblib.load(baseline_dir / "logistic_regression.joblib")
                with open(baseline_dir / "labels.json", "r", encoding="utf-8") as f:
                    self.baseline_labels = json.load(f)
                logger.info("Loaded baseline model from %s", baseline_dir)
            except Exception as e:
                logger.warning("Could not load baseline model: %s", e)

        # Load Transformer
        transformer_dir = Path(self.cfg["paths"]["model_dir"]) / "transformer"
        if transformer_dir.exists() and any(transformer_dir.iterdir()):
            try:
                with open(transformer_dir / "labels.json", "r", encoding="utf-8") as f:
                    self.transformer_labels = json.load(f)

                num_classes = len(self.transformer_labels["label2id"])
                self.transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_dir)

                if (transformer_dir / "adapter_config.json").exists():
                    base_name = self.cfg["transformer"].get("pretrained_model", "distilbert/distilbert-base-uncased")
                    base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=num_classes)
                    self.transformer_model = PeftModel.from_pretrained(base, transformer_dir)
                else:
                    self.transformer_model = AutoModelForSequenceClassification.from_pretrained(transformer_dir)

                self.transformer_model.to(self.device)
                self.transformer_model.eval()
                logger.info("Loaded transformer model from %s onto %s", transformer_dir, self.device)
            except Exception as e:
                logger.warning("Could not load transformer model: %s", e)

    @property
    def is_ready(self) -> bool:
        """Return True if at least one model is loaded and ready."""
        return self.transformer_model is not None or self.baseline_classifier is not None

    def get_loaded_models(self) -> List[str]:
        """Return list of currently active model types."""
        models = []
        if self.baseline_classifier is not None:
            models.append("baseline")
        if self.transformer_model is not None:
            models.append("transformer")
        return models

    def predict(
        self,
        text: str,
        model_type: Optional[str] = None,
        redact_pii: bool = True,
    ) -> Dict[str, Any]:
        """
        Classify single ticket text.
        Returns predicted category, class probabilities, confidence, decision, and model metadata.
        """
        if not self.is_ready:
            raise RuntimeError(
                "No trained model artifacts found. Please run baseline or transformer training first."
            )

        selected_model = model_type or self.default_model_type

        # Fallback logic if requested model is unavailable
        if selected_model == "transformer" and self.transformer_model is None:
            if self.baseline_classifier is not None:
                logger.info("Transformer model requested but not loaded. Falling back to baseline.")
                selected_model = "baseline"
            else:
                raise RuntimeError("Transformer model is not trained/loaded.")
        elif selected_model == "baseline" and self.baseline_classifier is None:
            if self.transformer_model is not None:
                logger.info("Baseline model requested but not loaded. Falling back to transformer.")
                selected_model = "transformer"
            else:
                raise RuntimeError("Baseline model is not trained/loaded.")

        cleaned_text = clean_and_redact_text(text, redact_pii=redact_pii)

        if selected_model == "baseline":
            return self._predict_baseline(cleaned_text)
        else:
            return self._predict_transformer(cleaned_text)

    def _predict_baseline(self, cleaned_text: str) -> Dict[str, Any]:
        """Inference with TF-IDF + Logistic Regression."""
        vec = self.baseline_vectorizer.transform([cleaned_text])
        probs = self.baseline_classifier.predict_proba(vec)[0]
        pred_idx = int(np.argmax(probs))

        id2label = {int(k): v for k, v in self.baseline_labels["id2label"].items()}
        predicted_category = id2label[pred_idx]
        confidence = float(probs[pred_idx])

        probabilities = {id2label[i]: round(float(probs[i]), 4) for i in range(len(probs))}
        decision = "Auto Accept" if confidence >= self.confidence_threshold else "Needs Review"

        return {
            "predicted_category": predicted_category,
            "confidence": round(confidence, 4),
            "decision": decision,
            "probabilities": probabilities,
            "model_type": "TF-IDF + Logistic Regression",
            "model_version": self.cfg.get("version", "0.1.0"),
            "confidence_threshold": self.confidence_threshold,
        }

    def _predict_transformer(self, cleaned_text: str) -> Dict[str, Any]:
        """Inference with DistilBERT LoRA sequence classifier."""
        inputs = self.transformer_tokenizer(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.transformer_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        id2label = {int(k): v for k, v in self.transformer_labels["id2label"].items()}
        predicted_category = id2label[pred_idx]
        confidence = float(probs[pred_idx])

        probabilities = {id2label[i]: round(float(probs[i]), 4) for i in range(len(probs))}
        decision = "Auto Accept" if confidence >= self.confidence_threshold else "Needs Review"

        return {
            "predicted_category": predicted_category,
            "confidence": round(confidence, 4),
            "decision": decision,
            "probabilities": probabilities,
            "model_type": "DistilBERT (PEFT / LoRA)",
            "model_version": self.cfg.get("version", "0.1.0"),
            "confidence_threshold": self.confidence_threshold,
        }


def main():
    parser = argparse.ArgumentParser(description="Predict IT ticket category")
    parser.add_argument("text", nargs="?", default="VPN connection keeps timing out with TLS error", help="Ticket text to classify")
    parser.add_argument("--model", choices=["transformer", "baseline"], default=None, help="Model architecture")
    args = parser.parse_args()

    predictor = TicketPredictor()
    result = predictor.predict(args.text, model_type=args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
