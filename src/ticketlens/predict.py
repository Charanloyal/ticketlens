"""Reproducible single and batch ticket inference engine."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
from ticketlens.config import get_device, is_production, load_config
from ticketlens.data import clean_and_redact_text
from ticketlens.itil_rules import CUE_WEIGHT, apply_itil_cues
from ticketlens.language import detect_non_english

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ticketlens.predict")


class TicketPredictor:
    """Inference engine supporting both TF-IDF baseline and DistilBERT (LoRA) models."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, model_type: Optional[str] = None):
        self.cfg = cfg or load_config()
        # Resolved only when a transformer is loaded, so torch is never imported otherwise.
        self.device: Any = "cpu"
        self.default_model_type = model_type or self.cfg.get("inference", {}).get("default_model", "transformer")
        self.confidence_threshold = float(self.cfg.get("inference", {}).get("confidence_threshold", 0.80))
        self.itil_cue_weight = float(self.cfg.get("inference", {}).get("itil_cue_weight", CUE_WEIGHT))

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
        if not (baseline_dir / "logistic_regression.joblib").exists():
            if is_production():
                # Training at startup is what previously exhausted the 512 MB Render instance;
                # production models must be built during the build step instead.
                logger.error(
                    "No baseline artifacts in %s and TICKETLENS_ENV=production: not training at startup. "
                    "Run 'python -m ticketlens.build' in the build step.",
                    baseline_dir,
                )
            else:
                try:
                    from ticketlens.train_baseline import train_baseline_model
                    logger.info("No baseline artifacts found. Bootstrapping quick baseline model for immediate serving...")
                    train_baseline_model(self.cfg, mode="quick", use_fixture=False)
                except Exception as e:
                    logger.warning("Could not auto-bootstrap baseline model: %s", e)

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
                # Heavy deep-learning stack is imported lazily to keep the baseline-only footprint small.
                from peft import PeftModel
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self.device = get_device()
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
                self.transformer_model = None
                self.transformer_tokenizer = None
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
            raw_probs = self._baseline_probabilities(cleaned_text)
            model_name = "TF-IDF + Logistic Regression"
        else:
            raw_probs = self._transformer_probabilities(cleaned_text)
            model_name = "DistilBERT (PEFT / LoRA)"

        return self._build_result(cleaned_text, raw_probs, model_name)

    def _build_result(self, cleaned_text: str, raw_probs: Dict[str, float], model_name: str) -> Dict[str, Any]:
        """Apply ITIL cue correction and language checks, then derive the routing decision."""
        probs, matched_cues = apply_itil_cues(raw_probs, cleaned_text, weight=self.itil_cue_weight)
        predicted_category = max(probs, key=probs.get)
        confidence = float(probs[predicted_category])

        warnings: List[str] = []
        language_warning = detect_non_english(cleaned_text)
        if language_warning:
            warnings.append(language_warning)

        # Non-English tickets are outside the training distribution, so never auto-route them.
        auto = confidence >= self.confidence_threshold and not language_warning
        decision = "Auto Accept" if auto else "Needs Review"

        return {
            "predicted_category": predicted_category,
            "confidence": round(confidence, 4),
            "decision": decision,
            "probabilities": {label: round(float(p), 4) for label, p in probs.items()},
            "model_type": model_name,
            "model_version": self.cfg.get("version", "0.1.0"),
            "confidence_threshold": self.confidence_threshold,
            "itil_cues": matched_cues,
            "warnings": warnings,
            "routing": self._route(predicted_category, auto),
            "processed_text": cleaned_text,
        }

    def _route(self, category: str, auto: bool) -> Dict[str, Any]:
        """Map the decision to the resolver queue and response target configured under `routing`."""
        routes = self.cfg.get("routing", {})
        suggested = routes.get(category, {})
        target = suggested if auto else routes.get("review", {})
        return {
            "queue": target.get("queue", category if auto else "Human triage"),
            "response_target": target.get("response_target", ""),
            "suggested_queue": suggested.get("queue", category),
            "auto_routed": auto,
        }

    def _baseline_probabilities(self, cleaned_text: str) -> Dict[str, float]:
        """Class probabilities from TF-IDF + Logistic Regression."""
        vec = self.baseline_vectorizer.transform([cleaned_text])
        probs = self.baseline_classifier.predict_proba(vec)[0]
        id2label = {int(k): v for k, v in self.baseline_labels["id2label"].items()}
        return {id2label[int(cls)]: float(p) for cls, p in zip(self.baseline_classifier.classes_, probs)}

    def _transformer_probabilities(self, cleaned_text: str) -> Dict[str, float]:
        """Class probabilities from the DistilBERT LoRA sequence classifier."""
        import torch

        inputs = self.transformer_tokenizer(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.transformer_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        id2label = {int(k): v for k, v in self.transformer_labels["id2label"].items()}
        return {id2label[i]: float(probs[i]) for i in range(len(probs))}


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
