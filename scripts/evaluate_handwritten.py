"""Evaluate the serving model on hand-written, real-world-style tickets.

The training data is long-form and synthetic, so validation scores overstate quality on short,
informal tickets. This script scores two hand-labelled sets:
  - data/fixtures/handwritten_eval.json    (development set; used while designing the ITIL cues)
  - data/fixtures/handwritten_holdout.json (holdout; written before tuning, never used to tune)

Usage: python scripts/evaluate_handwritten.py [--model baseline|transformer]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from ticketlens.predict import TicketPredictor  # noqa: E402

SETS = {
    "development": ROOT / "data" / "fixtures" / "handwritten_eval.json",
    "holdout": ROOT / "data" / "fixtures" / "handwritten_holdout.json",
}


def score(predictor: TicketPredictor, tickets: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    labels, preds, auto_correct, auto_total, errors = [], [], 0, 0, []
    for ticket in tickets:
        result = predictor.predict(ticket["text"], model_type=model)
        labels.append(ticket["label"])
        preds.append(result["predicted_category"])
        if result["decision"] == "Auto Accept":
            auto_total += 1
            auto_correct += int(result["predicted_category"] == ticket["label"])
        if result["predicted_category"] != ticket["label"]:
            errors.append({
                "text": ticket["text"],
                "expected": ticket["label"],
                "predicted": result["predicted_category"],
                "confidence": result["confidence"],
            })
    return {
        "tickets": len(tickets),
        "accuracy": round(accuracy_score(labels, preds), 4),
        "macro_f1": round(f1_score(labels, preds, average="macro", zero_division=0), 4),
        "auto_accept_rate": round(auto_total / len(tickets), 4),
        "auto_accept_precision": round(auto_correct / auto_total, 4) if auto_total else None,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=["baseline", "transformer"], default="baseline")
    parser.add_argument("--output", default=str(ROOT / "reports" / "handwritten_eval.json"))
    args = parser.parse_args()

    predictor = TicketPredictor()
    report: Dict[str, Any] = {"model": args.model}
    for cue_mode, weight in (("with_itil_cues", predictor.itil_cue_weight), ("model_only", 0.0)):
        predictor.itil_cue_weight = weight
        report[cue_mode] = {}
        for name, path in SETS.items():
            tickets = json.loads(path.read_text(encoding="utf-8"))
            report[cue_mode][name] = score(predictor, tickets, args.model)

    for cue_mode in ("model_only", "with_itil_cues"):
        for name in SETS:
            r = report[cue_mode][name]
            precision = "n/a" if r["auto_accept_precision"] is None else f"{r['auto_accept_precision']:.0%}"
            print(
                f"{cue_mode:15s} {name:12s} acc={r['accuracy']:.0%} macroF1={r['macro_f1']:.3f} "
                f"auto-accepted={r['auto_accept_rate']:.0%} (correct when auto-accepted: {precision})"
            )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
