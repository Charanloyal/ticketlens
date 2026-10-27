# TicketLens: Model Benchmark Summary

- **Dataset:** Tobi-Bueck/customer-support-tickets
- **Held-out test set:** 3,563 tickets (identical for every model)
- **Classes:** Change, Incident, Problem, Request

## Macro-F1 Benchmark

| Model | Train tickets | Accuracy | Macro-F1 | Weighted-F1 | Hand-written holdout acc. | P50 latency |
| :--- | ---: | :---: | :---: | :---: | :---: | ---: |
| **TF-IDF + Logistic Regression** | 7,000 | 0.8266 | **0.8352** | 0.8270 | 33% | 0.78 ms |
| **Keras CNN (TensorFlow)** | 7,000 | 0.7884 | **0.8095** | 0.7932 | 46% | 7.71 ms |
| **DistilBERT + LoRA (PyTorch/PEFT)** | 7,000 | 0.7920 | **0.7790** | 0.7777 | 62% | 56.66 ms |
| **Served TF-IDF (all data, model only)** | 16,622 | 0.8712 | **0.8782** | 0.8717 | 38% | 0.79 ms |
| **Served TF-IDF + ITIL cues (production)** | 16,622 | 0.8549 | **0.8635** | 0.8565 | 79% | 0.79 ms |

## Per-Class Precision / Recall

| Model | Change P / R | Incident P / R | Problem P / R | Request P / R |
| :--- | :---: | :---: | :---: | :---: |
| TF-IDF + Logistic Regression | 0.95 / 0.95 | 0.80 / 0.78 | 0.60 / 0.62 | 0.99 / 0.99 |
| Keras CNN (TensorFlow) | 0.95 / 0.93 | 0.81 / 0.65 | 0.51 / 0.71 | 0.98 / 0.99 |
| DistilBERT + LoRA (PyTorch/PEFT) | 0.94 / 0.96 | 0.71 / 0.85 | 0.54 / 0.34 | 0.98 / 0.98 |
| Served TF-IDF (all data, model only) | 0.97 / 0.97 | 0.85 / 0.83 | 0.70 / 0.72 | 0.99 / 0.99 |
| Served TF-IDF + ITIL cues (production) | 0.95 / 0.95 | 0.86 / 0.79 | 0.65 / 0.74 | 0.98 / 0.99 |

## Notes

- TF-IDF, Keras and DistilBERT are trained on the **same** training subset and scored on the **same** test tickets.
- The served TF-IDF model uses all training data plus short-text augmentation; the production row adds the ITIL cue correction.
- Hand-written holdout accuracy uses the 24 hand-labelled tickets in `data/fixtures/handwritten_holdout.json`, never used for tuning (the 48-ticket development set used to design the ITIL cues is reported separately in the JSON).
- See `error_analysis.md` for confusion pairs, confidence-gate precision/coverage, length-bucket parity and error examples.

## Figures

- `reports/figures/model_comparison_f1.png`
- `reports/figures/precision_coverage.png`
- `reports/figures/confusion_matrix_tfidf_benchmark.png`
- `reports/figures/confusion_matrix_keras.png`
- `reports/figures/confusion_matrix_transformer.png`
- `reports/figures/confusion_matrix_baseline.png`
- `reports/figures/confusion_matrix_baseline_itil.png`
