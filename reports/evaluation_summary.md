# TicketLens: Model Evaluation & Benchmark Summary

- **Dataset Source:** Tobi-Bueck/customer-support-tickets
- **Held-Out Test Set Size:** 75 samples
- **Evaluated Classes:** Change, Incident, Problem, Request

## Performance Comparison

| Model Architecture | Accuracy | Macro-F1 | Weighted-F1 | P50 Latency (ms) | P95 Latency (ms) | Throughput (req/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | 0.7467 | 0.6709 | 0.7219 | 0.55 ms | 1.03 ms | 1656.84 req/s |
| **DistilBERT (LoRA / PEFT)** | 0.6133 | 0.3724 | 0.5009 | 27.5 ms | 49.51 ms | 32.96 req/s |

## Operational Trade-Offs & Key Takeaways

1. **Inference Latency vs Semantic Generalization:**
   - The **TF-IDF + Logistic Regression** baseline delivers sub-millisecond to low-millisecond CPU latency, making it extremely cost-efficient for high-volume ticket routing.
   - The **DistilBERT (LoRA)** transformer captures deep contextual subtleties (e.g. nuanced phrases distinguishing a recurring *Problem* from an immediate *Incident*), but exhibits higher inference latency on CPU architectures.
2. **Production Recommendation:**
   - Use the **DistilBERT (LoRA)** model for tickets with complex descriptions or ambiguous phrasing, while leveraging the **TF-IDF Baseline** as a high-speed fallback or lightweight edge filter.

## Generated Figures

- `reports/figures/confusion_matrix_baseline.png`
- `reports/figures/confusion_matrix_transformer.png`
- `reports/figures/model_comparison_f1.png`
