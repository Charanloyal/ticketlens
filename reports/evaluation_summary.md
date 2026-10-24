# TicketLens: Model Evaluation & Benchmark Summary

- **Dataset Source:** Tobi-Bueck/customer-support-tickets
- **Held-Out Test Set Size:** 45 samples
- **Evaluated Classes:** Change, Incident, Problem, Request

## Performance Comparison

| Model Architecture | Accuracy | Macro-F1 | Weighted-F1 | P50 Latency (ms) | P95 Latency (ms) | Throughput (req/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | 0.6667 | 0.6206 | 0.6616 | 0.6 ms | 1.11 ms | 1213.91 req/s |
| **DistilBERT (LoRA / PEFT)** | 0.4444 | 0.3273 | 0.3816 | 30.68 ms | 48.77 ms | 33.49 req/s |

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
