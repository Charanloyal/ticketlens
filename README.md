# TicketLens: IT Support Ticket Classification & Routing System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Code License: Apache 2.0](https://img.shields.io/badge/Code%20License-Apache%202.0-blue.svg)](LICENSE)
[![Dataset License: CC BY-NC 4.0](https://img.shields.io/badge/Dataset-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Framework: FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Keras](https://img.shields.io/badge/Baseline-TensorFlow%20%2F%20Keras-FF6F00.svg)](https://keras.io/)
[![Benchmark](https://github.com/Charanloyal/ticketlens/actions/workflows/benchmark.yml/badge.svg)](https://github.com/Charanloyal/ticketlens/actions/workflows/benchmark.yml)
[![CI](https://github.com/Charanloyal/ticketlens/actions/workflows/ci.yml/badge.svg)](https://github.com/Charanloyal/ticketlens/actions/workflows/ci.yml)
[![Tests: Pytest](https://img.shields.io/badge/Tests-66-brightgreen.svg)](https://docs.pytest.org/)
[![Live demo](https://img.shields.io/badge/Live%20demo-ticketlens.onrender.com-46E3B7.svg?logo=render)](https://ticketlens.onrender.com/)

**▶ Live demo: [ticketlens.onrender.com](https://ticketlens.onrender.com/)** · [Swagger API](https://ticketlens.onrender.com/docs) — hosted on Render's free plan, so the first visit after ~15 idle minutes can return `503` or take a minute or more while the service wakes up. The live demo serves the **TF-IDF + Logistic Regression** model (plus ITIL keyword cues), not DistilBERT.

**TicketLens** is an end-to-end IT service desk ticket classification and triage project with a dark-mode web dashboard. It classifies tickets into the ITIL categories (**Incident**, **Request**, **Problem**, **Change**) and compares three approaches on **macro-F1** over the identical held-out test set: a **TF-IDF + Logistic Regression** baseline, a **Keras (TensorFlow) CNN**, and a pre-trained **DistilBERT** fine-tuned with **LoRA (Hugging Face Transformers + PEFT, PyTorch)**. On that test set the TF-IDF model scored highest (macro-F1 0.8352 vs. 0.8095 for Keras and 0.7790 for DistilBERT + LoRA, same 7,000 training tickets); the deployed model is TF-IDF + Logistic Regression retrained on the full training split, which also fits Render's 512 MB free plan.

The project includes precision/recall **error analysis**, **SHAP** interpretability, a **bias check** (class balance, performance parity by ticket length, PII audit), regex PII scrubbing (emails, phone numbers, IP addresses), latency benchmarking, and a **FastAPI** service with confidence-gated routing, packaged in **Docker** (a lightweight TF-IDF image that matches the Render deploy, and a separate DistilBERT + LoRA image).

**Status:** complete as a portfolio / learning project; the benchmark is reproducible through the [Benchmark workflow](.github/workflows/benchmark.yml).

**Data:** the tickets come from a public dataset of synthetic (machine-generated) customer-support emails, not from a real service desk, and the dataset is licensed for non-commercial use only (CC BY-NC 4.0). Results on real tickets will differ; see [Limitations](#limitations).

**Why it matters.** People who raise IT tickets write free text; each ITIL category goes to a different team with a
different deadline (Incident → on-call support within hours, Request → fulfilment, Problem → root-cause engineers,
Change → Change Advisory Board). TicketLens automates that first triage step: it classifies the ticket, auto-routes it
when confident, and sends uncertain or non-English tickets to a human with its suggestion attached.

| Live triage (PII redacted, routed to the on-call queue) | Benchmark report card (same 3,563 test tickets) |
| :---: | :---: |
| ![TicketLens live triage](docs/dashboard.png) | ![TicketLens benchmark tab](docs/benchmarks.png) |

---

## Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Ingestion["1. Data Ingestion & Governance"]
        HF[Hugging Face Hub: Tobi-Bueck/customer-support-tickets] --> ParquetCache[Local Parquet Cache: data/raw/]
        ParquetCache --> FilterEN[Filter English Tickets]
        FilterEN --> PII[PII Sanitizer: Emails, Phone, IPs Redacted]
        PII --> Dedup[Deduplication: Prevent Cross-Split Leakage]
        Dedup --> Split[Stratified Split of 23,747 tickets: 70% Train / 15% Val / 15% Test - test set fixed for every model]
    end

    subgraph Modeling["2. Model Training Pipelines"]
        Split -->|train.parquet| BaselinePipeline[Baseline Pipeline: TF-IDF + Logistic Regression]
        Split -->|train.parquet| KerasPipeline[Keras Baseline: TextVectorization + Embedding + Conv1D]
        Split -->|train.parquet| TransformerPipeline[Transformer Pipeline: DistilBERT + LoRA / PEFT]
        BaselinePipeline --> BaselineArtifacts[Save Vectorizer & Weights: artifacts/models/baseline/]
        KerasPipeline --> KerasArtifacts[Save Keras model: artifacts/models/keras/]
        TransformerPipeline --> TransformerArtifacts[Save LoRA Adapter & Tokenizer: artifacts/models/transformer/]
    end

    subgraph Evaluation["3. Evaluation, Explainability & Audit"]
        Split -->|test.parquet| Evaluator[Evaluation Engine: Macro-F1, Precision/Recall, Latency]
        BaselineArtifacts --> Evaluator
        KerasArtifacts --> Evaluator
        TransformerArtifacts --> Evaluator
        Evaluator --> ConfMatrix[Confusion Matrices: reports/figures/]
        Evaluator --> ErrorAnalysis[Error Analysis: reports/error_analysis.md]
        Evaluator --> SHAP[SHAP LinearExplainer: reports/shap_summary.json]
        Evaluator --> BiasAudit[Bias & Distribution Audit: reports/bias_and_fairness_audit.json]
    end

    subgraph Serving["4. Serving"]
        BaselineArtifacts -->|"served on Render + Dockerfile"| FastAPI[FastAPI Service: ticketlens.api:app]
        TransformerArtifacts -.->|"only in Dockerfile.transformer"| FastAPI
        FastAPI --> PredictEndpoint["POST /predict"]
        PredictEndpoint --> ConfidenceRouter{"Confidence >= 0.80?"}
        ConfidenceRouter -->|Yes| AutoAccept["Decision: Auto Accept"]
        ConfidenceRouter -->|No| NeedsReview["Decision: Needs Review (Human-in-the-Loop)"]
    end
```

---

## Dataset Source, Schema & License

- **Source:** [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (published by Softoft; also on Kaggle).
- **Nature of the data:** synthetic, machine-generated customer-support emails, not tickets from a real service desk. They are
  long-form and stylistically uniform, and the Incident/Problem labels are applied inconsistently (see the error analysis).
  The PII scan found no emails, phone numbers or IP addresses in a 500-ticket raw sample.
- **License:** [Creative Commons Attribution Non-Commercial 4.0 International (`CC-BY-NC-4.0`)](https://creativecommons.org/licenses/by-nc/4.0/).
- **DOI:** `10.57967/hf/6184`
- **Language:** English (`language == 'en'`).
- **Target Categories (ITIL Standards):**
  1. `Incident`: Unplanned service interruptions (e.g., VPN disconnections, hardware crashes, screen flicker).
  2. `Request`: Formal service and access requests (e.g., credentials, hardware upgrades, secondary monitors).
  3. `Problem`: Root-cause investigations of recurring outages (e.g., memory leaks, database connection pool exhaustion).
  4. `Change`: Scheduled configuration and infrastructure adjustments (e.g., firewall rule modification, node pool rolling upgrade).
- **Input Features:** `subject` + `body` combined and sanitized into a single context stream (`text`).

---

## Project Structure

```
TicketLens/
├── configs/
│   ├── config.yaml               # Centralized hyperparameters, dataset, and threshold configs
│   └── routing.yaml              # Resolver queue + response target per ITIL category
├── data/
│   ├── fixtures/
│   │   ├── synthetic_tickets.json # Offline test fixture for reproducible unit tests
│   │   ├── handwritten_eval.json  # 48 hand-written tickets (development set for ITIL cues)
│   │   └── handwritten_holdout.json # 24 hand-written tickets (holdout, never used for tuning)
│   ├── raw/                      # Cached raw parquet dataset (git-ignored)
│   └── processed/                # Preprocessed train, val, test parquet splits & label mapping (git-ignored)
├── artifacts/                    # Trained models (git-ignored; created by the training commands)
│   └── models/
│       ├── baseline/             # Served TF-IDF vectorizer + Logistic Regression joblib artifacts
│       ├── tfidf_benchmark/      # TF-IDF trained on the 7,000-ticket benchmark subset
│       ├── keras/                # Keras CNN
│       └── transformer/          # DistilBERT LoRA adapter weights, tokenizer, & labels
├── docs/                         # README screenshots of the dashboard
├── reports/
│   ├── figures/                  # Confusion matrices, model comparison, precision/coverage plots
│   ├── metrics_*.json            # Per-model validation (metrics_<model>.json) and test (metrics_test_<model>.json) metrics
│   ├── model_comparison.json     # Held-out test set comparison of every model (source of the benchmark table)
│   ├── evaluation_summary.md     # Benchmark summary generated by ticketlens.evaluate
│   ├── error_analysis.md         # Confusion pairs, confidence-gate precision/coverage, length parity
│   ├── handwritten_eval.json     # Served model on the hand-written tickets (with / without ITIL cues)
│   ├── benchmark_latency.json    # Output of scripts/benchmark_latency.py (local run; hardware not recorded)
│   ├── top_features_per_class.json # Top positive and negative n-gram coefficients
│   ├── shap_summary.json         # SHAP mean absolute attributions
│   ├── bias_and_fairness_audit.json
│   └── explainability_and_bias_report.md # Dataset balance, PII scan, and attribution report
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb # Interactive EDA and ticket distribution plots
├── src/
│   └── ticketlens/
│       ├── __init__.py
│       ├── config.py             # Path resolution and CUDA/CPU auto-detection
│       ├── data.py               # Data ingestion, schema validation, PII scrubbing, & splitting
│       ├── train_baseline.py     # TF-IDF + Logistic Regression training
│       ├── build.py              # Production build (download → corpus → train, one process each)
│       ├── train_transformer.py  # DistilBERT sequence classification with PEFT / LoRA
│       ├── train_keras.py        # Keras (TensorFlow) CNN baseline
│       ├── evaluate.py           # Benchmark on the shared test set + precision/recall error analysis
│       ├── explain.py            # SHAP attribution and dataset bias audit
│       ├── predict.py            # Reproducible CLI and inference engine
│       ├── itil_rules.py         # ITIL keyword cues that correct predictions on short tickets
│       ├── language.py           # Dependency-free non-English input detection
│       ├── api.py                # FastAPI inference service (/health, /metadata, /predict)
│       └── templates/index.html  # Web dashboard served at /
├── tests/
│   ├── conftest.py               # Pytest fixtures and TestClient configuration
│   ├── test_data.py              # Data cleaning, PII redaction, schema, and leakage tests
│   ├── test_models.py            # Baseline fit, tokenization, and metric tests
│   ├── test_api.py               # API health, metadata, auth, validation, and prediction tests
│   ├── test_quality.py           # ITIL cues, language detection, config overrides, augmentation
│   └── test_benchmark.py         # Fixed test split, Keras baseline, evaluation & error analysis
├── scripts/
│   ├── run_pipeline.py           # End-to-end pipeline runner
│   ├── benchmark_latency.py      # Latency & throughput benchmarking script
│   ├── evaluate_handwritten.py   # Accuracy on hand-written, real-world-style tickets
│   └── run.ps1                   # Windows PowerShell automation helper
├── main.py                       # ASGI entrypoint used by Render and Docker (uvicorn main:app)
├── Dockerfile                    # Lightweight TF-IDF serving image (same as Render)
├── Dockerfile.transformer        # DistilBERT + LoRA serving image (FastAPI)
├── .dockerignore                 # Container build exclusion rules
├── .env.example                  # Example environment variables
├── .gitignore                    # Git tracking ignore rules
├── requirements.txt              # Full dependency set (training, transformer, explainability)
├── requirements-render.txt       # Lightweight serving dependencies for Render (no torch)
├── render.yaml                   # Render Blueprint (free plan)
├── .github/workflows/ci.yml      # CI: replays the Render build, tests, smoke test, both Docker images
├── .github/workflows/benchmark.yml # Trains TF-IDF, Keras & DistilBERT, evaluates, publishes the "models" release
├── LICENSE                       # Apache License 2.0
├── pyproject.toml / setup.py     # Build configuration and pytest options
├── Makefile                      # Cross-platform development targets
└── README.md                     # Project documentation
```

---

## Installation & Setup

### Windows PowerShell

```powershell
# 1. Clone the repository
git clone https://github.com/Charanloyal/ticketlens.git
cd ticketlens

# 2. Create isolated virtual environment
python -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies and local package in editable mode
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` installs the full stack (PyTorch, Transformers, SHAP, and TensorFlow on Python < 3.13). To only serve
the TF-IDF model, `pip install -r requirements-render.txt pytest httpx` is enough (this is what CI and Render use).

### Linux / macOS

```bash
git clone https://github.com/Charanloyal/ticketlens.git
cd ticketlens
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## Quick-Start Commands

| Command | Purpose |
| :--- | :--- |
| `pytest tests/ -v` | Run the complete automated test suite (66 tests; the Keras test is skipped when TensorFlow is not installed) |
| `python -m ticketlens.train_baseline --mode quick` | Train TF-IDF baseline model (quick verification run) |
| `python -m ticketlens.build` | Production build exactly as on Render: download dataset, prepare corpus, train serving model (each step in its own process) |
| `python -m ticketlens.train_baseline --mode all` | Train the serving TF-IDF model on the full dataset in a single process (local development) |
| `python scripts/evaluate_handwritten.py` | Score the serving model on hand-written tickets (with/without ITIL cues) |
| `python -m ticketlens.train_baseline --mode full --model-name tfidf_benchmark --no-augment` | TF-IDF baseline on the benchmark subset (7,000 tickets) |
| `python -m ticketlens.train_keras --mode full` | Keras (TensorFlow) CNN baseline on the same subset |
| `python -m ticketlens.train_transformer --mode full` | Fine-tune DistilBERT with LoRA on the same subset (~1 hour on a 4-vCPU CPU runner; 59 min in the published run) |
| `python -m ticketlens.evaluate` | Benchmark every trained model on the identical test set + error analysis |
| `make benchmark` | All of the above plus SHAP / bias analysis |
| `python -m ticketlens.explain` | Generate SHAP attributions and bias audit report |
| `python scripts/benchmark_latency.py` | Benchmark inference latency (P50, P95, P99) & throughput |
| `python -m ticketlens.predict "VPN dropped with TLS error"` | Classify a sample ticket from terminal CLI |
| `uvicorn ticketlens.api:app --reload --port 8000` | Start the FastAPI server locally (Render/Docker use `uvicorn main:app`) |

Alternatively, use the convenience helper scripts:
- **Windows:** `.\scripts\run.ps1 test`, `.\scripts\run.ps1 train-quick`, `.\scripts\run.ps1 api`
- **Linux/macOS:** `make test`, `make train-quick`, `make api`

### Testing

```bash
pytest tests/ -q -rs
```

Most tests use the 8-ticket fixture in `data/fixtures/synthetic_tickets.json` or small in-memory data. The API tests
start the app, which (outside `TICKETLENS_ENV=production`) trains a quick TF-IDF model if none exists: it tries to download
the Hugging Face dataset and falls back to the fixture when offline. The Keras test is skipped when TensorFlow is not
installed, and the tokenization test needs the `distilbert-base-uncased` tokenizer (network or local Hugging Face cache;
set `HF_HUB_OFFLINE=1` to use the cache). The [CI workflow](.github/workflows/ci.yml) runs the suite on every push with the slim Render dependencies, and
the [Benchmark workflow](.github/workflows/benchmark.yml) runs it again with the full stack after training. A local run
writes artifacts to `artifacts/` and `data/` (git-ignored) and overwrites `reports/metrics_baseline.json` with a
quick-mode run; restore it with `git restore reports/` if needed.

---

## Benchmark Results (Macro-F1, identical held-out test set)

Produced by the [Benchmark workflow](.github/workflows/benchmark.yml) (GitHub Actions, CPU, Python 3.11) and
published with the trained models as the [`models` release](https://github.com/Charanloyal/ticketlens/releases/tag/models).
Full reports: [`reports/evaluation_summary.md`](reports/evaluation_summary.md) and
[`reports/error_analysis.md`](reports/error_analysis.md).

**Protocol.** The 23,747 cleaned, deduplicated English tickets are split once (70/15/15, stratified, seed 42).
Every model is scored on the **same 3,563 test tickets**. The three architectures are trained on the **same
7,000-ticket training subset** (CPU budget for the transformer) with a 1,500-ticket validation set for model
selection; the served TF-IDF model uses the whole training split (16,622 tickets plus short-text augmentation), so the
"Served" rows are **not** a like-for-like comparison with the three 7,000-ticket rows. All numbers in this section come from
[`reports/model_comparison.json`](reports/model_comparison.json) (Benchmark run on commit `73c90e4`).

**Result.** On the like-for-like comparison TF-IDF + Logistic Regression has the best Macro-F1 (0.8352), ahead of the
Keras CNN (0.8095) and DistilBERT + LoRA (0.7790). DistilBERT does best only on the small hand-written holdout (see below).

| Model | Framework | Train tickets | Accuracy | **Macro-F1** | Weighted-F1 | Hand-written holdout (24) | P50 latency |
| :--- | :--- | ---: | :---: | :---: | :---: | :---: | ---: |
| TF-IDF + Logistic Regression | scikit-learn | 7,000 | 0.8266 | **0.8352** | 0.8270 | 33.3% | 0.8 ms |
| Keras CNN (TextVectorization + Embedding + Conv1D) | TensorFlow / Keras | 7,000 | 0.7884 | **0.8095** | 0.7932 | 45.8% | 7.7 ms |
| DistilBERT + LoRA (r=8, q/v, 1.09% trainable) | PyTorch + HF Transformers + PEFT | 7,000 | 0.7920 | **0.7790** | 0.7777 | **62.5%** | 57 ms |
| Served TF-IDF (model only) | scikit-learn | 16,622 + aug. | 0.8712 | **0.8782** | 0.8717 | 37.5% | 0.8 ms |
| Served TF-IDF + ITIL cues (production on Render) | scikit-learn | 16,622 + aug. | 0.8549 | **0.8635** | 0.8565 | **79.2%** | 0.8 ms |

Latency is single-ticket (batch size 1) on a shared 4-vCPU GitHub runner, so treat it as relative. The ITIL-cue row
reuses the model-only latency measurement (the cue layer was not timed separately).

**Per-class precision / recall** (P / R):

| Model | Change | Incident | Problem | Request |
| :--- | :---: | :---: | :---: | :---: |
| TF-IDF + LR | 0.95 / 0.95 | 0.80 / 0.78 | 0.60 / 0.62 | 0.99 / 0.99 |
| Keras CNN | 0.95 / 0.93 | 0.81 / 0.65 | 0.51 / 0.71 | 0.98 / 0.99 |
| DistilBERT + LoRA | 0.94 / 0.96 | 0.71 / 0.85 | 0.54 / 0.34 | 0.98 / 0.98 |
| Served TF-IDF + ITIL cues | 0.95 / 0.95 | 0.86 / 0.79 | 0.65 / 0.74 | 0.98 / 0.99 |

### Error analysis (precision / recall)

- **Incident ↔ Problem accounts for 92-95% of all errors** for every model (e.g. TF-IDF: 303 Incident→Problem and
  279 Problem→Incident of 618 errors). Both describe something broken; ITIL separates them by recurrence /
  root-cause intent, which the synthetic labels apply inconsistently. Request and Change are near-perfect.
- **Confidence gate:** at the 0.80 threshold, auto-accepted tickets are far more precise than overall accuracy
  (TF-IDF 96.0% precision at 53% coverage; DistilBERT 94.4% at 55%; served model alone 96.4% at 65%; served model with
  ITIL cues, as deployed, 94.8% at 66%), so the gate routes the ambiguous Incident/Problem tickets to human review.
  See `reports/figures/precision_coverage.png`.
- **Length parity (bias check):** Macro-F1 is stable for tickets under 100 words (0.85 / 0.84 for TF-IDF,
  0.78 / 0.78 for DistilBERT) and drops on the rarer 100-199-word tickets for the bag-of-words model (0.76),
  while DistilBERT stays flat (0.77), see `error_analysis.md`.
- **Real-world generalisation:** on the 24-ticket hand-written holdout (`data/fixtures/handwritten_holdout.json`,
  never used for tuning), **DistilBERT + LoRA reaches 62.5% vs. 45.8% (Keras) and 33.3% (TF-IDF)** trained on the same
  data: the transformer transfers best beyond the long-form synthetic style, while the bag-of-words models win
  on the in-distribution test split. The production service closes that gap with the ITIL cue layer (79.2%).
  The holdout is small, so treat these as indicative.

### Training details

| | TF-IDF + LR | Keras CNN | DistilBERT + LoRA |
| :--- | :--- | :--- | :--- |
| Features / backbone | word 1-2-grams, 80k max | 30k vocab, 256 tokens, 128-d embeddings | `distilbert-base-uncased`, 128 tokens |
| Trainable parameters | linear | 3.9 M | 0.74 M of 67.7 M (1.09%) |
| Optimisation | L-BFGS, C=4, balanced | Adam 1e-3, balanced class weights, early stopping (8 epochs) | AdamW 3e-4, linear warm-up, 3 epochs, best epoch kept (2) |
| Validation Macro-F1 | 0.8103 | 0.7894 | 0.7922 |
| Training time (4 vCPU) | 2 s | 73 s | 59 min |

*Ablation (earlier run; its metrics are not in the committed reports):* LoRA on all six DistilBERT linear layers (r=16, 2.79% trainable) with a class-balanced loss
overfit after epoch 1 (validation Macro-F1 0.789 → 0.769), so the lighter q/v adapter is used.

### Serving model on Render (free plan)

The Render deployment runs the served TF-IDF model plus the ITIL cue correction from
[`itil_rules.py`](src/ticketlens/itil_rules.py) (the DistilBERT container needs ~430 MB at idle, too close to the 512 MB free-plan limit under load; it is served by
[`Dockerfile.transformer`](Dockerfile.transformer) instead). The cues trade ~1.5 Macro-F1 points on the
synthetic split for a large gain on the hand-written holdout (37.5% → 79.2%). Set `inference.itil_cue_weight: 0`
to disable them. The hand-written sets are small, so treat those numbers as indicative.

---

## FastAPI Inference Service

### 1. Health Endpoint (`GET /health`)

Example output from the DistilBERT image (`Dockerfile.transformer`) or a local setup with both models trained. The
live Render demo and the lightweight `Dockerfile` image load only `"baseline"` (the TF-IDF model).

```bash
curl -X GET http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "ready": true,
  "models_loaded": ["baseline", "transformer"]
}
```

### 2. Metadata Endpoint (`GET /metadata`)

`default_model` is `transformer` in [`configs/config.yaml`](configs/config.yaml), but the API reports and uses the
TF-IDF `baseline` whenever no transformer artifacts are loaded (as on the live demo, which returns
`"models_available": ["baseline"], "default_model": "baseline"`).

```bash
curl -X GET http://localhost:8000/metadata
```
```json
{
  "project_name": "TicketLens",
  "version": "0.1.0",
  "dataset_name": "Tobi-Bueck/customer-support-tickets",
  "target_categories": ["Incident", "Request", "Problem", "Change"],
  "device": "cpu",
  "models_available": ["baseline", "transformer"],
  "default_model": "transformer",
  "confidence_threshold": 0.8,
  "auth_required": false
}
```

### 3. Prediction Endpoint (`POST /predict`)

Responses also include `itil_cues` (ITIL keyword signals that adjusted the probabilities),
`warnings` (e.g. non-English input; any warning forces `"decision": "Needs Review"`), `routing` (the resolver
queue and response target from [`configs/routing.yaml`](configs/routing.yaml): auto-accepted tickets go to their
category's queue, everything else to the human triage queue with the model's suggestion) and `processed_text`
(the exact text the model classified, with PII replaced by `[EMAIL]`, `[PHONE]`, `[IP]`).

**Why classify at all?** People who raise tickets write free text and don't pick (or mis-pick) ITIL categories,
but each category goes to a different team with a different deadline: Incidents to on-call support within hours,
Requests to fulfilment, Problems to root-cause engineers, Changes to the Change Advisory Board. TicketLens
automates that first triage step and only auto-routes when it is confident; everything else goes to a human.

The responses below are illustrative (abridged, without `routing` / `processed_text`); exact probabilities depend on
the trained model.

#### Request (Incident Example)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Production Kubernetes worker node memory spiked to 98% and pods were evicted.", "model_type": "baseline"}'
```

#### Response
```json
{
  "predicted_category": "Incident",
  "confidence": 0.6384,
  "decision": "Needs Review",
  "probabilities": {
    "Change": 0.0821,
    "Incident": 0.6384,
    "Problem": 0.1642,
    "Request": 0.1153
  },
  "model_type": "TF-IDF + Logistic Regression",
  "model_version": "0.1.0",
  "confidence_threshold": 0.8,
  "itil_cues": {},
  "warnings": []
}
```

#### Request (Request Example - High Confidence)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Please provide read-only database credentials and analytical query access for our new analyst.", "model_type": "baseline"}'
```

#### Response (Auto Accept)
```json
{
  "predicted_category": "Request",
  "confidence": 0.8912,
  "decision": "Auto Accept",
  "probabilities": {
    "Change": 0.0211,
    "Incident": 0.0345,
    "Problem": 0.0532,
    "Request": 0.8912
  },
  "model_type": "TF-IDF + Logistic Regression",
  "model_version": "0.1.0",
  "confidence_threshold": 0.8,
  "itil_cues": {"Request": ["access request"]},
  "warnings": []
}
```

### 4. Optional API Key Authentication

By default `/predict` is open (public demo). Set the `TICKETLENS_API_KEY` environment variable to
require an `X-API-Key` header; requests without the correct key get `401`. `/health`, `/metadata`
and the dashboard stay public (Render health checks need `/health`), and the dashboard prompts for the
key once and stores it in the browser.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -H "X-API-Key: $TICKETLENS_API_KEY" \
  -d '{"text": "VPN keeps disconnecting since this morning"}'
```

### 5. Non-English Input

Models are trained on English tickets only. Non-English text (non-Latin scripts, or German, Spanish,
French, Portuguese, Italian and Dutch function words) is still classified but returns a warning and is
always routed to `Needs Review`.

---

## Explainability (SHAP) & Bias Auditing

TicketLens features automated feature attribution and dataset-level auditing generated via `python -m ticketlens.explain`:

### Top Driving N-Grams (TF-IDF Coefficients & SHAP Attributions)

From [`reports/top_features_per_class.json`](reports/top_features_per_class.json) (Logistic Regression coefficients of the
served model; per-class SHAP `LinearExplainer` mean |values| are in [`reports/shap_summary.json`](reports/shap_summary.json)):
- **`Change`:** `update`, `enhance`, `improve`, `revise`, `boost`, `enhancement`.
- **`Incident`:** `problem`, `incident`, `breach`, `issue`, `has`, `issues`.
- **`Problem`:** `problem`, `facing`, `issue`, `issues`, `problem with`, `despite`.
- **`Request`:** `for`, `details`, `strategies`, `integrating`, `guidance`, `information`.

The overlap of `problem` / `issue` between Incident and Problem explains the dominant confusion pair in the
error analysis.

### Bias Check

[`reports/bias_and_fairness_audit.json`](reports/bias_and_fairness_audit.json) and the error analysis cover:
class balance (Incident 39.7%, Request 29.0%, Problem 20.8%, Change 10.5%; imbalance 3.77:1, handled with
class-balanced training), ticket-length profiles per class, performance parity by ticket length, and a PII scan
of the raw text before redaction.

### Responsible AI & Privacy Considerations
1. **PII Sanitization:** Real IT tickets often contain emails, callback numbers, and internal IP addresses. TicketLens applies regex redaction before training and inference, replacing email strings with `[EMAIL]`, phone numbers with `[PHONE]`, and IPv4 addresses with `[IP]`. Names, addresses, account numbers and other identifiers are **not** redacted.
2. **Confidence-Gated Routing:** The routing threshold is set to `0.80`. Any ticket with classification confidence below 80% is marked `"Needs Review"` and flagged for human-in-the-loop triage, mitigating the risk of silent misrouting on high-impact incidents.
3. **Audit Scope:** A balanced class distribution does not guarantee absence of subtle organizational or linguistic bias across diverse technical departments. Periodic retraining and model drift monitoring are recommended.

---

## Docker Deployment

Two images are provided; both download the dataset during `docker build` (internet access to Hugging
Face is required at build time).

### Serving the fine-tuned DistilBERT + LoRA model

[`Dockerfile.transformer`](Dockerfile.transformer) serves the fine-tuned DistilBERT through the same FastAPI
app. It installs CPU-only PyTorch, downloads the LoRA adapter published by the
[Benchmark workflow](.github/workflows/benchmark.yml) (GitHub release `models`), bakes in the DistilBERT
base weights so the container runs offline, and defaults `/predict` to the transformer (~430 MB RAM, measured in CI).

```bash
docker build -f Dockerfile.transformer -t ticketlens:transformer .
docker run -p 8000:8000 --rm ticketlens:transformer
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "VPN is down for the whole office since this morning"}'   # model_type: DistilBERT (PEFT / LoRA)
```

### Lightweight TF-IDF image (same as Render)

[`Dockerfile`](Dockerfile) mirrors the Render setup: lightweight serving stack (no PyTorch) and the
TF-IDF model trained during the build; runs in ~240 MB.

```bash
# 1. Build the Docker container image (downloads the dataset and trains the serving model)
docker build -t ticketlens:latest .

# 2. Run the container on port 8000
docker run -p 8000:8000 --rm ticketlens:latest

# 3. Test container health
curl -f http://localhost:8000/health
```

---

## Deploying to Render Cloud

TicketLens is configured for deployment to **[Render](https://render.com/)** (free plan).

### Method 1: 1-Click Infrastructure Blueprint (Recommended)

The repository includes a ready-to-use [`render.yaml`](render.yaml) specification file.

1. Push your repository to GitHub: `https://github.com/Charanloyal/ticketlens`.
2. Log in to [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** → **Blueprint**.
4. Connect your `ticketlens` repository.
5. Render will automatically detect `render.yaml` and configure:
   - **Service Name:** `ticketlens`
   - **Environment:** `Python 3.11+`
   - **Build Command:** `pip install --upgrade pip && pip install -r requirements-render.txt && PYTHONPATH=src python -m ticketlens.build`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips='*'`
   - **Health Check Path:** `/health`
6. Click **Apply**. Render will deploy the application with live logging.
7. *(Optional)* To protect `/predict`, add `TICKETLENS_API_KEY` under the service's **Environment** tab.

**Deploy safety nets**
- `python -m ticketlens.build` runs dataset download, corpus preparation and training as separate processes,
  so only one step's memory is resident at a time (CI fails if any step exceeds 512 MB).
- The build **fails** (`--require-dataset`) if the Hugging Face dataset can't be downloaded, instead of
  silently training on the 8-ticket test fixture. A failed build leaves the previous deploy live.
- `/health` returns **503** when no model is loaded, so Render never routes traffic to a broken deploy.
- With `TICKETLENS_ENV=production` the app never trains at startup (which would exceed 512 MB).
- The [CI workflow](.github/workflows/ci.yml) replays the exact Render build on Python 3.11, runs the tests,
  boots the server, smoke-tests `/health` and `/predict`, and fails if the server uses more than 400 MB.

**Free plan behaviour:** free web services sleep after ~15 minutes without traffic. The first visit then
can return `503` or take a minute or more while Render wakes the service; the dashboard shows a "Waking up server…"
status and a hint during that time. Subsequent requests are fast.

> **Why a separate `requirements-render.txt`?** Render's free plan has a 512 MB RAM limit. The full
> `requirements.txt` pulls in PyTorch, Transformers, SHAP and plotting libraries, which alone exceed that
> limit. The Render build installs only the lightweight serving stack and trains the TF-IDF + Logistic
> Regression model during the build (`ticketlens.build` runs download, corpus preparation and training as separate processes to keep
> each step's peak memory low; ~1 minute total), so the server binds `$PORT` within seconds
> and runs in ~240 MB.
> Requests for the `transformer` model automatically fall back to the baseline when no fine-tuned
> transformer artifacts are present.

### Method 2: Manual Web Service Setup

If you configure the Web Service manually via Render UI:

1. In Render Dashboard, click **New +** → **Web Service**.
2. Connect your GitHub repository `https://github.com/Charanloyal/ticketlens`.
3. Fill in the following settings:
   - **Name:** `ticketlens`
   - **Language / Runtime:** `Python`
   - **Region:** Any (e.g. `Oregon (US West)` or `Frankfurt (EU)`)
   - **Branch:** `main`
   - **Build Command:**
     ```bash
     pip install --upgrade pip && pip install -r requirements-render.txt && PYTHONPATH=src python -m ticketlens.build
     ```
   - **Start Command:**
     ```bash
     uvicorn main:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips='*'
     ```
   - **Plan:** `Free`
4. Add the following **Environment Variables**:
   - `PYTHON_VERSION`: `3.11.9`
   - `TICKETLENS_ENV`: `production`
   - `PYTHONPATH`: `src:.`
   - `TICKETLENS_CONFIDENCE_THRESHOLD`: `0.80`
   - `TICKETLENS_API_KEY`: *(optional)* any secret value to require an `X-API-Key` header on `/predict`
5. Set **Health Check Path** under *Advanced*: `/health`.
6. Click **Create Web Service**. Your live public application URL will be provisioned at `https://<service-name>.onrender.com` (this project runs at [ticketlens.onrender.com](https://ticketlens.onrender.com/)).

---

## Troubleshooting & FAQ

- **Issue:** `ModuleNotFoundError: No module named 'ticketlens'`
  - **Resolution:** Run `pip install -e .` inside the active virtual environment to install the project package in editable mode.
- **Issue:** Render deploy fails with `Out of memory (used over 512Mi)` / `No open ports detected`
  - **Resolution:** Make sure the service uses `requirements-render.txt` (not `requirements.txt`) and the build command from `render.yaml`. Installing PyTorch/Transformers or training at startup exceeds the free-plan memory limit.
- **Issue:** Render build fails with `Refusing to fall back to the tiny synthetic fixture`
  - **Resolution:** The Hugging Face dataset download failed (network issue or rate limit). The previous deploy stays live; click **Manual Deploy → Deploy latest commit** to retry.
- **Issue:** Render deploy fails its health check / `/health` returns `503`
  - **Resolution:** No model artifacts were found at startup. Check the build log for the `Baseline validation: Macro-F1=...` line; the build command must include the `python -m ticketlens.build` step.
- **Issue:** The live site takes a minute or more (or returns `503`) on the first request
  - **Resolution:** Expected on Render's free plan (the service sleeps when idle). Upgrade to a paid instance for always-on serving.
- **Issue:** `CUDA not detected / Using CPU`
  - **Resolution:** `TicketLens` automatically detects whether CUDA is available and selects `cuda` when present, gracefully falling back to `cpu` without configuration changes.
- **Issue:** `Hugging Face download timeout or offline environment`
  - **Resolution:** For local development the system falls back to `data/fixtures/synthetic_tickets.json` when the Hugging Face Hub is unreachable. Production builds (`python -m ticketlens.build`) fail instead.

---

## Limitations

- **Synthetic data.** All training and test tickets are synthetic customer-support emails. The 0.83–0.88 Macro-F1 scores
  are in-distribution numbers; on short, hand-written tickets the models score much lower (33–62% accuracy on the
  24-ticket holdout without the ITIL cue layer).
- **Small hand-written sets.** The 48 development and 24 holdout tickets were written and labelled by the author, who
  also designed the ITIL cue rules on the development set. The 79.2% holdout figure is indicative only.
- **Not like-for-like.** The served TF-IDF model (Macro-F1 0.8782 / 0.8635 with cues) is trained on 2.4x more data than
  the three benchmarked architectures; it should not be compared directly with the 7,000-ticket rows.
- **DistilBERT is not on the live demo.** It is only served by `Dockerfile.transformer`; the Render free plan runs TF-IDF.
- **PII redaction is regex-based** and covers emails, phone numbers and IPv4 addresses only.
- **Latency** numbers are from shared GitHub runners (batch size 1) and are relative, not a load test;
  `reports/benchmark_latency.json` is from a separate local run whose hardware is not recorded.
- **Licence.** The dataset is CC BY-NC 4.0, so models trained on it should not be used commercially.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE). The underlying dataset (`Tobi-Bueck/customer-support-tickets`) is licensed under [Creative Commons Attribution Non-Commercial 4.0 (`CC-BY-NC-4.0`)](https://creativecommons.org/licenses/by-nc/4.0/).
