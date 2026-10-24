"""End-to-end pipeline runner for TicketLens."""

import argparse
import logging
import sys
import time
from ticketlens.config import load_config
from ticketlens.data import get_dataset_splits
from ticketlens.evaluate import evaluate_models
from ticketlens.explain import run_explainability_pipeline
from ticketlens.train_baseline import train_baseline_model
from ticketlens.train_transformer import train_transformer_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ticketlens.pipeline")


def main():
    parser = argparse.ArgumentParser(description="Run complete TicketLens ML pipeline")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick", help="Execution mode ('quick' for functional verification, 'full' for comprehensive training)")
    parser.add_argument("--fixture", action="store_true", help="Force synthetic test fixture usage")
    parser.add_argument("--skip-transformer", action="store_true", help="Skip transformer training")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("==================================================")
    logger.info("   Starting TicketLens Pipeline (Mode: %s)", args.mode)
    logger.info("==================================================")

    cfg = load_config()

    # Step 1: Data preparation
    logger.info("[Step 1/5] Preparing and splitting dataset...")
    train_df, val_df, test_df, label2id = get_dataset_splits(cfg, mode=args.mode, use_fixture=args.fixture)
    logger.info("Data splits ready: Train=%d, Val=%d, Test=%d", len(train_df), len(val_df), len(test_df))

    # Step 2: Baseline training
    logger.info("[Step 2/5] Training TF-IDF + Logistic Regression baseline...")
    baseline_metrics = train_baseline_model(cfg, mode=args.mode, use_fixture=args.fixture)
    logger.info("Baseline validation Macro-F1: %.4f", baseline_metrics["validation_macro_f1"])

    # Step 3: Transformer training
    if not args.skip_transformer:
        logger.info("[Step 3/5] Training DistilBERT (LoRA) sequence classifier...")
        transformer_metrics = train_transformer_model(cfg, mode=args.mode, use_fixture=args.fixture)
        logger.info("Transformer validation Macro-F1: %.4f", transformer_metrics["validation_macro_f1"])
    else:
        logger.info("[Step 3/5] Skipped transformer training (--skip-transformer set).")

    # Step 4: Held-out test set evaluation
    logger.info("[Step 4/5] Evaluating models on identical test set...")
    eval_results = evaluate_models(cfg, use_fixture=args.fixture)

    # Step 5: Explainability & Bias Audit
    logger.info("[Step 5/5] Running explainability (SHAP) and bias audit...")
    explain_results = run_explainability_pipeline(cfg, use_fixture=args.fixture)

    total_duration = round(time.time() - start_time, 2)
    logger.info("==================================================")
    logger.info("   TicketLens Pipeline Completed in %.2f seconds", total_duration)
    logger.info("   Reports and charts saved under: reports/")
    logger.info("==================================================")


if __name__ == "__main__":
    main()
