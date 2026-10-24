"""Fine-tuning DistilBERT for IT support ticket classification using PEFT / LoRA and PyTorch."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from ticketlens.config import get_device, load_config
from ticketlens.data import get_dataset_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ticketlens.train_transformer")


class TicketDataset(Dataset):
    """PyTorch Dataset for tokenized IT support tickets."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def train_transformer_model(
    cfg: Optional[Dict[str, Any]] = None,
    mode: str = "quick",
    use_fixture: bool = False,
) -> Dict[str, Any]:
    """Train DistilBERT sequence classifier with optional LoRA adapter and save checkpoint."""
    if cfg is None:
        cfg = load_config()

    device = get_device()
    logger.info("Using compute device: %s", device)

    # 1. Prepare data
    train_df, val_df, test_df, label2id = get_dataset_splits(cfg, mode=mode, use_fixture=use_fixture)
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)

    t_cfg = cfg["transformer"]
    pretrained_name = t_cfg.get("pretrained_model", "distilbert/distilbert-base-uncased")
    max_length = t_cfg.get("max_length", 128)
    batch_size = t_cfg.get("batch_size", 16)
    eval_batch_size = t_cfg.get("eval_batch_size", 32)
    lr = float(t_cfg.get("learning_rate", 3.0e-5))
    weight_decay = float(t_cfg.get("weight_decay", 0.01))
    epochs = t_cfg.get("epochs_quick", 1) if mode == "quick" else t_cfg.get("epochs_full", 3)
    use_peft = t_cfg.get("use_peft", True)

    logger.info("Loading tokenizer from '%s'...", pretrained_name)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)

    train_dataset = TicketDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length)
    val_dataset = TicketDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)

    # 2. Initialize Model
    logger.info("Initializing DistilBERT sequence classifier with %d classes...", num_labels)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    if use_peft:
        logger.info("Applying LoRA adapter (r=%d, alpha=%d, modules=%s)...",
                    t_cfg.get("lora_r", 8), t_cfg.get("lora_alpha", 16), t_cfg.get("lora_target_modules", ["q_lin", "v_lin"]))
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=t_cfg.get("lora_r", 8),
            lora_alpha=t_cfg.get("lora_alpha", 16),
            lora_dropout=t_cfg.get("lora_dropout", 0.1),
            target_modules=t_cfg.get("lora_target_modules", ["q_lin", "v_lin"]),
            modules_to_save=["classifier", "pre_classifier"],
        )
        model = get_peft_model(base_model, peft_config)
        trainable_params, all_params = model.get_nb_trainable_parameters()
        logger.info(
            "LoRA params: %d trainable / %d total (%.2f%%)",
            trainable_params,
            all_params,
            100 * trainable_params / all_params,
        )
    else:
        model = base_model

    model.to(device)

    # 3. Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    logger.info("Starting training: %d epochs, %d batches per epoch, total steps=%d", epochs, len(train_loader), total_steps)
    start_time = time.time()
    best_val_macro_f1 = -1.0
    best_epoch = 0
    training_history = []

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # Validation loop
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = accuracy_score(all_labels, all_preds)
        val_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        logger.info(
            "Epoch %d/%d - Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f | Val Macro-F1: %.4f",
            epoch,
            epochs,
            avg_train_loss,
            avg_val_loss,
            val_acc,
            val_macro_f1,
        )

        training_history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_acc": round(float(val_acc), 4),
            "val_macro_f1": round(float(val_macro_f1), 4),
        })

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch

    training_duration = round(time.time() - start_time, 3)
    logger.info("Training complete in %.2f seconds. Best Val Macro-F1: %.4f (Epoch %d)", training_duration, best_val_macro_f1, best_epoch)

    # 5. Measure inference latency on 50 samples
    model.eval()
    latency_trials = []
    sample_texts = val_df["text"].iloc[: min(50, len(val_df))].tolist()
    with torch.no_grad():
        for text in sample_texts:
            t0 = time.perf_counter()
            inputs = tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
            _ = model(**inputs)
            latency_trials.append((time.perf_counter() - t0) * 1000)

    p50_latency_ms = round(float(np.percentile(latency_trials, 50)), 3)
    p95_latency_ms = round(float(np.percentile(latency_trials, 95)), 3)

    # 6. Save model checkpoint and metadata
    model_save_dir = Path(cfg["paths"]["model_dir"]) / "transformer"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving transformer checkpoint and tokenizer to %s...", model_save_dir)
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)

    with open(model_save_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    target_names = [id2label[i] for i in range(num_labels)]
    report_dict = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True, zero_division=0)

    metrics = {
        "model_type": "DistilBERT (PEFT / LoRA)" if use_peft else "DistilBERT",
        "pretrained_backbone": pretrained_name,
        "mode": mode,
        "device": str(device),
        "use_peft": use_peft,
        "epochs": epochs,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "classes": target_names,
        "training_duration_seconds": training_duration,
        "validation_accuracy": round(float(val_acc), 4),
        "validation_macro_f1": round(float(val_macro_f1), 4),
        "validation_weighted_f1": round(float(val_weighted_f1), 4),
        "latency_p50_ms": p50_latency_ms,
        "latency_p95_ms": p95_latency_ms,
        "training_history": training_history,
        "classification_report": report_dict,
    }

    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics_transformer.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Transformer validation metrics saved to %s", reports_dir / "metrics_transformer.json")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train transformer model")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick", help="Execution mode")
    parser.add_argument("--fixture", action="store_true", help="Force synthetic fixture usage")
    args = parser.parse_args()

    train_transformer_model(mode=args.mode, use_fixture=args.fixture)
