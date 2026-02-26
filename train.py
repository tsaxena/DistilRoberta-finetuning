"""
Fine-tune DistilRoBERTa for token-level Named Entity Recognition (NER).

Usage:
    python train.py                        # uses defaults in config.py
    python train.py --train_file data/train.json --val_file data/val.json
    python train.py --epochs 10 --lr 3e-5
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import NERConfig
from dataset import NERDataset

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_compute_metrics(id2label: Dict[int, str], ignore_index: int = -100):
    """
    Returns a compute_metrics function that uses seqeval for span-level
    precision, recall and F1.
    """
    from seqeval.metrics import f1_score, precision_score, recall_score

    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)  # (batch, seq_len)

        true_seqs, pred_seqs = [], []
        for pred_row, label_row in zip(predictions, labels):
            true_seq, pred_seq = [], []
            for p, l in zip(pred_row, label_row):
                if l == ignore_index:
                    continue
                true_seq.append(id2label[l])
                pred_seq.append(id2label[p])
            true_seqs.append(true_seq)
            pred_seqs.append(pred_seq)

        return {
            "precision": precision_score(true_seqs, pred_seqs),
            "recall":    recall_score(true_seqs, pred_seqs),
            "f1":        f1_score(true_seqs, pred_seqs),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(cfg: NERConfig) -> NERConfig:
    parser = argparse.ArgumentParser(description="Fine-tune DistilRoBERTa for NER")
    parser.add_argument("--model_name",  default=cfg.model_name)
    parser.add_argument("--output_dir",  default=cfg.output_dir)
    parser.add_argument("--train_file",  default=cfg.train_file)
    parser.add_argument("--val_file",    default=cfg.val_file)
    parser.add_argument("--test_file",   default=cfg.test_file)
    parser.add_argument("--data_format", default=cfg.data_format, choices=["json", "conll"])
    parser.add_argument("--max_length",  type=int,   default=cfg.max_length)
    parser.add_argument("--batch_size",  type=int,   default=cfg.batch_size)
    parser.add_argument("--epochs",      type=int,   dest="num_epochs", default=cfg.num_epochs)
    parser.add_argument("--lr",          type=float, dest="learning_rate", default=cfg.learning_rate)
    parser.add_argument("--fp16",        action="store_true", default=cfg.fp16)
    parser.add_argument("--seed",        type=int,   default=cfg.seed)
    args = parser.parse_args()

    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = parse_args(NERConfig())
    set_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- Label maps ---
    label_list: List[str] = cfg.label_list
    label2id: Dict[str, int] = {l: i for i, l in enumerate(label_list)}
    id2label: Dict[int, str] = {i: l for i, l in enumerate(label_list)}
    num_labels = len(label_list)
    logger.info(f"Labels ({num_labels}): {label_list}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, add_prefix_space=True)

    # --- Datasets ---
    logger.info(f"Loading train data from {cfg.train_file}")
    train_dataset = NERDataset(
        cfg.train_file, tokenizer, label2id,
        max_length=cfg.max_length,
        data_format=cfg.data_format,
        ignore_index=cfg.ignore_index,
    )

    logger.info(f"Loading val data from {cfg.val_file}")
    val_dataset = NERDataset(
        cfg.val_file, tokenizer, label2id,
        max_length=cfg.max_length,
        data_format=cfg.data_format,
        ignore_index=cfg.ignore_index,
    )

    test_dataset = None
    if cfg.test_file and Path(cfg.test_file).exists():
        logger.info(f"Loading test data from {cfg.test_file}")
        test_dataset = NERDataset(
            cfg.test_file, tokenizer, label2id,
            max_length=cfg.max_length,
            data_format=cfg.data_format,
            ignore_index=cfg.ignore_index,
        )

    # --- Model ---
    logger.info(f"Loading model: {cfg.model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,   # new classification head vs. pretrained
    )

    # --- Training arguments ---
    total_steps = (len(train_dataset) // cfg.batch_size) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        fp16=cfg.fp16,
        dataloader_num_workers=cfg.dataloader_num_workers,
        seed=cfg.seed,
        logging_dir=os.path.join(cfg.output_dir, "logs"),
        logging_steps=10,
        report_to="none",   # swap to "wandb" or "tensorboard" as needed
    )

    # --- Data collator (handles dynamic padding within a batch) ---
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=cfg.ignore_index,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(id2label, cfg.ignore_index),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # --- Train ---
    logger.info("Starting training...")
    train_result = trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    logger.info(f"Training complete. Best model saved to: {cfg.output_dir}")

    # --- Evaluate on validation set ---
    logger.info("Evaluating on validation set...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    logger.info(f"Val metrics: {eval_metrics}")

    # --- Evaluate on test set ---
    if test_dataset is not None:
        logger.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)
        logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
