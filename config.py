"""
Configuration for DistilRoBERTa NER fine-tuning.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NERConfig:
    # Model
    model_name: str = "distilroberta-base"
    output_dir: str = "./outputs"

    # Data
    train_file: str = "data/train.json"
    val_file: str = "data/val.json"
    test_file: Optional[str] = "data/test.json"
    # Data format: "json" or "conll"
    data_format: str = "json"

    # NER labels (BIO scheme). Modify to match your task.
    label_list: List[str] = field(default_factory=lambda: [
        "O",
        "B-PER", "I-PER",
        "B-ORG", "I-ORG",
        "B-LOC", "I-LOC",
        "B-MISC", "I-MISC",
    ])

    # Training hyperparameters
    max_length: int = 128
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Evaluation
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True

    # Misc
    seed: int = 42
    fp16: bool = False          # Set True if GPU with fp16 support
    dataloader_num_workers: int = 0

    # Label assigned to ignored subword tokens (must be -100 for CrossEntropy)
    ignore_index: int = -100
