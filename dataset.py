"""
Dataset utilities for NER fine-tuning.

Handles two input formats:
  - JSON: list of {"tokens": [...], "ner_tags": [...]} dicts
  - CoNLL: word + label per line, blank lines between sentences

The core challenge with token classification on subword models:
  - A single word may be split into multiple subword tokens.
  - Labels exist only at the word level.
  - We assign the word's label to the *first* subword and -100 (ignored) to the rest,
    so the loss is only computed on the first subword of each word.
"""

import json
from typing import Dict, List

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


class NERDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizerFast,
        label2id: Dict[str, int],
        max_length: int = 128,
        data_format: str = "json",
        ignore_index: int = -100,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.ignore_index = ignore_index

        if data_format == "json":
            raw_data = self._load_json(file_path)
        elif data_format == "conll":
            raw_data = self._load_conll(file_path)
        else:
            raise ValueError(f"Unsupported data_format: {data_format!r}. Use 'json' or 'conll'.")

        self.encodings = self._tokenize_and_align(raw_data)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_json(self, path: str) -> List[Dict]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._validate(data, path)
        return data

    def _load_conll(self, path: str) -> List[Dict]:
        """
        Expects CoNLL format:
            word  label
            word  label
            <blank line>
            ...
        Lines starting with '-DOCSTART-' are skipped.
        """
        sentences, tokens, tags = [], [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("-DOCSTART-"):
                    continue
                if line == "":
                    if tokens:
                        sentences.append({"tokens": tokens, "ner_tags": tags})
                        tokens, tags = [], []
                else:
                    parts = line.split()
                    tokens.append(parts[0])
                    tags.append(parts[-1])   # label is always the last column
        if tokens:
            sentences.append({"tokens": tokens, "ner_tags": tags})
        return sentences

    # ------------------------------------------------------------------
    # Tokenisation + label alignment
    # ------------------------------------------------------------------

    def _tokenize_and_align(self, data: List[Dict]) -> List[Dict]:
        """
        Tokenises word-level data and aligns BIO labels to subword tokens.

        Strategy:
          - First subword of a word  → word's label id
          - Subsequent subwords      → ignore_index (-100)
          - Special tokens [CLS]/[SEP]/[PAD] → ignore_index
        """
        results = []
        for sample in data:
            words: List[str] = sample["tokens"]
            word_labels: List[str] = sample["ner_tags"]

            encoding = self.tokenizer(
                words,
                is_split_into_words=True,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=False,
            )

            word_ids = encoding.word_ids()  # None for special tokens
            aligned_labels = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    # Special token
                    aligned_labels.append(self.ignore_index)
                elif word_id != prev_word_id:
                    # First subword of this word
                    label_str = word_labels[word_id]
                    aligned_labels.append(
                        self.label2id.get(label_str, self.label2id["O"])
                    )
                else:
                    # Continuation subword — ignore in loss
                    aligned_labels.append(self.ignore_index)
                prev_word_id = word_id

            results.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": aligned_labels,
            })
        return results

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, data: List[Dict], path: str) -> None:
        for i, item in enumerate(data):
            if len(item["tokens"]) != len(item["ner_tags"]):
                raise ValueError(
                    f"{path} sample {i}: tokens/ner_tags length mismatch "
                    f"({len(item['tokens'])} vs {len(item['ner_tags'])})"
                )

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict:
        import torch
        item = self.encodings[idx]
        return {k: torch.tensor(v, dtype=torch.long) for k, v in item.items()}
