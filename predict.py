"""
Run NER inference with a fine-tuned DistilRoBERTa model.

Usage:
    python predict.py --model_dir ./outputs --text "Apple was founded by Steve Jobs in California."
    python predict.py --model_dir ./outputs --file sentences.txt
"""

import argparse
import json
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def predict(
    text: str,
    tokenizer,
    model,
    device: torch.device,
) -> List[Dict]:
    """
    Returns a list of entity spans:
        [{"word": str, "entity": str, "start": int, "end": int}, ...]
    """
    words = text.split()
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits[0]                    # (seq_len, num_labels)
    predictions = torch.argmax(logits, dim=-1)    # (seq_len,)
    id2label: Dict[int, str] = model.config.id2label

    word_ids = encoding["input_ids"][0]           # reuse word_id mapping
    word_ids_list = tokenizer(
        words, is_split_into_words=True, truncation=True, max_length=512
    ).word_ids()

    # Aggregate: only keep first subword of each word
    results = []
    seen_word_ids = set()
    for idx, word_id in enumerate(word_ids_list):
        if word_id is None or word_id in seen_word_ids:
            continue
        seen_word_ids.add(word_id)
        label = id2label[predictions[idx].item()]
        if label != "O":
            results.append({
                "word": words[word_id],
                "entity": label,
                "word_index": word_id,
            })

    return _merge_bio_spans(results, words)


def _merge_bio_spans(token_preds: List[Dict], words: List[str]) -> List[Dict]:
    """
    Merge consecutive B-/I- tags into full entity spans.
    e.g. [B-PER, I-PER] for ["Steve", "Jobs"] → {"text": "Steve Jobs", "type": "PER"}
    """
    entities, current = [], None
    for pred in token_preds:
        tag = pred["entity"]
        prefix, etype = tag.split("-", 1) if "-" in tag else (tag, "")
        if prefix == "B":
            if current:
                entities.append(current)
            current = {"text": pred["word"], "type": etype, "start": pred["word_index"]}
        elif prefix == "I" and current and current["type"] == etype:
            current["text"] += " " + pred["word"]
        else:
            if current:
                entities.append(current)
            current = None
    if current:
        entities.append(current)
    return entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model directory")
    parser.add_argument("--text", help="Single sentence to annotate")
    parser.add_argument("--file", help="Path to a text file (one sentence per line)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(args.model_dir)
    model.to(device)

    sentences = []
    if args.text:
        sentences.append(args.text)
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Provide --text or --file")

    for sentence in sentences:
        entities = predict(sentence, tokenizer, model, device)
        print(f"\nInput : {sentence}")
        if entities:
            for ent in entities:
                print(f"  [{ent['type']}] {ent['text']}")
        else:
            print("  (no entities found)")


if __name__ == "__main__":
    main()
