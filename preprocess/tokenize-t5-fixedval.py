import os

from transformers import AutoTokenizer

from datasets import Dataset
import pandas 

os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
os.environ["MODIN_MEMORY"] = "10000000000"

import modin.pandas as pd 
import modin.config as cfg; print(cfg.CpuCount.get()); cfg.CpuCount.put(8); print(cfg.CpuCount.get())

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, default="data/seahorse-cleaned/seahorse-train.jsonl")
    parser.add_argument("--validation_dataset", type=str, default="data/seahorse-cleaned/seahorse-validation.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/seahorse-cleaned/")
    
    return parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
PROMPT = """Can the target text be inferred from the source text? In other words, is the target text factually consistent with the source text? Answer with a "Yes" or "No"."""

def filter_long_example(source, target, max_input_length=2048):
    text = f"{PROMPT}\n\nSource Text: {source}\n\nTarget Text: {target}\n\nAnswer:"
    inputs = tokenizer(text, truncation=False, max_length=max_input_length)
    
    if len(inputs["input_ids"]) + 2 > max_input_length:
        return True
    return False


def filter_long_examples(row):
    source = row["source"]
    target = row["target"]
    
    row["remove"] = filter_long_example(source, target)
    return row


def tokenize_example(example, tokenizer, max_input_length=2048, max_target_length=2):
    source = example['source']
    target = example['target']
    label = example['label']
    
    input_text = f"{PROMPT}\n\nSource Text: {source}\n\nTarget Text: {target}\n\nAnswer:"
    output_text = f"{label}"

    model_inputs = tokenizer(input_text, max_length=max_input_length, truncation=True)
    labels = tokenizer(output_text, max_length=max_target_length, truncation=True).input_ids

    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [labels_example if labels_example != 0 else -100]
        labels_with_ignore_index.extend(labels_example)
    
    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs


if __name__ == "__main__":
    args = parse_args()

    print("Loading datasets into pandas...")
    train = pandas.read_json(args.train_dataset, lines=True)
    validation = pandas.read_json(args.validation_dataset, lines=True)
    print(f"Train size: {len(train)}, Validation size: {len(validation)}")
    
    print("Converting datasets to Modin...")
    train = pd.DataFrame(train)
    validation = pd.DataFrame(validation)
    
    print("Filtering long examples...")
    train = train.apply(filter_long_examples, axis=1)
    validation = validation.apply(filter_long_examples, axis=1)
    
    train = train[train["remove"] == False][["source", "target", "label"]]
    validation = validation[validation["remove"] == False][["source", "target", "label"]]
    print(f"Train size: {len(train)}, Validation size: {len(validation)}")
    
    print("Converting datasets back to pandas...")
    train = pandas.DataFrame(train, columns=["source", "target", "label"])
    validation = pandas.DataFrame(validation, columns=["source", "target", "label"])
    
    print("Converting datasets to HuggingFace Datasets...")
    train_dataset = Dataset.from_pandas(train)
    validation_dataset = Dataset.from_pandas(validation)
    
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(lambda examples: tokenize_example(examples, tokenizer),
                                      remove_columns=["source", "target", "label"], num_proc=8)
    validation_dataset = validation_dataset.map(lambda examples: tokenize_example(examples, tokenizer),
                                                remove_columns=["source", "target", "label"], num_proc=8)
    
    train_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    validation_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    
    print("Saving datasets to disk...")
    train_dataset.save_to_disk(args.output_dir + args.train_dataset.split("/")[-1].replace(".jsonl", "-tokenized"))
    validation_dataset.save_to_disk(args.output_dir + args.validation_dataset.split("/")[-1].replace(".jsonl", "-tokenized"))
    
    print("Done!")