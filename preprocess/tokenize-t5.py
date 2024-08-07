import os

from transformers import AutoTokenizer

from datasets import Dataset
import pandas 

import ray
ray.init()

os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
os.environ["MODIN_MEMORY"] = "10000000000"

import modin.pandas as pd 
import modin.config as cfg; print(cfg.CpuCount.get()); cfg.CpuCount.put(12); print(cfg.CpuCount.get())

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, default="data/combined-train.jsonl")
    parser.add_argument("--validation_dataset", type=str, default="data/combined-validation.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/tokenized/")
    
    return parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
PROMPT = """Can the target text be inferred from the source text? In other words, is the target text factually consistent with the source text? Answer with a "Yes" or "No"."""

def filter_long_example(source, target, max_input_length=2048):
    text = f"{PROMPT}\n\n_target_ {target}\n\n_source_ {source}"
    inputs = tokenizer(text, truncation=False, max_length=max_input_length)
    
    if len(inputs["input_ids"]) + 2 > max_input_length:
        return True
    return False


def filter_long_examples(row):
    source = row["source"]
    target = row["target"]
    
    row["remove"] = filter_long_example(source, target)
    return row


def tokenize_example(example, tokenizer, max_input_length=2048, max_target_length=2, training=True):
    source = example['source']
    target = example['target']
    label = example['label']
    
    input_text = f"{PROMPT}\n\n_target_ {target}\n\n_source_ {source}"
    output_text = f"{label}"

    if training:
        model_inputs = tokenizer(input_text, max_length=max_input_length, truncation=True)
        labels = tokenizer(output_text, max_length=max_target_length, truncation=True).input_ids
    else:
        # We apply padding in the validation set. During evaluation, we do not use a collator for dynamic padding.
        model_inputs = tokenizer(input_text, max_length=max_input_length, padding="max_length", truncation=True)
        labels = tokenizer(output_text, max_length=max_target_length, padding="max_length", truncation=True).input_ids

    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [labels_example if labels_example != 0 else -100]
        labels_with_ignore_index.extend(labels_example)
    
    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs


def tokenize_save_train(ds_path, output_dir):
    print(f"Dataset: {ds_path}")
    print("Loading dataset into pandas...")
    data = pandas.read_json(ds_path, lines=True)
    data = data.sample(frac=1, random_state=42)
    data = data.reset_index(drop=True)
    
    print(f"Dataset size: {len(data)}")
    
    print("Converting dataset to Modin...")
    data = pd.DataFrame(data)
    
    print("Filtering long examples...")
    data = data.apply(filter_long_examples, axis=1)
    
    data = data[data["remove"] == False][["source", "target", "label"]]
    print(f"Dataset size: {len(data)}")
    
    print("Converting dataset back to pandas...")
    data = pandas.DataFrame(data, columns=["source", "target", "label"])
    
    print("Converting dataset to HuggingFace Datasets...")
    dataset = Dataset.from_pandas(data)
    
    print("Tokenizing dataset...")
    dataset = dataset.map(lambda examples: tokenize_example(examples, tokenizer),
                            remove_columns=["source", "target", "label"], num_proc=12)
    
    dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    
    print("Saving dataset to disk...")
    dataset.save_to_disk(output_dir + ds_path.split("/")[-1].replace(".jsonl", f"-tokenized"))
    
    print("Done!")
    
def tokenize_save_val(ds_path, output_dir):
    print(f"Dataset: {ds_path}")
    print("Loading dataset into pandas...")
    data = pandas.read_json(ds_path, lines=True)
    data = data.sample(frac=1, random_state=42)
    data = data.reset_index(drop=True)
    
    print(f"Dataset size: {len(data)}")
    
    print("Converting dataset to Modin...")
    data = pd.DataFrame(data)
    
    print("Filtering long examples...")
    data = data.apply(filter_long_examples, axis=1)
    
    subsets = set(data["subset"].to_list())
    print(f"Subsets: {subsets}")
    for subset in subsets:
        print(f"Subset: {subset}")
        subset_data = data[data["subset"] == subset][["source", "target", "label"]]
        print(f"Subset size: {len(subset_data)}")
        
        print("Converting dataset back to pandas...")
        subset_data = pandas.DataFrame(subset_data, columns=["source", "target", "label"])
        
        print("Converting dataset to HuggingFace Datasets...")
        dataset = Dataset.from_pandas(subset_data)
        
        print("Tokenizing dataset...")
        dataset = dataset.map(lambda examples: tokenize_example(examples, tokenizer, training=False),
                                remove_columns=["source", "target", "label"], num_proc=12)
        
        dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
        
        print("Saving dataset to disk...")
        subset = subset.lower()
        dataset.save_to_disk(output_dir + ds_path.split("/")[-1].replace(".jsonl", f"-{subset}-tokenized"))
    
    print("Done!")
    


if __name__ == "__main__":
    args = parse_args()

    print("Train")
    tokenize_save_train(args.train_dataset, args.output_dir+"/train/")
    print("Validation")
    tokenize_save_val(args.validation_dataset, args.output_dir+"/validation/")
    
    print("Done!")