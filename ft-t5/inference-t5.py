# accelerate launch ft-t5/inference-t5.py --checkpoint_dir checkpoint --ds_path data/test.json 

import argparse
import os

import torch
from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from accelerate import Accelerator

import evaluate
from datasets import Dataset
import pandas 
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='google/flan-t5-small')
    parser.add_argument('--checkpoint_dir', type=str, default='t5-small/')
    parser.add_argument('--ds_path', type=str, default='./data')
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--precision', type=str, default='fp32')
    parser.add_argument('--outfile', type=str, default='default')
    parser.add_argument('--compile', action='store_true')
    return parser.parse_args()

PROMPT = """Can the target text be inferred from the source text? In other words, is the target text factually consistent with the source text? Answer with a "Yes" or "No"."""

def tokenize_example(example, tokenizer, max_input_length=2048, max_target_length=2):
    source = example['source']
    target = example['target']
    label = example['label']
    
    input_text = f"{PROMPT}\n\n_target_ {target}\n\n_source_ {source}"
    output_text = f"{label}"

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

accuracy = evaluate.load('accuracy')
f1 = evaluate.load('f1')
precision = evaluate.load('precision')
recall = evaluate.load('recall')


def calc_metric(preds, labs):
    predictions = []
    for pred in preds:
        if "Yes" in pred:
            predictions.append(1)
        elif "No" in pred:
            predictions.append(0)
        else:
            predictions.append(2)
            
    labels = []
    for label in labs:
        if "Yes" in label:
            labels.append(1)
        elif "No" in label:
            labels.append(0)
        else:
            labels.append(2)
    
    # Calculate accuracy, precision, recall, and f1-score
    accuracy_score = accuracy.compute(references=labels, predictions=predictions)['accuracy']
    precision_score = precision.compute(references=labels, predictions=predictions)['precision']
    recall_score = recall.compute(references=labels, predictions=predictions)['recall']
    try:
        f1_= (precision_score*recall_score*2)/(precision_score+recall_score)
    except:
        f1_ = 0
    return {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_
    }


if __name__ == '__main__':
    args = parse_args()
    
    accelerator = Accelerator(mixed_precision=args.precision)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.checkpoint_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_dir, config=config)
    model.to(accelerator.device) 
    print("Model loaded successfully!")
    
    if args.compile:
        # torch.backends.cuda.enable_math_sdp(False)
        # torch.backends.cuda.enable_mem_efficient_sdp(False)
        # torch.backends.cuda.enable_flash_sdp(True)
        accelerator.print("Compiling model...")
        model = torch.compile(model)
    
    
    print(f"Dataset: {args.ds_path}")
    print("Loading dataset into pandas...")
    data = pandas.read_json(args.ds_path, lines=True)
    data = data.reset_index(drop=True)
    
    
    subsets = set(data["subset"].to_list())
    print(f"Subsets: {subsets}")
    test_datasets = {}
    
    for subset in subsets:
        if not subset:
            continue
        print(f"Subset: {subset}")
        subset_data = data[data["subset"] == subset][["source", "target", "label"]]
        print(f"Subset size: {len(subset_data)}")
        
        print("Converting dataset to HuggingFace Datasets...")
        dataset = Dataset.from_pandas(subset_data)
        
        print("Tokenizing dataset...")
        dataset = dataset.map(lambda examples: tokenize_example(examples, tokenizer),
                                remove_columns=["source", "target", "label"], num_proc=args.num_workers)
        # print keys
        print(dataset.column_names)
        
        dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])

        test_datasets[subset.lower()] = dataset
        
        
    # Create a separate dataloader for each validation dataset
    test_dataloaders = {}
    for test_subset_name, test_dataset in test_datasets.items():
        test_dataloaders[test_subset_name] = accelerator.prepare(DataLoader(test_dataset, shuffle=False, 
                                            batch_size=args.eval_batch_size,
                                            num_workers=args.num_workers))
        
    print("Tokenization and DataLoader creation completed!")
    
    print("Starting evaluation...")
    model.eval()
    all_inputs = []
    all_preds = []
    all_labels = []
    all_subsets = []
    all_probs = []
    
    for test_subset_name, test_dataloader in test_dataloaders.items():
        print(f"Subset: {test_subset_name}")
        subset_inputs = []
        subset_preds = []
        subset_labels = []
        subset_probs = []
        
        for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            
            with torch.no_grad():
                # outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                generated_ids = accelerator.unwrap_model(model).generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask, 
                    max_length=2,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True
                    )
                
                # Get logits for the first generated token and calculate the softmax
                step_logits = generated_ids['scores'][0]
                probs = torch.nn.functional.softmax(step_logits, dim=-1)
                
                top_prob, _ = torch.topk(probs, 1) # Get the top probabilities for each first token in the full batch
                top_probs = [prob.item() for prob in top_prob] # Get the raw values of the top probabilities
                
                generated_ids = accelerator.pad_across_processes(
                        generated_ids['sequences'], dim=1, pad_index=tokenizer.pad_token_id
                )
                
                input_ids, generated_ids, labels, top_probs = accelerator.gather((input_ids, generated_ids, labels, top_probs))
                input_ids = input_ids.cpu().numpy()
                generated_ids = generated_ids.cpu().numpy()
                labels = labels.cpu().numpy()
                top_probs = top_probs

                decoded_input = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                subset_inputs += decoded_input
                subset_preds += decoded_preds
                subset_labels += decoded_labels
                subset_probs += top_probs
        
        accelerator.wait_for_everyone()    
        if accelerator.is_main_process: 
            subset = test_subset_name
            eval_metric = calc_metric(subset_preds, subset_labels)
            eval_metric = {f'{k}': v for k, v in eval_metric.items()}
            accelerator.print(f'\nEval Metric on {subset}: ', eval_metric)
            
            subset_name = [subset] * len(subset_inputs)
                
            all_inputs.extend(subset_inputs)
            all_preds.extend(subset_preds)
            all_labels.extend(subset_labels)
            all_subsets.extend(subset_name)
            all_probs.extend(subset_probs)
            
                
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        preds_df = pandas.DataFrame({"input": all_inputs, 
                                     "label": all_labels, 
                                     "preds": all_preds,
                                     "subset": all_subsets,
                                    "label_prob": all_probs
                                     })
        eval_metric = calc_metric(all_preds, all_labels)
        accelerator.print('Overall Dataset Metrics: ', eval_metric)
        
        #Store outputs for each epoch
        if not os.path.exists(f'test-predictions/'):
            os.makedirs(f'test-predictions/')
        if args.outfile == 'default':
            preds_df.to_csv(f'test-predictions/{args.checkpoint_dir.replace("/", "-")}.csv', index=False)
        else:
            preds_df.to_csv(f'test-predictions/{args.outfile}.csv', index=False)