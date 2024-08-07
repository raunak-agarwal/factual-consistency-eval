#accelerate launch finetune.py --args_to_my_script

from transformers import SchedulerType, get_scheduler, set_seed, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator

from datasets import Dataset
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.metrics import roc_auc_score
import evaluate

import pandas as pd
import numpy as np

from functools import partial
# from data.helper import load_json_file
from tqdm.auto import tqdm

import os
import argparse
import math

# import nltk
# nltk.download('punkt')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default='data/seahorse-train.jsonl')
    parser.add_argument('--val_dataset', type=str, default='data/seahorse-validation.jsonl')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--model_name', type=str, default='google/flan-t5-small')
    parser.add_argument('--max_input_length', type=int, default=2048)
    parser.add_argument('--max_target_length', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_warmup_steps', type=int, default=10)
    parser.add_argument('--lr_scheduler_type', type=SchedulerType, default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument('--validation_split', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    #parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--wandb_project', type=str, default='finetune-t5-for-seahorse')
    parser.add_argument('--num_update_steps_per_epoch', type=int, default=100)
    parser.add_argument('--checkpointing_frequency', type=int, default=5)

    return parser.parse_args()

accuracy = evaluate.load('accuracy')
f1 = evaluate.load('f1')
precision = evaluate.load('precision')
recall = evaluate.load('recall')

PROMPT = """Can the target text be inferred from the source text? In other words, is the target text factually consistent with the source text? Answer with a "Yes" or "No"."""

def preprocess_examples(examples, max_input_length=2048, max_target_length=2):
    # encode the question-answer pairs
    source = examples['source']
    target = examples['target']
    label = examples['label']
    
    input_text = f"{PROMPT}\n\nSource Text: {source}\n\nTarget Text: {target}\n\nAnswer:"
    output_text = f"{label}"

    model_inputs = tokenizer(input_text, max_length=max_input_length, padding="max_length", truncation=True)
    labels = tokenizer(output_text, max_length=max_target_length, padding="max_length", truncation=True).input_ids
    # print(output_text, labels)

    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        # print(labels_example)
        labels_example = [labels_example if labels_example != 0 else -100]
        labels_with_ignore_index.extend(labels_example)
    
    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs


# df = pd.read_json("seahorse-train.jsonl", lines=True).sample(20)
# dataset = Dataset.from_pandas(df)

# dataset = dataset.map(preprocess_examples, remove_columns=["source", "target", "label", "__index_level_0__"])


def calc_metric(predictions, labels):
    #First convert "Yes" and "No" to 1 and 0
    predictions = [1 if pred == "Yes" else 0 for pred in predictions]
    labels = [1 if label == "Yes" else 0 for label in labels]
    
    # Calculate accuracy, precision, recall, and f1-score
    accuracy_score = accuracy.compute(references=labels, predictions=predictions)
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


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # dataset = load_json_file(args.dataset)
    # dataset = Dataset.from_pandas(pd.DataFrame(data=dataset))
    # dataset = dataset.map(partial(preprocess_examples, tokenizer=tokenizer, max_input_length=args.max_input_length, max_target_length=args.max_target_length), batched=True, num_proc=16)  
    
    train_dataset = pd.read_json(args.train_dataset, lines=True)
    train_dataset = Dataset.from_pandas(train_dataset)
    train_dataset = train_dataset.map(preprocess_examples, remove_columns=["source", "target", "label"])
    train_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    
    val_dataset = pd.read_json(args.val_dataset, lines=True)
    val_dataset = Dataset.from_pandas(val_dataset)
    val_dataset = val_dataset.map(preprocess_examples, remove_columns=["source", "target", "label"])
    val_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    
    # dataset = dataset.train_test_split(test_size=args.validation_split, seed=args.seed)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * len(train_dataloader)
    )

    # accelerator
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)

    # wandb
    if bool(args.wandb_project) & accelerator.is_main_process:
        import wandb
        wandb.init(project=args.wandb_project, config=args)
    
    max_train_steps = len(train_dataloader)*args.num_train_epochs
    progress_bar = tqdm(range(max_train_steps))
    global_steps = 0

    # Train the model
    for epoch in range(args.num_train_epochs):
        accelerator.print("Epoch: {}".format(epoch))
        model.train()
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            with accelerator.accumulate(model):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            progress_bar.update(1)
            global_steps += 1
            loss = loss.item()
            if bool(args.wandb_project) & accelerator.is_main_process:
                wandb.log({"loss": loss})
                            
            if (step + 1) % args.num_update_steps_per_epoch == 0:
                accelerator.print("Step: {}/{}".format(step + 1, max_train_steps))
                accelerator.print("Loss: {}".format(loss))
                accelerator.print("LR: {}".format(optimizer.param_groups[0]["lr"]))
                accelerator.print("\n")
                
                
        # evaluate on validation set
        model.eval()
        all_input = []
        all_preds = []
        all_labels = []
        for step, batch in tqdm(enumerate(val_dataloader),total=len(val_dataloader)):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            # valid loss
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss = outputs.loss.item()

            # text generation
            generated_ids = accelerator.unwrap_model(model).generate(
                input_ids = input_ids,
                attention_mask = attention_mask, 
                max_length=2,
                do_sample=False
                )


            generated_ids = accelerator.pad_across_processes(
                    generated_ids, dim=1, pad_index=tokenizer.pad_token_id
                )
            
            input_ids, generated_ids, labels = accelerator.gather((input_ids, generated_ids, labels))
            input_ids = input_ids.cpu().numpy()
            generated_ids = generated_ids.cpu().numpy()
            labels = labels.cpu().numpy()

            decoded_input = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            all_input += decoded_input
            all_preds += decoded_preds
            all_labels += decoded_labels
        
        accelerator.wait_for_everyone()
        
        # evaluate
        if accelerator.is_main_process: 
            eval_metric = calc_metric(all_preds, all_labels)
            accelerator.print('Metric: ', eval_metric)

        # checkpoints
        if epoch % args.checkpointing_frequency == 0:
            accelerator.save_state(f'checkpoints/epoch_{epoch}')

        # save predictions
        
        if accelerator.is_main_process:
            if not os.path.exists('preds'): os.makedirs('preds')
            preds_df = pd.DataFrame({"input": all_input, "preds": all_preds, "labels": all_labels})
            preds_df.to_json(f"preds/epoch_{epoch}.json", orient="split")

        accelerator.wait_for_everyone()

        if bool(args.wandb_project) & accelerator.is_main_process:
            wandb.log({"val_loss": val_loss})
            wandb.log({'validation predictions': wandb.Table(dataframe=preds_df.head(1000))})
            wandb.log(eval_metric)
            wandb.save('preds/epoch_{}.json'.format(epoch))
            wandb.save('checkpoints/epoch_{}/*'.format(epoch))
    
        accelerator.print("Valid loss: {}".format(val_loss))
        accelerator.print("\n")

    accelerator.wait_for_everyone()

    if bool(args.wandb_project) & accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained('final')
        wandb.save('final/*')
        wandb.finish()
    
