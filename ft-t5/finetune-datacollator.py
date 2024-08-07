#accelerate launch finetune.py --args_to_my_script

from transformers import SchedulerType, get_scheduler, set_seed, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq

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
from tqdm.auto import tqdm

import os
import argparse
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default='data/tokenized/train/combined-train-tokenized')
    parser.add_argument('--val_dataset', type=str, default='data/tokenized/validation/')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--model_name', type=str, default='google/flan-t5-small')
    parser.add_argument('--max_input_length', type=int, default=2048)
    parser.add_argument('--max_target_length', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_warmup_steps', type=int, default=5000)
    parser.add_argument('--lr_scheduler_type', type=SchedulerType, default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument('--use_prodigy_optimizer', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb_project', type=str, default='finetune-t5-all-datasets')
    parser.add_argument('--num_update_steps_per_epoch', type=int, default=100)
    parser.add_argument('--checkpointing_frequency', type=int, default=3)
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--compile', action='store_true', default=False)
    parser.add_argument('--model_output_dir', type=str, default='model-output')

    return parser.parse_args()

accuracy = evaluate.load('accuracy')
f1 = evaluate.load('f1')
precision = evaluate.load('precision')
recall = evaluate.load('recall')


def calc_metric(preds, labs):
    #First convert "Yes" and "No" to 1 and 0
    # predictions = [1 if "Yes" in pred else 0 for pred in preds]        
    # labels = [1 if label == "Yes" else 0 for label in labels]
    
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


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    if args.precision == 'bf16' or args.precision == 'fp16':
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, 
                                  mixed_precision=args.precision)
    elif args.precision == 'tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    else:
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True # disable annoying warning about padding
    
    
    if args.compile:
        # torch.backends.cuda.enable_math_sdp(False)
        # torch.backends.cuda.enable_mem_efficient_sdp(False)
        # torch.backends.cuda.enable_flash_sdp(True)
        accelerator.print("Compiling model...")
        model = torch.compile(model)
    
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest", pad_to_multiple_of=8)
    
    train_dataset = Dataset.load_from_disk(args.train_dataset)
    accelerator.print("Loaded training dataset: ", args.train_dataset)
    
    # val_dataset = Dataset.load_from_disk(args.val_dataset)
    import glob
    val_dataset_paths = glob.glob(args.val_dataset+"/*")
    val_datasets = {}
    for val_path in val_dataset_paths:
        val_datasets[val_path] = Dataset.load_from_disk(val_path)
    
    accelerator.print("Loaded validation datasets: ", val_dataset_paths)
    
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collator)

    # Create a separate dataloader for each validation dataset
    val_dataloaders = {}
    for val_path, val_dataset in val_datasets.items():
        val_dataloaders[val_path] = accelerator.prepare(DataLoader(val_dataset, shuffle=False, 
                                               batch_size=args.eval_batch_size,
                                               num_workers=args.num_workers))

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
    
    if args.use_prodigy_optimizer:
        from prodigyopt import Prodigy
        optimizer = Prodigy(optimizer_grouped_parameters, lr=1., weight_decay=args.weight_decay, safeguard_warmup=True)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * len(train_dataloader),
    )
    
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    if bool(args.wandb_project) & accelerator.is_main_process:
        import wandb
        wandb.init(project=args.wandb_project, config=args)
    
    max_train_steps = len(train_dataloader)*args.num_train_epochs
    progress_bar = tqdm(range(max_train_steps))
    global_steps = 0

    loss_avg = 0
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
            loss_avg += loss
            if bool(args.wandb_project) & accelerator.is_main_process:
                if (step + 1) % args.num_update_steps_per_epoch == 0:
                    wandb.log({f"avg loss ({args.num_update_steps_per_epoch} steps)": loss_avg/args.num_update_steps_per_epoch})
                    wandb.log({"lr": optimizer.param_groups[0]["lr"]})
                    loss_avg = 0
                            
            if (step + 1) % args.num_update_steps_per_epoch == 0:
                accelerator.print("Step: {}/{}".format(global_steps + 1, max_train_steps))
                accelerator.print("Loss: {}".format(loss))
                accelerator.print("LR: {}".format(optimizer.param_groups[0]["lr"]))
                accelerator.print("\n")
                                
        model.eval()
        all_inputs = []
        all_preds = []
        all_labels = []
        overall_validation_loss = 0
        for val_dataloader_path, val_dataloader in val_dataloaders.items():
            subset_inputs = []
            subset_preds = []
            subset_labels = []
            for step, batch in tqdm(enumerate(val_dataloader),total=len(val_dataloader)):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                
                with torch.no_grad():
                    # outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    # val_loss = outputs.loss.item()

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
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                subset_inputs += decoded_input
                subset_preds += decoded_preds
                subset_labels += decoded_labels
                # overall_validation_loss += val_loss 
            
            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process: 
                subset = val_dataloader_path.split("/")[-1].replace("-tokenized", "").replace("combined-validation-", "").lower()
                eval_metric = calc_metric(subset_preds, subset_labels)
                eval_metric = {f'{k}': v for k, v in eval_metric.items()}
                wandb.log({f'{subset} metrics': eval_metric})
                accelerator.print(f'\nEval Metric on {subset}: ', eval_metric)
                
            all_inputs.extend(subset_inputs)
            all_preds.extend(subset_preds)
            all_labels.extend(subset_labels)
            
        # accelerator.print("Valid loss: {}\n".format(overall_validation_loss))
        accelerator.wait_for_everyone()

        if bool(args.wandb_project) & accelerator.is_main_process:
            # wandb.log({"val_loss": overall_validation_loss})
            preds_df = pd.DataFrame({"input": all_inputs, "labels": all_labels, "preds": all_preds})
            wandb.log({'validation predictions': wandb.Table(dataframe=preds_df.head(1000))})
            eval_metric = calc_metric(all_preds, all_labels)
            wandb.log({"epoch": epoch})
            wandb.log({'overall metrics': eval_metric})
            accelerator.print('Overall Dataset Metrics: ', eval_metric)
            
            #Store outputs for each epoch
            import os
            if not os.path.exists(f'output-predictions/'):
                os.makedirs(f'output-predictions/')
            preds_df.to_csv(f'output-predictions/epoch_{epoch}.csv', index=False)

        # checkpoints
        if epoch % args.checkpointing_frequency == 0:
            accelerator.save_state(f'checkpoints/epoch_{epoch}', safe_serialization=False)
        
    accelerator.wait_for_everyone()

    if bool(args.wandb_project) & accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        # if args.compile:
        #     accelerator.print("Uncompiling model...")
        #     model = model.reverse_bettertransformer()
        model.save_pretrained(args.model_output_dir+f"/{args.model_name}-{args.wandb_project}", safe_serialization=False)
        wandb.finish()