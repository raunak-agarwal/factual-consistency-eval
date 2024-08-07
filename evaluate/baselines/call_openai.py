"""
This script is used to call the OpenAI API to predict whether the target text can be inferred from the source text.
Features:
- Caches the results in a JSON file to avoid repeated calls to the API for the same data
- Calculates metrics like accuracy, F1 score, and ROC AUC score
- Calculates the probability of the prediction based on the logprobs returned by the API

"""
from openai import OpenAI
import pandas as pd
import time
import json
import numpy as np
import os
import tiktoken
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

client = OpenAI()
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

SYSTEM_PROMPT = """You are a helpful assistant designed to output JSON. Answer the following question and present your output in a JSON consistent with the provided spec.\nThe question is: Can the target text be inferred from the source text? In other words, is the target text factually consistent with the source text? Answer with a "Yes" or "No" in the following JSON format: {"answer": "Yes"} or {"answer": "No"}\nMake sure to not include any additional information in the output.
"""

def get_cache():
    if os.path.exists("../../data/eval-data/gpt3.5.json"):
        CACHE = pd.read_json("../../data/eval-data/gpt3.5.json", lines=True)
        print(f"Cache loaded with {len(CACHE)} rows")
        if len(CACHE) > 0:
            return CACHE
        else:
            print("Cache is empty")
            return pd.DataFrame(columns=['source', 'target', 'label', 'subset', 'prediction', 'prob'])
    else:
        print("No cache found. Creating one...")
        dummy = pd.DataFrame(columns=['source', 'target', 'label', 'subset', 'prediction', 'prob'])
        dummy.to_json("../../data/eval-data/gpt3.5.json", lines=True, orient='records')
        return dummy

def is_valid_json(obj):
    "Check if the output from the API is a valid JSON object with the key 'answer' and value 'Yes'/'No'"
    try:
        x = json.loads(obj)['answer']
        if x != "Yes" and x != "No":
            return False
    except Exception as e:
        return False
    return True

def sum_logprobs(logprobs):
    "Sum the logprobs for 'Yes' and 'No' tokens in the logprobs object. (Second last element in the logprobs list)"
    logprobs = logprobs.content[-2].top_logprobs
    prob_dict = {"Yes": 0, "No": 0}
    for l in logprobs:
        token = l.token.lower().strip()
        prob = l.logprob
        if "yes" in token:
            prob_dict["Yes"] += np.exp(prob)
        if "no" in token:
            prob_dict["No"] += np.exp(prob)
    
    return prob_dict

def predict(source_text, target_text):
    INPUT = f"Source Text: {source_text}\nTarget Text: {target_text}\nNow please answer the question: Can the target text be inferred from the source text?\n" + """The output must be either {"answer": "Yes"} or {"answer": "No"}."""
    
    print(f"Total Tokens: {len(encoding.encode(INPUT))}")
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={ "type": "json_object" },
        logprobs=True, top_logprobs=10, temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": INPUT}
        ]
    )
    output_content = response.choices[0].message.content
    logprobs = response.choices[0].logprobs
    
    return {'output': output_content, 'logprobs': logprobs}

def get_metrics(data):
    # Invert probabilities for cases where prediction is "No"
    def invert_prob(row):
        label = row['label']
        preds = row['prediction']
        label_prob = row['prob']
                    
        if label == 'Yes':
            row['label'] = 1
        if label == 'No':
            row['label'] = 0
        if preds == 'Yes':
            row['prediction'] = 1
        if preds == 'No':
            row['prediction'] = 0              
            row['prob'] = 1 - label_prob
        return row

    data = data.apply(invert_prob, axis=1)
    
    y_true = data['label']
    y_pred = data['prediction']
    y_pred_prob = data['prob']
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred_prob)
    
    return {'accuracy': acc, 'f1': f1, 'roc_auc': roc}

def apply_openai_across_df(data):
    time.sleep(0.65)
    # out_df = pd.DataFrame(columns=['source', 'target', 'label', 'subset', 'prediction', 'prob'])
    for i, row in enumerate(data.itertuples()):        
        source = row[1]
        target = row[2]
        label = row[3]
        
        # l = source.split()
        # if len(l) > 11000:
        #     print(f"Skipping row {i} due to source length > 11000 tokens")
        #     continue
        
        try:
            print(f"Processing row {i}")
            response = predict(source, target)
        except Exception as e:
            # print(source, target)
            print(e)
            continue
            
        if not is_valid_json(response['output']):
            print(f"Invalid JSON: {response['output']}")
            continue
        
        answer = json.loads(response['output'])['answer']
        prob = sum_logprobs(response['logprobs'])[answer]
        print(f"Label: {label}")
        print(f"Answer: {answer}")
        print(f"Prob: {prob}")
        
        out_row = {
            'source': source,
            'target': target,
            'label': label,
            'subset': row[4],
            'prediction': answer,
            'prob': prob
        }

        with open("../../data/eval-data/gpt3.5.json", "a") as f:
            json_str = json.dumps(out_row)
            f.write(json_str + "\n")        
        
        print()

    return data
        
df = pd.read_json("../../data/test-data/final_combined_test_data.json", lines=True)

# Check CACHE and copy the following columns to df: 'prediction', 'prob'
CACHE = get_cache()

merged = pd.merge(df, CACHE, on=['source', 'target'], how='left')
merged = merged[['source', 'target', 'label_x', 'subset_x', 'prediction', 'prob']]
merged = merged.rename(columns={'label_x': 'label', 'subset_x': 'subset'})

remaining = merged[merged['prediction'].isnull()]
print(f"Total Rows: {len(merged)}, Skipped Rows: {len(CACHE)}, Remaining rows: {len(remaining)}")

# remaining = remaining.sample(36000).reset_index(drop=True)
x = apply_openai_across_df(remaining)

# open cache and do summary statistics over label and prediction
y = get_cache()
print(y['label'].value_counts())
print()
print(y['prediction'].value_counts())
print()
print(f"Overall Metrics: {get_metrics(y)}")


# Get metrics for each subset
for subset in y['subset'].unique():
    try:
        print(f"Metrics for Subset {subset}")
        subset_data = y[y['subset'] == subset]
        print(f"Size: {len(subset_data)} rows")
        print(get_metrics(subset_data))
        print()
    except Exception as e:
        print(e)
        continue