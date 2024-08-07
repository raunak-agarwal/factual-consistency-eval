"""
This script is used to call the OpenAI API to predict whether the target text can be inferred from the source text.
Features:
- Caches the results in a JSON file to avoid repeated calls to the API for the same data
- Calculates metrics like accuracy, F1 score, and ROC AUC score
- Calculates the probability of the prediction based on the logprobs returned by the API

"""
import pandas as pd
import time
import os
import json
from sklearn.metrics import roc_auc_score
from summac.model_summac import SummaCZS, SummaCConv

model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cuda:0") # If you have a GPU: switch to: device="cuda"
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cuda:0", start_file="default", agg="mean")


def get_cache():
    if os.path.exists("../../data/eval-data/summac.json"):
        CACHE = pd.read_json("../../data/eval-data/summac.json", lines=True)
        print(f"Cache loaded with {len(CACHE)} rows")
        if len(CACHE) > 0:
            return CACHE
        else:
            print("Cache is empty")
            return pd.DataFrame(columns=['source', 'target', 'label', 'subset', 'summac_zs','summac_conv'])
    else:
        print("No cache found. Creating one...")
        dummy = pd.DataFrame(columns=['source', 'target', 'label', 'subset', 'summac_zs', 'summac_conv'])
        dummy.to_json("../../data/eval-data/summac.json", lines=True, orient='records')
        return dummy


def predict(source_text, target_text):
    zs_score = model_zs.score([source_text], [target_text])
    conv_score = model_conv.score([source_text], [target_text])
    
    return {'zs_score': zs_score["scores"][0], 'conv_score': conv_score["scores"][0]}

def get_metrics(data):
    
    y_true = data['label']
    y_pred_zs = data['summac_zs']
    y_pred_conv = data['summac_conv']
    
    roc_zs = roc_auc_score(y_true, y_pred_zs)
    roc_conv = roc_auc_score(y_true, y_pred_conv)

    
    return {'roc_auc_zs': roc_zs, 'roc_auc_conv': roc_conv}

def apply_summac_across_df(data):
    # out_df = pd.DataFrame(columns=['source', 'target', 'label', 'subset', 'prediction', 'prob'])
    for i, row in enumerate(data.itertuples()):        
        source = row[1]
        target = row[2]
        label = row[3]
        
        try:
            print(f"Processing row {i}")
            response = predict(source, target)
        except Exception as e:
            print(source, target)
            print(e)
            continue

        
        summac_zs = response['zs_score']
        summac_conv = response['conv_score']
        print(f"Label: {label}")
        print(f"SummaC ZS: {summac_zs}")
        print(f"SummaC Conv: {summac_conv}")
        
        out_row = {
            'source': source,
            'target': target,
            'label': label,
            'subset': row[4],
            'summac_zs': summac_zs,
            'summac_conv': summac_conv
        }

        with open("../../data/eval-data/summac.json", "a") as f:
            json_str = json.dumps(out_row)
            f.write(json_str + "\n")        
        
        print()

    return data
        
df = pd.read_json("../../data/test-data/benchmark_test_data.json", lines=True)

# Check CACHE and copy the following columns to df: 'prediction', 'prob'
CACHE = get_cache()

merged = pd.merge(df, CACHE, on=['source', 'target'], how='left')
merged = merged[['source', 'target', 'label_x', 'subset_x', 'summac_zs', 'summac_conv']]
merged = merged.rename(columns={'label_x': 'label', 'subset_x': 'subset'})

remaining = merged[merged['summac_zs'].isnull()]
print(f"Total Rows: {len(merged)}, Skipped Rows: {len(CACHE)}, Remaining rows: {len(remaining)}")

# remaining = remaining.sample(100).reset_index(drop=True)
x = apply_summac_across_df(remaining)

# open cache and do summary statistics over label and prediction
y = get_cache()
print(y['label'].value_counts())
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