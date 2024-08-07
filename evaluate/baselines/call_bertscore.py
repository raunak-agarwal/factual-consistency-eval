
import pandas as pd
import numpy as np
import time
import os
import json
from sklearn.metrics import roc_auc_score

from bert_score import score


def get_metrics(data):
    y_true = data['label']
    bertscore = data['bertscore']
    roc = roc_auc_score(y_true, bertscore)
    return roc
        
df = pd.read_json("../../data/test-data/final_combined_test_data.json", lines=True).tail(10000).sample(100)
src_list = df.source.to_list()
output_list = df.target.to_list()
scores = score(output_list, src_list, lang='en', verbose=True,rescale_with_baseline=True,  device="cuda:0", batch_size=64)

f1 = list(np.array(scores[2]))
# scale to 0 to 1 from -1 to 1
f1 = [(x + 1) / 2 for x in f1]
f1 = [0 if x < 0 else x for x in f1]

roc = roc_auc_score(df.label.to_list(), f1)



df['bertscore'] = f1
df.to_json("../../data/eval-data/bertscore.json", orient='records', lines=True)


# Get metrics for each subset
for subset in df['subset'].unique():
    try:
        print(f"Metrics for Subset {subset}")
        subset_data = df[df['subset'] == subset]
        print(f"Size: {len(subset_data)} rows")
        print(get_metrics(subset_data))
        print()
    except Exception as e:
        print(e)
        continue
    
print(f"Overall ROC AUC: {roc}")

