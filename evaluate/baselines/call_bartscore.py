"""
clone bartscore repo and copy this file into the folder before running it.

"""
import pandas as pd
import numpy as np
import time
import os
import json
from sklearn.metrics import roc_auc_score

from bart_score import BARTScorer


def get_metrics(data):
    y_true = data['label']
    bartscore = data['bartscore']
    roc = roc_auc_score(y_true, bartscore)
    return roc
        
df = pd.read_json("../../../data/test-data/final_combined_test_data.json", lines=True).tail(10000).sample(1000)
src_list = df.source.to_list()
output_list = df.target.to_list()

bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
bart_scorer.load(path='bart_score.pth')

scores = bart_scorer.score(src_list, output_list, batch_size=4)

# f1 = list(np.array(scores[2]))
# # scale to 0 to 1 from -1 to 1
# f1 = [(x + 1) / 2 for x in f1]
# f1 = [0 if x < 0 else x for x in f1]

roc = roc_auc_score(df.label.to_list(), scores)



df['bartscore'] = scores
df.to_json("../../../data/eval-data/bartscore.json", orient='records', lines=True)


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

