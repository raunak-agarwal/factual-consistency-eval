"""
Features:
- Caches the results in a JSON file to avoid repeated calls to the API for the same data
- Calculates metrics like  ROC AUC score

"""
import pandas as pd
import time
import os
import json
from sklearn.metrics import roc_auc_score

from alignscore import AlignScore
# scorer = AlignScore(model='roberta-base', batch_size=32, device='cuda:1', ckpt_path='AlignScore/models/AlignScore-base.ckpt', evaluation_mode='bin_sp')
scorer = AlignScore(model='roberta-large', batch_size=12, device='cuda:1', ckpt_path='AlignScore/models/AlignScore-large.ckpt', evaluation_mode='bin_sp')





def get_metrics(data):
    y_true = data['label']
    # alignscore_base = data['alignscore_base']
    alignscore_large = data['alignscore_large']
    # roc = roc_auc_score(y_true, alignscore_base)
    roc = roc_auc_score(y_true, alignscore_large)
    return roc
        
df = pd.read_json("../../data/test-data/benchmark_test_data.json", lines=True)
scores = scorer.score(contexts=df.source.to_list(), claims=df.target.to_list())
roc = roc_auc_score(df.label.to_list(), scores)


df['alignscore_large'] = scores
df.to_json("../../data/eval-data/alignscore_large.json", orient='records', lines=True)
# df['alignscore_base'] = scores
# df.to_json("../../data/eval-data/alignscore_base.json", orient='records', lines=True)


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

