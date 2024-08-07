"""
clone minicheck repo and copy this file into the folder before running it.

"""
import pandas as pd
import time
import os
import json
from sklearn.metrics import roc_auc_score


from minicheck.minicheck import MiniCheck



def get_metrics(data):
    y_true = data['label']
    minicheck = data['minicheck']
    roc = roc_auc_score(y_true, minicheck)
    return roc
        
df = pd.read_json("../../../data/test-data/final_combined_test_data.json", lines=True)

src_list = df.source.to_list()
output_list = df.target.to_list()

scorer = MiniCheck(model_name='flan-t5-large', device=f'cuda:1', cache_dir='./ckpts')
pred_label, raw_prob, _, _ = scorer.score(docs=src_list, claims=output_list)


scores = raw_prob

    
roc = roc_auc_score(df.label.to_list(), scores)



df['minicheck'] = scores
df.to_json("../../../data/eval-data/minicheck.json", orient='records', lines=True)


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

