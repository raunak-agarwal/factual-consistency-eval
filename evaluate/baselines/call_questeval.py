"""
clone questeval repo 

"""
import pandas as pd
import time
import os
import json
from sklearn.metrics import roc_auc_score

from questeval.questeval_metric import QuestEval
questeval = QuestEval(no_cuda=False, qg_batch_size=48, clf_batch_size=60)
        
df = pd.read_json("../../data/test-data/benchmark_test_data.json", lines=True)

src_list = df.source.to_list()
output_list = df.target.to_list()

t1 = time.time()

scores = questeval.corpus_questeval(
    hypothesis=output_list, 
    sources=src_list
)['ex_level_scores']

t2 = time.time()

roc = roc_auc_score(df.label.to_list(), scores)

df['questeval'] = scores
df.to_json("../../data/eval-data/questeval.json", orient='records', lines=True)

# Get metrics for each subset
for subset in df['subset'].unique():
    try:
        print(f"Metrics for Subset {subset}")
        subset_data = df[df['subset'] == subset]
        print(f"Size: {len(subset_data)} rows")
        print(roc_auc_score(subset_data['label'], subset_data['questeval']))
        print()
    except Exception as e:
        print(e)
        continue
    
print(f"Overall ROC AUC: {roc}")
print(f"Time taken: {t2-t1}")
