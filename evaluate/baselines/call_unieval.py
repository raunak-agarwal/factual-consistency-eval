"""
clone unieval repo and copy this file into the folder before running it.

"""
import pandas as pd
import time
import os
import json
from sklearn.metrics import roc_auc_score

from utils import convert_to_json
from metric.evaluator import get_evaluator

task = 'fact'


def get_metrics(data):
    y_true = data['label']
    unieval = data['unieval']
    roc = roc_auc_score(y_true, unieval)
    return roc
        
df = pd.read_json("../../../data/test-data/benchmark_test_data.json", lines=True).sample(300)

src_list = df.source.to_list()
output_list = df.target.to_list()

data = convert_to_json(output_list=output_list, src_list=src_list)
evaluator = get_evaluator(task)
eval_scores = evaluator.evaluate(data, print_result=True)

scores = []

for score in eval_scores:
    scores.append(score['consistency'])
    
roc = roc_auc_score(df.label.to_list(), scores)



df['unieval'] = scores
df.to_json("../../../data/eval-data/unieval.json", orient='records', lines=True)
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

