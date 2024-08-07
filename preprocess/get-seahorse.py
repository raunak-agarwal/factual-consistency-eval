# import pandas as pd



import modin.pandas as pd 

import modin.config as cfg; print(cfg.CpuCount.get()); cfg.CpuCount.put(8); print(cfg.CpuCount.get())

from datasets import load_dataset

from tqdm import tqdm

from ftfy import fix_text


folder_path = "../data/seahorse_data/"

seahorse_train = pd.read_csv(folder_path + "train.tsv", sep="\t")
seahorse_validation = pd.read_csv(folder_path + "validation.tsv", sep="\t")
seahorse_test = pd.read_csv(folder_path + "test.tsv", sep="\t")

print(len(seahorse_train), len(seahorse_validation), len(seahorse_test))

seahorse_train = seahorse_train[seahorse_train['worker_lang'] == "en-US"]
seahorse_validation = seahorse_validation[seahorse_validation['worker_lang'] == "en-US"]
seahorse_test = seahorse_test[seahorse_test['worker_lang'] == "en-US"]


seahorse_train = seahorse_train[seahorse_train['question4'] != "Unsure"]
seahorse_validation = seahorse_validation[seahorse_validation['question4'] != "Unsure"]
seahorse_test = seahorse_test[seahorse_test['question4'] != "Unsure"]

print(len(seahorse_train), len(seahorse_validation), len(seahorse_test))

seahorse_train = seahorse_train.dropna(subset=['question4'])
seahorse_validation = seahorse_validation.dropna(subset=['question4'])
seahorse_test = seahorse_test.dropna(subset=['question4'])

print(len(seahorse_train), len(seahorse_validation), len(seahorse_test))


def apply_gem(row):
    gem_id = row['gem_id']
    if "wiki" in gem_id:
        row['gem'] = "wiki_lingua"
    if "xsum" in gem_id:
        row['gem'] = 'xsum'
    if "xlsum" in gem_id:
        row['gem'] = 'xlsum'
    return row


seahorse_train = seahorse_train.apply(apply_gem, axis=1)
seahorse_validation = seahorse_validation.apply(apply_gem, axis=1)
seahorse_test = seahorse_test.apply(apply_gem, axis=1)

print(seahorse_train['gem'].value_counts().to_dict())
print(seahorse_validation['gem'].value_counts().to_dict())
print(seahorse_test['gem'].value_counts().to_dict())


combined_train = []
combined_validation = []
combined_test = []


#all datasets get mapped to a dictionary, then we apply get_source to assign the source texts to seahorse

xlsum = load_dataset("GEM/xlsum", "english")
xlsum_validation = {example['gem_id']: example['text'] for example in xlsum['validation']}    
xlsum_test = {example['gem_id']: example['text'] for example in xlsum['test']}

xsum = load_dataset('GEM/xsum')
xsum_validation = {example['gem_id']: example['document'] for example in xsum['validation']}    
xsum_test = {example['gem_id']: example['document'] for example in xsum['test']}

wiki_lingua = load_dataset("gem", "wiki_lingua_english_en")
wiki_lingua_validation = {example['gem_id']: example['source'] for example in wiki_lingua['validation']}    
wiki_lingua_test = {example['gem_id']: example['source'] for example in wiki_lingua['test']}


source_dict = {
    "xsum": [xsum_validation, xsum_test],
    "xlsum": [xlsum_validation, xlsum_test],
    "wiki_lingua": [wiki_lingua_validation, wiki_lingua_test],
}


def get_source(row):
    gem = row["gem"]
    gem_id = row["gem_id"]
    source_dataset = source_dict[gem]
    if "val" in gem_id:
        source_dataset = source_dataset[0]
    else:
        source_dataset = source_dataset[1]

    source_text = source_dataset[gem_id]
    summary = row['summary']
    
    row["source"] = fix_text(source_text)
    row['summary'] = fix_text(summary).encode('utf-8', 'backslashreplace').decode('unicode_escape')

    return row


combined_train = seahorse_train.apply(get_source, axis=1)[['source', 'summary', 'question4']].rename(columns={"question4": "label", "summary": "target"})
combined_validation = seahorse_validation.apply(get_source, axis=1)[['source', 'summary', 'question4']].rename(columns={"question4": "label", "summary": "target"})
combined_test = seahorse_test.apply(get_source, axis=1)[['source', 'summary', 'question4']].rename(columns={"question4": "label", "summary": "target"})

print(len(combined_train), len(combined_validation), len(combined_test))


print("Writing to jsonl files...")
combined_train.to_json(folder_path + 'seahorse-train.jsonl', lines=True, orient="records", force_ascii=False)
combined_validation.to_json(folder_path + 'seahorse-validation.jsonl', lines=True, orient="records", force_ascii=False)
combined_test.to_json(folder_path + 'seahorse-test.jsonl', lines=True, orient="records", force_ascii=False)