import glob
import torch
import nltk
import numpy as np
import pickle
import json
import jsonlines
import time
import argparse
from tqdm import tqdm
import pandas as pd
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from tabulate import tabulate

import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, BertTokenizer, AlbertTokenizer
from transformers.models.pegasus.modeling_pegasus import PegasusForConditionalGeneration
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_target_from_cnndm(path_to_cnndm: str, path_to_write: str, split: str = "valid"):
    """
    Extracts the target summary from the CNN/DM dataset.

    Args:
        path_to_cnndm: path to the CNN/DM dataset
        path_to_write: path to write the extracted target summaries
        split: the split to extract the target summaries from
    
    Returns:
        None
    """
    # use glob to get all the files in the CNN/DM dataset
    files = glob.glob(f"{path_to_cnndm}/cnndm.{split}.*.bert.pt")

    src_tgt_list = []

    for file in files:
        data = torch.load(file)
        src_tgt_list_in_this_split = [(" ".join(x['src_txt']), x['tgt_txt'].replace("<q>", " . ")) for x in data]
        src_tgt_list.extend(src_tgt_list_in_this_split)

    src_tgt_json_list = [{"summary": tgt, "source": src} for (src, tgt) in src_tgt_list]

    # write the extracted target summaries to a jsonlines file 
    with jsonlines.open(path_to_write, mode="w") as writer:
        writer.write_all(src_tgt_json_list)
    print(f"Saved {len(src_tgt_json_list)} target summaries to {path_to_write}")
    return 


def gen_pseudo_data_with_pegasus(split: str, st: int, ed: int):
    """
    Generates pseudo data using Pegasus.

    Args:
        split: the split to generate pseudo data from
        st: the starting index of the split
        ed: the ending index of the split
    
    Returns:
        None
    
    """
    ## load pegasus-large tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large").to(device)
    model.eval()

    ## load cnndm dataset
    cnndm = load_dataset("json", data_files={split: f"../data/cnndm/{split}.target.json"})
    cnndm = cnndm[split]

    ## generate pseudo data in batches
    pseudo_data = []
    batch_size = 5
    chunk_size = 1000
    bin_idx = st // chunk_size

    for i in tqdm(range(st, min(len(cnndm), ed), batch_size)):
        batch = cnndm[i:i+batch_size]
        batch = batch["summary"]

        # for each text in batch, randomly replace one sentence with <mask_1> and store in new_batch
        new_batch = []
        for text in batch:
            sentences = nltk.sent_tokenize(text)
            if len(sentences) > 1:
                idx = np.random.randint(0, len(sentences))
                sentences[idx] = "<mask_1>"
            new_batch.append(" ".join(sentences))

        batch_encoding = tokenizer(new_batch, max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(input_ids=batch_encoding["input_ids"], max_length=512, num_beams=5, min_length=0)
        
        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        pseudo_data.extend([{"pred": pred, "masked_summary": new_batch[idx], "orginal_summary": batch[idx]} for idx, pred in enumerate(preds)])

        # if len(pesudo_data) meets chunk_size, save the pseudo data to a jsonlines file
        if len(pseudo_data) >= chunk_size:
            with jsonlines.open(f"../data/cnndm/{split}.pseudo_pegasus_{bin_idx}.json", mode="w") as writer:
                writer.write_all(pseudo_data)
            print(f"Saved {len(pseudo_data)} pseudo data to ../data/cnndm/{split}.pseudo_pegasus_{bin_idx}.json")
            pseudo_data = []
            bin_idx += 1

    if pseudo_data:
        with jsonlines.open(f"../data/cnndm/{split}.pseudo_pegasus_{bin_idx}.json", mode="w") as writer:
            writer.write_all(pseudo_data)
        print(f"Saved {len(pseudo_data)} pseudo data to ../data/cnndm/{split}.pseudo_pegasus_{bin_idx}.json")
        
    return


def sample_level_correlation_summeval(human_metric, file_name, metric_list):
    """
    Computes the correlation between human metric and auto metric at the sample level.

    Args:
        human_metric: the human metric to compute correlation with
        file_name: the path to the file containing the data
        metric_list: a list of auto metrics to compute correlation with
    
    Returns:
        None
    
    """

    print(f'Human metric: {human_metric}')
    assert human_metric in ['coherence', 'relevance', 'consistency', 'fluency']
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    auto_metrics = metric_list
    headers = ['metric', 'spearman', 'pearsonr', 'kendalltau']
    metric_with_corr = []

    first_key, first_value = next(iter(data.items()))
    first_key, first_value = next(iter(first_value['sys_summs'].items()))
    metrics_to_append = [metric for metric in first_value['scores'] if "coherence" in metric and "coherence" != metric]
    metric_list.extend(metrics_to_append)

    for metric in auto_metrics:
        correlations = []
        for doc_id in data:

            target_scores = []
            prediction_scores = []
            sys_summs = data[doc_id]['sys_summs']
            for sys_name in sys_summs:
                prediction_scores.append(sys_summs[sys_name]['scores'][metric])
                target_scores.append(sys_summs[sys_name]['scores'][human_metric])

            if len(set(prediction_scores)) == 1 or len(set(target_scores)) == 1:
                continue

            correlations.append([
                spearmanr(target_scores, prediction_scores)[0],
                pearsonr(target_scores, prediction_scores)[0],
                kendalltau(target_scores, prediction_scores)[0],
            ])
  
        corr_mat = np.array(correlations)
        spearman,  pearman, ktau = np.mean(corr_mat[:, 0]), np.mean(corr_mat[:, 1]), np.mean(corr_mat[:, 2])
        metric_with_corr.append([metric, spearman, pearman, ktau])

    result_table = tabulate(metric_with_corr, headers=headers, tablefmt='simple')
    print(result_table)

    return result_table


def dataset_level_correlation_summeval(human_metric, file_name, metric_list):

    """
    Computes the correlation between human metric and auto metric at the dataset level.
    
    Args:
        human_metric: the human metric to compute correlation with
        file_name: the path to the file containing the data
        metric_list: a list of auto metrics to compute correlation with
    """

    print(f'Human metric: {human_metric}')
    assert human_metric in ['coherence', 'relevance', 'consistency', 'fluency']
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    auto_metrics = metric_list
    headers = ['metric', 'spearman', 'pearsonr', 'kendalltau']
    metric_with_corr = []

    for metric in auto_metrics:
        correlations = []
        target_scores = []
        prediction_scores = []
        for doc_id in data:

            sys_summs = data[doc_id]['sys_summs']
            for sys_name in sys_summs:
                prediction_scores.append(sys_summs[sys_name]['scores'][metric])
                target_scores.append(sys_summs[sys_name]['scores'][human_metric])

        correlations.append([
            spearmanr(target_scores, prediction_scores)[0],
            pearsonr(target_scores, prediction_scores)[0],
            kendalltau(target_scores, prediction_scores)[0],
        ])
        corr_mat = np.array(correlations)
        spearman,  pearman, ktau = np.mean(corr_mat[:, 0]), np.mean(corr_mat[:, 1]), np.mean(corr_mat[:, 2])
        metric_with_corr.append([metric, spearman, pearman, ktau])

    result_table = tabulate(metric_with_corr, headers=headers, tablefmt='simple')
    print(result_table)

    return result_table

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--split", type=str, default="valid")
    # parser.add_argument("--st", type=int, default=0)
    # parser.add_argument("--ed", type=int, default=10000)
    # args = parser.parse_args()
    
    # print("Using device:", device)
    # gen_pseudo_data_with_pegasus(split=args.split, st=args.st, ed=args.ed)

    for split in ['valid', 'test', 'train']:
        extract_target_from_cnndm(path_to_cnndm="../../Corpus/CNNDM", path_to_write=f"../data/cnndm/{split}.json", split=split)





    