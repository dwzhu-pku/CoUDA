import torch
import nltk
import numpy as np
import json
import time
import argparse

from pathlib import Path

current_file_path = Path(__file__).resolve()
root_file_path = (current_file_path / ".." / "..").resolve()

from scipy.spatial.distance import cosine
from transformers import AlbertTokenizer, AlbertForSequenceClassification

from coh_utils import sample_level_correlation_summeval, dataset_level_correlation_summeval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def get_avg_sim_score(summary, tokenizer, model):

    texts = nltk.sent_tokenize(summary)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    cos_sim_list = [1]

    for idx in range(len(texts)-1):
        cos_sim_list.append(1-cosine(embeddings[idx],embeddings[idx+1]))

    sim_score = np.mean(cos_sim_list)
    return sim_score


def get_albert_local_score(summary, tokenizer, model, aggregate="avg"):

    texts = nltk.sent_tokenize(summary)
    sop_score_list = [1]
    model = model.to(device)
    model.eval()
    for idx in range(len(texts)-1):
        encoding = tokenizer(texts[idx], texts[idx+1], return_tensors="pt")
        encoding = {key: tensor.to(device) for key, tensor in encoding.items()}
        with torch.no_grad():
            outputs = model(**encoding, labels=torch.LongTensor([1]).to(device))
        logits = outputs.sop_logits
        # logits = outputs.logits
        logits = torch.softmax(logits, dim=-1)
        sop_score_list.append((logits[0][0]-logits[0][1]).item())

    if aggregate == "avg":
        sop_score = np.mean(sop_score_list)
    elif aggregate == "min":
        sop_score = np.min(sop_score_list)
    return float(sop_score)

def get_albert_global_score(summary, tokenizer, model):

    nsp_score = 0
    model = model.to(device)
    model.eval()

    encoding = tokenizer(summary, return_tensors="pt")
    encoding = {key: tensor.to(device) for key, tensor in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding, labels=torch.LongTensor([1]).to(device))
        logits = outputs.logits
        logits = torch.softmax(logits, dim=-1)
        nsp_score = (logits[0][1]-logits[0][0]).item()

    return nsp_score

def get_albert_both_score(summary, tokenizer, model, aggregate="avg"):
    
    model = model.to(device)
    model.eval()
    global_score = 0
    texts = nltk.sent_tokenize(summary)
    local_score_list = [1]
    text_list = [summary]
    pair_list = list()
    for idx in range(len(texts)-1):
        pair_list.append(f"{texts[idx]} {texts[idx+1]}")
    text_list.extend(pair_list)
    encoding = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
    encoding = {key: tensor.to(device) for key, tensor in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        logits = torch.softmax(logits, dim=-1)
        global_score = (logits[0][1]-logits[0][0]).item()
        local_score_list.extend((logits[1:, 1]-logits[1:, 0]).tolist())
    if aggregate == "avg":
        local_score = np.mean(local_score_list)
    elif aggregate == "min":
        local_score = np.min(local_score_list)
    return (global_score, local_score)

def get_albert_both_score_fast(summary, tokenizer, model, aggregate="avg"):
    
    model = model.to(device)
    model.eval()
    global_score = 0
    texts = nltk.sent_tokenize(summary)

    # calculate global score
    encoding = tokenizer(summary, return_tensors="pt", padding=True, truncation=True)
    encoding = {key: tensor.to(device) for key, tensor in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        logits = torch.softmax(logits, dim=-1)
        global_score = (logits[0][1]-logits[0][0]).item()
    
    # calculate local score
    local_score_list = [1]
    pair_list = list()
    for idx in range(len(texts)-1):
        pair_list.append(f"{texts[idx]} {texts[idx+1]}")
    if len(pair_list) > 0:
        encoding = tokenizer(pair_list, return_tensors="pt", padding=True, truncation=True)
        encoding = {key: tensor.to(device) for key, tensor in encoding.items()}
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            logits = torch.softmax(logits, dim=-1)
            local_score_list.extend((logits[:, 1]-logits[:, 0]).tolist())
    if aggregate == "avg":
        local_score = np.mean(local_score_list)
    elif aggregate == "min":
        local_score = np.min(local_score_list)
        
    return (global_score, local_score)


def get_correlation_results(path_to_file: str, path_to_save: str):

    aspect = 'coherence'
    metric_list = []

    print("------------Start sample level correlation evaluation------------")
    sample_level_correlation_tabel = sample_level_correlation_summeval(aspect, path_to_file, metric_list)
    print("------------Start dataset level correlation evaluation------------")
    dataset_level_correlation_tabel = dataset_level_correlation_summeval(aspect, path_to_file, metric_list)

    with open(path_to_save, "w") as f:
        f.write("------------Start sample level correlation evaluation------------\n")
        f.write(f"{sample_level_correlation_tabel}\n")
        f.write("------------Start dataset level correlation evaluation------------\n")
        f.write(f"{dataset_level_correlation_tabel}\n")

    return

def helper_merge_score(scores_dict, key, candidate1, candidate2, enum_list):

    if candidate1 in scores_dict and candidate2 in scores_dict:
        for xlambda in enum_list:
            scores_dict[f'{key}_lambda_{xlambda}'] = scores_dict[candidate1] * (1-xlambda) + scores_dict[candidate2] * xlambda

    return


def get_albert_merged_score(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    for key in data:
        sys_summs = data[key]["sys_summs"]
        for model_name in sys_summs.keys():
            scores = sys_summs[model_name]["scores"]

            helper_merge_score(scores, "albert_merged_xxlarge_coherence", "albert_global_xxlarge_coherence", "albert_local_avg_xxlarge_coherence", [0.5])

    with open(filename, 'w') as fout:
        fout.write(json.dumps(data, indent=4, ensure_ascii=False))
    
    

def calculate(args, filename):

    """
    Calculate the correlation between the scores and the human evaluation scores.
    
    Args:
        args: The arguments.
        
    Returns:
        None
    """

    tokenizer = AlbertTokenizer.from_pretrained(args.path_to_ckp)
    model = AlbertForSequenceClassification.from_pretrained(args.path_to_ckp)

    with open(filename, 'r') as fin:
        data=json.load(fin)

    failed_cases_cnt = 0
    
    st = time.time()
    for key in data.keys():
        print(f"src: {key}")
        sys_sums = data[key]['sys_summs']
        for model_name in sys_sums.keys():
            hypo = sys_sums[model_name]['sys_summ']
            coh_score = 0
            try:
                if args.mode in ["albert_local"]:
                    coh_score = get_albert_local_score(hypo, tokenizer, model, args.aggregate)
                elif args.mode in ["albert_global"]:
                    coh_score = get_albert_global_score(hypo, tokenizer, model)
                elif args.mode in ["albert_both"]:
                    coh_score = get_albert_both_score_fast(hypo, tokenizer, model)
                
                
                if "both" in args.mode:
                    print(f"{model_name}: global: {coh_score[0]:.3f}, local: {coh_score[1]:.3f}")
                else:
                    print(f"{model_name}: {coh_score:.3f}")

            except Exception as e:
                print(f"Error occurred for in case {key}, model {model_name}, {e}")
                failed_cases_cnt += 1
                continue
            if "both" not in args.mode:
                prefix = f"{args.mode}_{args.arch}" if "global" in args.mode else f"{args.mode}_{args.aggregate}_{args.arch}"
                sys_sums[model_name]['scores'][f'{prefix}_coherence'] = coh_score
            else:
                prefix_global = f"albert_global_{args.arch}"
                prefix_local = f"albert_local_{args.aggregate}_{args.arch}"
                sys_sums[model_name]['scores'][f'{prefix_global}_coherence'] = coh_score[0]
                sys_sums[model_name]['scores'][f'{prefix_local}_coherence'] = coh_score[1]
        
        print(f"Finished in {time.time()-st:.3f}s, failed in {failed_cases_cnt} cases")

    with open(filename, 'w') as fout:
        fout.write(json.dumps(data, indent=4, ensure_ascii=False))

    return


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="albert_global")
    parser.add_argument("--arch", type=str, default="xxlarge")
    parser.add_argument("--aggregate", type=str, default="avg")
    parser.add_argument("--path_to_ckp", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="summeval")
    args = parser.parse_args()
    if args.dataset == "summeval":
        file_name = (root_file_path / "data/eval" / "summeval" / "summeval.json").resolve().as_posix()
    else:
        raise ValueError(f"dataset {args.dataset} not supported")
    print(f"file_name: {file_name}")

    if args.path_to_ckp is not None:
        path_to_name = args.path_to_ckp.split("/")[-1]
    else:
        path_to_name = args.mode.split("_")[0] + "_" + args.arch
    mode_to_name = "_".join(args.mode.split("_")[1:])
    path_to_save = root_file_path / "results" / "corr_scores" / args.dataset / f"{path_to_name}_{mode_to_name}"
    path_to_save = path_to_save.resolve()

    calculate(args, file_name)
    get_albert_merged_score(file_name)
    get_correlation_results(file_name, path_to_save)


if __name__ == "__main__":
    main()