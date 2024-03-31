import nltk
import numpy as np
import torch
import jsonlines
import argparse
import random

from tqdm import tqdm
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AutoConfig
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_file_path = Path(__file__).parent.parent.resolve()


def get_albert_score_batched(summary_list, tokenizer, model, aggregate="avg", batch_size=16):
    
    global_score_list = list()
    local_score_list = list()

    # add tqdm here
    for idx in tqdm(range(0, len(summary_list), batch_size)):
    # for idx in range(0, len(summary_list), batch_size):
        batched_summary_list = summary_list[idx:idx+batch_size]

        # calculate global score
        encoding = tokenizer(batched_summary_list, return_tensors="pt", padding=True, truncation=True)
        encoding = {key: tensor.to(device) for key, tensor in encoding.items()}
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            logits = torch.softmax(logits, dim=-1)
            global_score_list.extend((logits[:, 1]-logits[:, 0]).tolist())
        
        # calculate local score
        idx_range_list = list()
        text_list = list()
        for summary in batched_summary_list:
            st_idx = len(text_list)
            texts = nltk.sent_tokenize(summary)
            for idx in range(len(texts)-1):
                text_list.append(f"{texts[idx]} {texts[idx+1]}")
            ed_idx = len(text_list)
            idx_range_list.append((st_idx, ed_idx))

        if len(text_list) == 0:
            local_score_list.extend([1]*batch_size)
        else:
            encoding = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
            encoding = {key: tensor.to(device) for key, tensor in encoding.items()}
            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
                logits = torch.softmax(logits, dim=-1)
                # restore each summary's scores in batch
                for idx, (st_idx, ed_idx) in enumerate(idx_range_list):
                    if aggregate == "avg":
                        local_score_list.append(np.mean((logits[st_idx:ed_idx, 1]-logits[st_idx:ed_idx, 0]).tolist()+[1]))
                    elif aggregate == "min":
                        local_score_list.append(np.min((logits[st_idx:ed_idx, 1]-logits[st_idx:ed_idx, 0]).tolist()+[1]))

    assert len(global_score_list) == len(summary_list)
    assert len(local_score_list) == len(summary_list)
    
    return np.array(global_score_list) + np.array(local_score_list)
    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="albert")
    parser.add_argument("--dataset", type=str, default="insted_cnn")
    args = parser.parse_args()

    if args.dataset == "insted_cnn":
        path_to_data = root_file_path / "data/eval/insted/insted_cnn.json"
    elif args.dataset == "insted_wiki":
        path_to_data = root_file_path / "data/eval/insted/insted_wiki.json"
    else:
        raise ValueError("dataset not found")

    with jsonlines.open(path_to_data) as reader:
        eval_data = list(reader)

    random.seed(42)
    random.shuffle(eval_data)
    print(len(eval_data))

    pos_list = [data["pos"]  for data in eval_data]
    neg_list = [data["neg"]  for data in eval_data]

    if args.model == "albert":
        model_name_or_path = (root_file_path / "models/couda_model").resolve()
        tokenizer = AlbertTokenizer.from_pretrained(model_name_or_path)
        model = AlbertForSequenceClassification.from_pretrained(model_name_or_path)
        model = model.to(device)
        model.eval()

        pos_score_list = get_albert_score_batched(pos_list, tokenizer, model, batch_size=8)
        neg_score_list = get_albert_score_batched(neg_list, tokenizer, model, batch_size=8)

        assert len(pos_score_list) == len(neg_score_list)

        correct_cnt = np.sum(np.array(pos_score_list) > np.array(neg_score_list))
        print(f"model {args.model} on dataset {args.dataset}: {correct_cnt / len(eval_data)}")
    
    else:
        raise ValueError("model not implemented")

    return




if __name__ == "__main__":
    main()