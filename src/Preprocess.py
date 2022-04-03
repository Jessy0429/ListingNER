import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from LabelsUtils import labels_map

MAX_LEN = 100
MODEL_NAME = 'bert-base-chinese'
# MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'


def loadData(data_pth, is_train):
    data = []
    if is_train:
        with open(data_pth, 'r', encoding='utf-8') as f:
            listing_sentence = []
            listing_labels = []
            for item in f.readlines():
                if item.strip() == "":
                    data.append({'tokens': listing_sentence, "labels": listing_labels})
                    listing_sentence = []
                    listing_labels = []
                else:
                    arr = item.strip().split()
                    if len(arr) == 1:
                        continue
                    listing_sentence.append(arr[0])
                    listing_labels.append(arr[1])
    else:
        with open(data_pth, 'r', encoding='utf-8') as f:
            listing_sentence = []
            for item in f.readlines():
                arrs = item.strip().split()
                if len(arrs) == 0:
                    continue
                for arr in arrs:
                    listing_sentence.extend(arr)
                data.append({'tokens': listing_sentence})
                listing_sentence = []
    return data


def tokenizeAndPreserveLabels(tokens, labels, tokenizer, label_map):
    result = tokenizer.encode_plus(
        tokens,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_attention_mask=True)
    if labels is not None:
        labels.insert(0, "O")  # 给[CLS] token添加O标签
        labels = labels + ["O" for _ in range(MAX_LEN - len(labels))]
        labels_id = [label_map[item] for item in labels]

        result['labels_id'] = labels_id
    return result.data


if __name__ == "__main__":
    labels_mapping = labels_map('../data/train_data/train.txt')
    data = loadData('../data/train_data/train.txt', True)
    # max = 0
    # for sentence in data:
    #     length = len(sentence['tokens'])
    #     if length > max:
    #         max = length
    # MAX_LEN = max  # 固定tokens长度
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    encoding_result = []
    for item in data:
        result = tokenizeAndPreserveLabels(item['tokens'], item['labels'], tokenizer, labels_mapping)
        encoding_result.append(result)
    with open('../data/train_data/processed_data.json', 'w') as f_obj:
        json.dump(encoding_result, f_obj)

    data = loadData('../data/preliminary_test_a/sample_per_line_preliminary_A.txt', False)
    encoding_result = []
    for item in data:
        result = tokenizeAndPreserveLabels(item['tokens'], None, tokenizer, labels_mapping)
        encoding_result.append(result)
    with open('../data/preliminary_test_a/processed_data.json', 'w') as f_obj:
        json.dump(encoding_result, f_obj)
