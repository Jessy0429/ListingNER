import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from LabelsUtils import labels_map

MAX_LEN = 93
MODEL_NAME = 'bert-base-chinese'
# MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'


def loadData():
    data = []
    with open('../data/example/train_500.txt', 'r', encoding='utf-8') as f:
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
    return data


def tokenizeAndPreserveLabels(tokens, labels, tokenizer, label_map):
    labels.insert(0, "O")  # 给[CLS] token添加O标签
    labels = labels + ["O" for _ in range(MAX_LEN - len(labels))]
    labels_id = [label_map[item] for item in labels]

    result = tokenizer.encode_plus(
        tokens,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_attention_mask=True)

    result['labels_id'] = labels_id

    return result.data


if __name__ == "__main__":
    labels_mapping = labels_map()
    data = loadData()
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
    with open('../data/example/processed_data.json', 'w') as f_obj:
        json.dump(encoding_result, f_obj)
