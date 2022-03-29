import pandas as pd
import os
import torch
from torch import nn
import json
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertConfig, BertForTokenClassification
from Preprocess import loadData


class NerDataset(Dataset):
    def __init__(self, data, is_train):
        self.data = data
        self.is_train = is_train

    def __getitem__(self, index):
        if self.is_train:
            return {
                'input_ids': torch.tensor(self.data[index]['input_ids'], dtype=torch.long),
                'mask': torch.tensor(self.data[index]['attention_mask'], dtype=torch.bool),
                'token_type_ids': torch.tensor(self.data[index]['token_type_ids'], dtype=torch.long),
                'targets': torch.tensor(self.data[index]['labels_id'], dtype=torch.long)
            }
        else:
            return {
                'input_ids': torch.tensor(self.data[index]['input_ids'], dtype=torch.long),
                'mask': torch.tensor(self.data[index]['attention_mask'], dtype=torch.bool),
                'token_type_ids': torch.tensor(self.data[index]['token_type_ids'], dtype=torch.long)
            }

    def __len__(self):
        return len(self.data)


def makeTrainData(args):
    with open(args.data_dir+'processed_data.json', 'r') as f_obj:
        data = json.load(f_obj)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=23)
    print("FULL Dataset: {}".format(len(data)))
    print("TRAIN Dataset: {}".format(len(train_data)))
    print("TEST Dataset: {}".format(len(test_data)))
    train_dataset = NerDataset(train_data, True)
    test_dataset = NerDataset(test_data, True)
    # print(train_dataset[0])
    train_dataloader = DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, args.test_batch_size, shuffle=True, num_workers=0)

    return train_dataloader, test_dataloader


def makePredictData(args):
    with open(args.predict_dir + 'processed_data.json', 'r') as f_obj:
        data = json.load(f_obj)
    pred_dataset = NerDataset(data, False)
    pred_dataloader = DataLoader(pred_dataset, args.train_batch_size, shuffle=False, num_workers=0)

    return pred_dataloader


class BertCrfModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.config = BertConfig.from_pretrained(args.model_config, num_labels=args.labels_len)
        self.bert = BertForTokenClassification.from_pretrained(args.pre_train_model, config=self.config)
        self.crf = CRF(num_tags=self.config.num_labels, batch_first=True)

    def forward(self, ids, mask, targets):
        logits = self.bert(input_ids=ids, attention_mask=mask)[0]
        if targets is not None:
            loss = self.crf(emissions=logits, mask=mask, tags=targets)
            loss = -1*loss
            logits = self.crf.decode(emissions=logits, mask=mask)
            return loss, logits
        logits = self.crf.decode(emissions=logits, mask=mask)
        return logits


if __name__ == "__main__":
    print(1)
