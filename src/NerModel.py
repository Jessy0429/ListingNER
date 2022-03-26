import pandas as pd
import os
import torch
import json
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertPreTrainedModel, BertForTokenClassification




class NerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return {
            'input_ids': torch.tensor(self.data[index]['input_ids'], dtype=torch.long),
            'mask': torch.tensor(self.data[index]['attention_mask'], dtype=torch.bool),
            'token_type_ids': torch.tensor(self.data[index]['token_type_ids'], dtype=torch.long),
            'targets': torch.tensor(self.data[index]['labels_id'], dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)


def makeData(args):
    with open(args.data_dir+'processed_data.json', 'r') as f_obj:
        data = json.load(f_obj)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=23)
    print("FULL Dataset: {}".format(len(data)))
    print("TRAIN Dataset: {}".format(len(train_data)))
    print("TEST Dataset: {}".format(len(test_data)))
    train_dataset = NerDataset(train_data)
    test_dataset = NerDataset(test_data)
    # print(train_dataset[0])
    train_dataloader = DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, args.test_batch_size, shuffle=True, num_workers=0)

    return train_dataloader, test_dataloader


class BertCrfModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfModel, self).__init__(config)
        self.bert = BertForTokenClassification(config)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

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
