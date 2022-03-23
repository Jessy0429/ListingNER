import pandas as pd
import os
import torch
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForTokenClassification
from LabelsUtils import labels_unmap

TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
LEARNING_RATE = 1e-05
EPOCHS = 5
LABELS_LEN = 81
MODEL_NAME = 'bert-base-chinese'
# MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
labels_unmapping = labels_unmap()


class NerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return {
            'input_ids': torch.tensor(self.data[index]['input_ids'], dtype=torch.long),
            'mask': torch.tensor(self.data[index]['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(self.data[index]['token_type_ids'], dtype=torch.long),
            'targets': torch.tensor(self.data[index]['labels_id'], dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)


def makeData():
    with open('../data/example/processed_data.json', 'r') as f_obj:
        data = json.load(f_obj)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=23)
    print("FULL Dataset: {}".format(len(data)))
    print("TRAIN Dataset: {}".format(len(train_data)))
    print("TEST Dataset: {}".format(len(test_data)))
    train_dataset = NerDataset(train_data)
    test_dataset = NerDataset(test_data)
    # print(train_dataset[0])
    train_dataloader = DataLoader(train_dataset, TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, TEST_BATCH_SIZE, shuffle=True, num_workers=0)

    return train_dataloader, test_dataloader


def train(tr_dataloader, epoch, model, optimizer):
    tr_loss, tr_accuracy, tr_steps= 0, 0, 0
    tr_preds, tr_labels = [], []
    model.train()

    for idx, batch in enumerate(tr_dataloader):
        ids = batch['input_ids'].to(device, dtype=torch.long)  # 4,93 batch_size,max_len
        mask = batch['mask'].to(device, dtype=torch.long)  # 4,93
        targets = batch['targets'].to(device, dtype=torch.long)  # 4,93

        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
        initial_loss = outputs[0]
        tr_logits = outputs[1]
        tr_loss += initial_loss.item()

        tr_steps += 1
        if idx % 50 == 0:
            loss_step = tr_loss / tr_steps
            print(f"Training loss / 50 steps: {loss_step}")

        flattened_targets = targets.view(-1)  # 真实标签大小 batch_size * seq_len
        active_logits = tr_logits.view(-1, model.num_labels)  # 模型输出shape batch_size * seq_len, num_labels
        flattened_predictions = torch.argmax(active_logits,
                                             axis=1)  # 取出每个token对应概率最大的标签索引 shape (batch_size * seq_len,)
        # MASK
        active_accuracy = mask.view(-1) == 1  # batch_size * seq_len
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_preds.extend(predictions)
        tr_labels.extend(targets)

        tr_accuracy += accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())

        # 反向传播
        optimizer.zero_grad()
        initial_loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / tr_steps
    tr_accuracy = tr_accuracy / tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


def evaluate(model, te_dataloader):
    model.eval()
    te_loss, te_accuracy = 0, 0
    f1_score = 0
    eval_steps = 0
    te_preds, te_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(te_dataloader):
            ids = batch['input_ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            initial_loss = outputs[0]
            te_logits = outputs[1]
            te_loss += initial_loss.item()

            eval_steps += 1
            if idx % 100 == 0:
                loss_step = te_loss / eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")

            flattened_targets = targets.view(-1)
            active_logits = te_logits.view(-1, model.num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)
            active_accuracy = mask.view(-1) == 1
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            te_preds.extend(predictions)
            te_labels.extend(targets)

            te_accuracy += accuracy_score(targets.cpu().numpy(), predictions.cpu().numpy())

    eval_loss = te_loss / eval_steps
    eval_accuracy = te_accuracy / eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    labels = [labels_unmapping[id.item()] for id in te_labels]
    predictions = [labels_unmapping[id.item()] for id in te_preds]

    print(classification_report(labels, predictions))
    return labels, predictions



if __name__ == "__main__":
    train_dataloader, test_dataloader = makeData()

    # for idx, batch in enumerate(train_dataloader):
    #     print(idx)
    #     print(batch)

    model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=LABELS_LEN)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    model.to(device)

    # for epoch in range(EPOCHS):
    #     print(f"Training epoch: {epoch + 1}")
    #     train(train_dataloader, epoch, model, optimizer)
    # model.save_pretrained('../models/Bert/')

    model = BertForTokenClassification.from_pretrained('../models/Bert/')
    labels, predictons = evaluate(model, test_dataloader)



