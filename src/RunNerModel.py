import os
import torch
import json
import logging
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from LabelsUtils import labels_unmap
from NerModel import *
from  collections import OrderedDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train(args, model, train_dataloader, test_dataloader):
    tr_dataloader = train_dataloader
    no_decay = ['bias', 'LayerNorm.weight', 'transitions']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(tr_dataloader) * args.train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_proportion * t_total,
                                                num_training_steps=t_total)


    logger.info("******Run Training******")

    model.zero_grad()
    tr_loss, logging_loss, tr_steps = 0.0, 0.0, 0
    best_f1 = 0.0

    for epoch in range(args.train_epochs):
        for idx, batch in enumerate(tr_dataloader):
            model.train()
            ids = batch['input_ids'].to(args.device, dtype=torch.long)  # 4,93 batch_size,max_len
            mask = batch['mask'].to(args.device, dtype=torch.bool)  # 4,93
            targets = batch['targets'].to(args.device, dtype=torch.long)  # 4,93

            outputs = model(ids=ids, mask=mask, targets=targets)
            initial_loss = outputs[0]
            tr_logits = outputs[1]

            if args.gradient_accumulation_steps > 1:
                loss = initial_loss / args.gradient_accumulation_steps
            else:
                loss = initial_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            logging_loss += loss.item()

            if (idx + 1) % args.gradient_accumulation_steps == 0:
                tr_steps += 1
                optimizer.step()
                scheduler.step()
                model.zero_grad()

                if tr_steps % 100 == 0:
                    logger.info("EPOCH = [%d/%d] train_steps = %d   loss = %f", epoch, args.train_epochs, tr_steps,
                                logging_loss / 100)
                    logging_loss = 0.0
        logger.info("Training loss epoch[%d/%d]: %f", epoch, args.train_epochs, tr_loss / tr_steps)
        f1 = evaluate(args, model, test_dataloader)
        if float(f1) >= best_f1:
            torch.save(model, args.output_dir + 'BertCrf.pth')
            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        tr_loss = 0.0
        tr_steps = 0

    # f1 = evaluate(args, model, test_dataloader)
    # model.train()
    # if f1 >= best_f1:
    #     torch.save(model, args.output_dir + 'BertCrf.pth')
        # Good practice: save your training arguments together with the trained model
        # torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


def evaluate(args, model, el_dataloader):
    model.eval()
    logger.info("***** Running evaluation *****")

    el_loss = 0
    eval_steps = 0
    el_preds, el_labels = [], []
    with torch.no_grad():
        for idx, batch in enumerate(el_dataloader):
            ids = batch['input_ids'].to(args.device, dtype=torch.long)
            mask = batch['mask'].to(args.device, dtype=torch.bool)
            targets = batch['targets'].to(args.device, dtype=torch.long)

            outputs = model(ids=ids, mask=mask, targets=targets)
            initial_loss = outputs[0]
            el_logits = outputs[1]
            el_loss += initial_loss.item()

            eval_steps += 1

            el_preds.extend(el_logits)
            active_accuracy = mask.view(-1) == 1
            targets = torch.masked_select(targets.view(-1), active_accuracy).cpu()
            el_labels.extend(np.array(targets).flatten())

    avg_loss = el_loss / eval_steps
    logger.info("Evaluation Loss: %f", avg_loss)

    el_preds = [y for x in el_preds for y in x]

    ret = classification_report(el_labels, el_preds, output_dict=True)
    logger.info("Evaluation F1_Score: %f", ret['accuracy'])
    return ret['accuracy']


def predict(args, model, pred_dataloader):
    model.eval()
    logger.info("***** Running Prediction *****")
    pred_ids = []
    pred_labels = []
    with torch.no_grad():
        for idx, batch in enumerate(pred_dataloader):
            ids = batch['input_ids'].to(args.device, dtype=torch.long)
            mask = batch['mask'].to(args.device, dtype=torch.bool)
            output = model(ids=ids, mask=mask, targets=None)
            pred_ids.extend(output)
    for ids in pred_ids:
        pred_labels.append([args.labels_unmapping[id] for id in ids][1:-1])
    return pred_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data/example/")
    parser.add_argument("--vob_file", default=None, help="预训练词表")
    parser.add_argument("--model_config", default='bert-base-chinese')
    parser.add_argument("--output_dir", default=None)

    parser.add_argument("--pre_train_model", default='bert-base-chinese', type=str, required=False,
                        help="预训练的模型文件，参数矩阵")

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="输入到bert的最大长度")
    parser.add_argument("--labels_len", default=128, type=int,
                        help="labels长度")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="训练集的batch_size")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="验证集的batch_size")
    parser.add_argument("--pred_batch_size", default=8, type=int,
                        help="训练集的batch_size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="梯度累计更新的步骤，用来弥补GPU过小的情况")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="学习率")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="最大的梯度更新")
    parser.add_argument("--train_epochs", default=16, type=float,
                        help="epoch 数目")
    parser.add_argument('--seed', type=int, default=23,
                        help="random seed for initialization")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="让学习增加到1的步数比例，在warmup_steps后，再衰减到0")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="权重衰减")
    parser.add_argument("--mlm_pretrain", default=False, type=bool,
                        help="mlm预训练")

    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.train_batch_size = 32
    args.test_batch_size = 32
    args.train_epochs = 10
    args.output_dir = '../models/BertCrf/'
    args.data_dir = '../data/train_data/'
    args.predict_dir = '../data/preliminary_test_a/'
    args.learning_rate = 2e-5
    args.labels_len = 105
    args.max_seq_length = 100
    args.labels_unmapping = labels_unmap('../data/train_data/train.txt')
    args.mlm_pretrain = True
    # args.pre_train_model = 'BertCrf'
    args.pre_train_model = 'hfl/chinese-roberta-wwm-ext'
    train_dataloader, test_dataloader = makeTrainData(args)
    if args.pre_train_model == 'bert-base-chinese' or args.pre_train_model == 'hfl/chinese-roberta-wwm-ext':
        model = BertCrfModel(args)
        # model = BertModel(args)
    else:
        model = torch.load(args.output_dir+'BertCrf.pth')

    if args.mlm_pretrain:
        state_dict_eb = torch.load("../models/MLM_BertCrf/Pre-training/bert_mlm_ep_1_eb.bin",
                                map_location=torch.device('cpu'))
        model.bert.bert.embeddings.load_state_dict(state_dict_eb)
        state_dict_ec = torch.load("../models/MLM_BertCrf/Pre-training/bert_mlm_ep_1_ec.bin",
                                map_location=torch.device('cpu'))
        model.bert.bert.encoder.load_state_dict(state_dict_ec)
        # for i, sub_model in enumerate(model.bert.bert.encoder.layer):
        #     sub_model = model.bert.bert.encoder.layer[i]
        #     state_dict = OrderedDict()
        #     for key, value in state_dict_ec.items():
        #         if int(key.split('.')[1]) == i:
        #             state_dict['.'.join(key.split('.')[2:])] = value
        #     sub_model.load_stact_dict(state_dict)

        args.output_dir = '../models/MLM_BertCrf/Fine-tuning'
    model.to(args.device)
    train(args, model, train_dataloader, test_dataloader)
    evaluate(args, model, test_dataloader)

    model = torch.load(args.output_dir+'BertCrf.pth')
    pred_dataloader = makePredictData(args)
    pred = predict(args, model, pred_dataloader)

    output_list = []
    i = 0
    j = 0
    with open('../data/preliminary_test_a/sample_per_line_preliminary_A.txt', 'r', encoding='utf-8') as f:
        for item in f.readlines():
            if item.strip() == "":
                continue
            else:
                if item.strip() == 0:
                    continue
                else:
                    for arr in item:
                        if arr == "\n":
                            break
                        if arr == " ":
                            output_list.append(arr+" O")
                        else:
                            output_list.append(arr+" "+pred[i][j])
                            j += 1
            i += 1
            j = 0
            output_list.append("")
    with open('../data/preliminary_test_a/output_BertCrf.txt', "w") as writer:
        for record in output_list:
            writer.write(record + '\n')



if __name__ == "__main__":
    # MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'
    main()
