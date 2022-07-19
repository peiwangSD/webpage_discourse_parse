import enum
import torch
import random
import numpy as np
from collections import defaultdict
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_constant_schedule, get_cosine_schedule_with_warmup
from module.pair_classifier import father_id_to_previous_id

PREVIOUS_RELATION_dict = {0:"Continue", 1:"Break", 2:"Combine", 3:None}

def save_devide(a, b):
    if b != 0:
        return a / b
    else:
        return 0.0


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    assert len(batch) == 1  # TODO
    num_paragraph = len(batch[0]["input_ids"])

    max_len = max([len(paragraph) for instance in batch for paragraph in instance["input_ids"]])
    input_ids = [ [paragraph + [0] * (max_len - len(paragraph)) for paragraph in instance["input_ids"]] for instance in batch]
    input_mask = [ [[1.0] * len(paragraph) + [0.0] * (max_len - len(paragraph)) for paragraph in instance["input_ids"]] for instance in batch]
    father_labels = [instance["Father"] for instance in batch]
    previous_labels = [instance["Previous_Relation_ids"] for instance in batch]

    # TODO 0704
    previous_node_ids = father_id_to_previous_id([[idx+1 for idx in father_ids] for father_ids in father_labels])
    # previous_labels = [[idx if idx!=3 else 1 for idx in instance["Previous_Relation_ids"]] for instance in batch]  # abolish the "NA" type in previous relations, replacing by "Break"
    previous_labels = [[idx if (idx!=3 or previous_node_ids[i][j+1]==0) else 1 for j, idx in enumerate(lst)] for i, lst in enumerate(previous_labels)]


    ids = [instance["id"] for instance in batch]
    node_modal = [instance["Node_modal"] for instance in batch]  # TODO 0719
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    # print(father_labels)
    father_labels = torch.tensor(father_labels, dtype=torch.long)
    previous_labels = torch.tensor(previous_labels, dtype=torch.long)
    upper_triangluar_padding = torch.ones(num_paragraph+1, num_paragraph+1) - torch.tril(torch.ones(num_paragraph+1, num_paragraph+1))
    previous_node_ids = torch.tensor(previous_node_ids, dtype=torch.long)
    # labels = torch.tensor(labels, dtype=torch.long)
    # ss = torch.tensor(ss, dtype=torch.long)
    # os = torch.tensor(os, dtype=torch.long)

    # output = (input_ids, input_mask, father_labels, previous_labels)
    output = {"ids": ids,
              "node_modal": node_modal,
              "input_ids": input_ids,
              "input_mask": input_mask,
              "padding": upper_triangluar_padding,
              "golden_parent": father_labels,
              "golden_previous_ids": previous_node_ids,
              "golden_previous": previous_labels}
    return output


def prepare_scheduler(name, optimizer, num_warmup_steps, num_training_steps):
    scheduler_dict = {"linear": get_linear_schedule_with_warmup, "warmup": get_constant_schedule_with_warmup, "constant": get_constant_schedule, "cosine": get_cosine_schedule_with_warmup}
    param_dict = {"optimizer":optimizer, "num_warmup_steps":num_warmup_steps, "num_training_steps":num_training_steps}
    if name == "warmup":
        param_dict.pop("num_training_steps")
    if name == "constant":
        param_dict.pop("num_training_steps")
        param_dict.pop("num_warmup_steps")
    scheduler = scheduler_dict[name](**param_dict)
    return scheduler

def get_acc(goldens, preds, mask_id=-1):
    assert len(goldens) == len(preds)
    correct = 0
    total = 0
    for g, p in zip(goldens, preds):
        if g == mask_id:
            continue
        if g == p:
            correct += 1
        total += 1
    return correct / total * 100

"""def get_f1(key, prediction,):
    correct_by_relation = ((key == prediction) & (prediction != none_type_id)).astype(np.int32).sum()
    guessed_by_relation = (prediction != none_type_id).astype(np.int32).sum()
    gold_by_relation = (key != none_type_id).astype(np.int32).sum()

    # total metrics
    prec_micro = 1.0
    if guessed_by_relation > 0:
        prec_micro = float(correct_by_relation) / float(guessed_by_relation)
    recall_micro = 1.0
    if gold_by_relation > 0:
        recall_micro = float(correct_by_relation) / float(gold_by_relation)
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)

    # by type metrics
    metrics_by_type = dict()
    if id2label:
        golden_dict = defaultdict(int)
        predict_dict = defaultdict(int)
        correct_dict = defaultdict(int)
        key = list(key)
        prediction = list(prediction)
        for golden, predict in zip(key, prediction):
            if golden != none_type_id:
                golden_dict[id2label[golden]] += 1
            if golden != none_type_id and predict == golden:
                correct_dict[id2label[golden]] += 1
            if predict != none_type_id:
                predict_dict[id2label[predict]] += 1
        for t, num in golden_dict.items():
            r = correct_dict[t] / golden_dict[t]
            p = save_devide(correct_dict[t], predict_dict[t])
            f = save_devide(2 * p * r, p + r)
            metrics_by_type[t] = (p*100, r*100, f*100)

    return prec_micro, recall_micro, f1_micro, metrics_by_type"""

def difference_between_list(golden, predict):
    assert len(golden) == len(predict)
    error_dict = {}
    for i, (g, p) in enumerate(zip(golden, predict)):
        if g != p:
            error_dict[i] = f"{g} -> {p}"
    # print(error_dict)
    return error_dict


def create_log(args):
    with open(args.log_dir, "a+", encoding="utf-8") as metric_file:
        metric_file.write("\n" + "="*20 + "\n")
        metric_file.write(f"train_batch_size = {args.train_batch_size} \
        gradient_accumulation_steps = {args.gradient_accumulation_steps} \
        learning_rate = {args.learning_rate} \
        loss_type = {args.loss_type} \
        alpha = {args.alpha}" + "\n")

