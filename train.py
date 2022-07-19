# -*- coding: utf-8 -*-
import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from utils import set_seed, collate_fn, get_acc, prepare_scheduler, difference_between_list
from utils import create_log
from processor import WebDataProcessor
from dataset import WebDataset
from model import BaselineModel

# from data_backup import build_vocab_from_dataset, PairDataset


def train(args, model, train_features, benchmarks,):
    with open(args.log_dir, "a+", encoding="utf-8") as metric_file:
        metric_file.write("\n" + "="*20 + "\n")
        metric_file.write(f"train_batch_size = {args.train_batch_size} \
        gradient_accumulation_steps = {args.gradient_accumulation_steps} \
        learning_rate = {args.learning_rate} \
        loss_type = {args.loss_type} \
        alpha = {args.alpha}" + "\n")
    
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                  drop_last=False)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print('Total steps: {}'.format(total_steps))
    print('Warmup steps: {}'.format(warmup_steps))

    # scaler = GradScaler()  # TODO "torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling."

    params = [{"params": model.paragraph_encoder.parameters(),
               "lr": args.transformer_learning_rate}]
    params += [{"params": filter(lambda p: id(p) not in list(map(id, model.paragraph_encoder.parameters())), model.parameters()),
                "lr": args.learning_rate}]
    # optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = AdamW(params, eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scheduler = prepare_scheduler(args.scheduler, optimizer, warmup_steps, total_steps)

    num_steps = 0
    max_train_acc, max_dev_acc, max_test_acc = 0, 0, 0
    max_train_epoch, max_dev_epoch, max_test_epoch = 0, 0, 0
    for epoch in range(1, int(args.num_train_epochs) + 1):
        print("=" * 10 + "Epoch {} / {}".format(epoch, int(args.num_train_epochs)) + "=" * 10)
        model.zero_grad()
        epoch_loss = 0.0
        epoch_father_loss = 0.0
        epoch_previous_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            # print(batch)
            ids = batch.pop("ids")
            node_modal = batch.pop("node_modal")
            golden_previous_ids = batch.pop("golden_previous_ids")
            inputs = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs[0]
            epoch_loss += loss.detach() / args.train_batch_size
            epoch_father_loss += outputs[1].detach() / args.train_batch_size
            epoch_previous_loss += outputs[2].detach() / args.train_batch_size
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            # scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                if args.max_grad_norm > 0:
                    # scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            """if (num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                for tag, features in benchmarks:
                    print("=" * 10 + "Evaluation on {} set".format(tag) + "=" * 10)
                    f1, output = evaluate(args, model, features, labelfield, tag=tag)
                    if f1 > max_f:
                        max_f = f1
                        max_iter = (epoch, step)
                        print(f"max F1 score update at Eopch {epoch}, Step {step}")"""

        print(f"epoch_loss = {epoch_loss} \t epoch_father_loss = {epoch_father_loss} \t epoch_previous_loss = {epoch_previous_loss}")
        with open(args.log_dir, "a+", encoding="utf-8") as metric_file:
            metric_file.write(f"epoch_loss = {epoch_loss} \t epoch_father_loss = {epoch_father_loss} \t epoch_previous_loss = {epoch_previous_loss}" + "\n")
        
        # evaluate per epoch
        for tag, features in benchmarks:
            print("=" * 10 + "Evaluation on {} set".format(tag) + "=" * 10)
            parent_acc, previous_acc = evaluate(args, epoch, model, features, tag=tag)
            with open(args.log_dir, "a+", encoding="utf-8") as metric_file:
                metric_file.write(f"Epoch = {epoch}" + "\n")
                metric_file.write(f"Evaluate on {tag} set" + "\n")
                metric_file.write(f"parent_acc = {parent_acc}, previous_acc = {previous_acc}" + "\n")
                """if parent_acc > max_acc:
                    max_acc = parent_acc
                    max_epoch = epoch
                    print(f"max parent_acc score update at Eopch {epoch}")"""
                if tag == "train":
                    if parent_acc > max_train_acc:
                        max_train_acc = parent_acc
                        max_train_epoch = epoch
                        print(f"max parent_acc score on {tag} set update at Eopch {epoch}")
                        metric_file.write(f"max parent_acc score on {tag} set update at Eopch {epoch}")
                if tag == "dev":
                    if parent_acc > max_dev_acc:
                        max_dev_acc = parent_acc
                        max_dev_epoch = epoch
                        print(f"max parent_acc score on {tag} set update at Eopch {epoch}")
                        metric_file.write(f"max parent_acc score on {tag} set update at Eopch {epoch}")
                if tag == "test":
                    if parent_acc > max_test_acc:
                        max_test_acc = parent_acc
                        max_test_epoch = epoch
                        print(f"max parent_acc score on {tag} set update at Eopch {epoch}")
                        metric_file.write(f"max parent_acc score on {tag} set update at Eopch {epoch}")
                if args.earlystop > 0:
                    if epoch - max_dev_epoch >= args.earlystop:
                        print(f"max parent_acc score on dev set do not increase for {args.earlystop} epochs, early stop")
                        metric_file.write(f"max parent_acc score on dev set do not increase for {args.earlystop} epochs, early stop")
                        exit(0)

        # save model checkpoint
        if args.save_checkpoint:
            torch.save(model.state_dict(), os.path.join(args.model_path, f"checkpoint_{epoch}.pkl"))
            print("Model checkpoint saved")

    print(f"max parent_acc score on train set update at Eopch {max_train_epoch}")
    print(f"max parent_acc score on dev set update at Eopch {max_dev_epoch}")
    print(f"max parent_acc score on test set update at Eopch {max_test_epoch}")


def evaluate(args, epoch, model, features, tag='dev'):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    parent_goldens, parent_preds, previous_goldens, previous_preds = [], [], [], []
    node_modals = []  # TODO 0719
    for i_b, batch in enumerate(tqdm(dataloader)):
        model.eval()
        ids = batch.pop("ids")
        node_modal = batch.pop("node_modal")
        node_modals += node_modal[0]
        golden_previous_ids = batch.pop("golden_previous_ids").reshape(-1).tolist()
        inputs = {k: v.to(args.device) for k, v in batch.items()}
        parent_golden = inputs["golden_parent"].reshape(-1).tolist()
        previous_golden = inputs["golden_previous"].reshape(-1).tolist()
        if not args.golden_parent_when_evaluate:
            inputs["golden_parent"] = None
            inputs["golden_previous"] = None
        # inputs["golden_parent"] = None  # ???
        # inputs["golden_previous"] = None
        parent_goldens += parent_golden
        previous_goldens += previous_golden
        """goldens += [(np.array(inputs["golden_parent"], dtype=np.int64),
                     np.array(inputs["golden_previous"], dtype=np.int64))]"""
        with torch.no_grad():
            outputs = model(**inputs)
            predict_parents = outputs[3].reshape(-1).tolist()
            predict_previous_scores = outputs[4]
            # print(predict_parents, inputs["golden_parent"].tolist())
            # print(predict_previous_scores, inputs["golden_previous"].tolist())
            predict_previous = torch.argmax(predict_previous_scores, dim=-1).reshape(-1).tolist()
        """preds += [(np.array(predict_parents, dtype=np.int64),
                   np.array(predict_previous, dtype=np.int64))]"""
        parent_preds += predict_parents
        previous_preds += predict_previous
        # if i_b < 1:
        #     print("predict parents: ", predict_parents)
        #     print("golden parents: ", parent_golden)
        #     print("predict previous: ", predict_previous)
        #     print("golden previous: ", previous_golden)
        with open("evaluation_output_"+args.log_dir, "a+", encoding="utf-8") as output_file:
            # output_file.write(f"Epoch = {epoch}" + "\n")
            output_file.write("="*20 + f"Epoch = {epoch}" + "="*20)
            output_file.write(f"Evaluate on {tag} set" + "\t" + f"id = {ids}" + "\n")
            output_file.write(f"golden parents: {parent_golden}" + "\n")
            output_file.write(f"predict parents: {predict_parents}" + "\n")
            output_file.write(f"{difference_between_list(parent_golden, predict_parents)}" + "\n")
            output_file.write(f"golden previous ids: {[i-1 for i in golden_previous_ids][1:]}" + "\n")
            output_file.write(f"golden previous: {previous_golden}" + "\n")
            output_file.write(f"predict previous: {predict_previous}" + "\n")
            output_file.write(f"{difference_between_list(previous_golden, predict_previous)}" + "\n")
            

    # keys = np.array(keys, dtype=np.int64)
    # preds = np.array(preds, dtype=np.int64)
    # P, R, max_f1, metrics_by_type = get_f1(keys, preds,)
    parent_acc = get_acc(parent_goldens, parent_preds, mask_id=1000)
    previous_acc = get_acc(previous_goldens, previous_preds, mask_id=3)
    metrics = {"parent_acc": parent_acc, "previous_acc": previous_acc,}

    parent_acc_wo_figure = get_acc([id if (node_modals[i]!="Figure" and node_modals[i]!="Figure&Title") else 1000 for i, id in enumerate(parent_goldens)], parent_preds, mask_id=1000)
    metrics.update({"parent_acc_wo_figure": parent_acc_wo_figure})  # TODO 0719, Do not consider whether the father node of a figure is correctly predicted 
    # parent_acc_correct_figure = get_acc([id if (node_modals[i]!="Figure" and node_modals[i]!="Figure&Title") else 1000 for i, id in enumerate(parent_goldens)], parent_preds, mask_id=1000)
    # metrics.update({"parent_acc_correct_figure": parent_acc_correct_figure})
    
    # print(f"parent_acc = {parent_acc}, previous_acc = {previous_acc}")
    print(", ".join([f"{metric_name} = {metric_value}" for metric_name, metric_value in metrics.items()]))
    
    # output = {tag + "_precision": P * 100, tag + "_recall": R * 100, tag + "_f1": max_f1 * 100, }
    # print(output)
    # print(metrics_by_type)
    # return max_f1, output
    # return parent_acc, previous_acc
    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_set_dir", default="data_split/train", type=str)
    parser.add_argument("--dev_set_dir", default="data_split/dev", type=str)
    parser.add_argument("--test_set_dir", default="data_split/test", type=str)

    parser.add_argument("--test_only", default=False, action="store_true",
                        help="Directly doing test on test set, a pretrained model checkpoint should be given")
    parser.add_argument("--test_checkpoint_id", default=0, type=int, )
    parser.add_argument("--log_dir", default="log.txt", type=str, )

    parser.add_argument("--device_id", default=0, type=int, help="device id for GPU training.")
    parser.add_argument("--save_checkpoint", default=False, type=bool, help="save checkpoint for each epoch.")
    parser.add_argument("--golden_parent_when_evaluate", default=False, action="store_true")

    parser.add_argument("--model_name_or_path", default="hfl/chinese-roberta-wwm-ext-large", type=str)
    parser.add_argument("--model_path", default="model2", type=str)
    parser.add_argument("--data_cache_path", default="data_split", type=str)

    parser.add_argument("--config_name", default="hfl/chinese-roberta-wwm-ext-large", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="hfl/chinese-roberta-wwm-ext-large", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated.")

    parser.add_argument("--loss_type", default="margin", type=str, choices=["margin", "ce"])
    parser.add_argument("--layer_type", default="bilinear", type=str, choices=["linear", "bilinear", "attention", "mixed"])
    parser.add_argument("--additional_encoder", default=False, action="store_true")
    parser.add_argument("--additional_encoder_type", default="transformer", choices=["transformer", "lstm", "gru", "linear"])
    parser.add_argument("--preprocess_figure_node", default=False, action="store_true",
                        help="whether to convert url of all figures to a unified linguistic utterance, e.g. 'tu-pian', during data processing")
    

    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=1, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--transformer_learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam on finetuning PTLM.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam on other modules.")
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--num_train_epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--earlystop", default=-1, type=int)

    parser.add_argument("--alpha", default=0.5, type=float,
                        help="hyperparameter for joint loss.")
    parser.add_argument("--scheduler", default="linear", type=str, choices=["constant", "warmup", "linear", "cosine"],
                        help="type of learning rate scheduler.")

    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    if args.seed > 0:
        set_seed(args)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        # num_labels=args.num_class,
    )
    config.gradient_checkpointing = True  # TODO
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )


    # _, _, labelfield = build_vocab_from_dataset(train_file)
    # labelfield.add_one("None")

    train_ds = WebDataset(args.train_set_dir)
    dev_ds = WebDataset(args.dev_set_dir)
    test_ds = WebDataset(args.test_set_dir)

    processor = WebDataProcessor(args, tokenizer)
    if os.path.exists(os.path.join(args.data_cache_path, "train.pkl")):
        print("Loading features from archive ...")
        train_feature_file = open(os.path.join(args.data_cache_path, "train.pkl"), 'rb')
        train_features = pickle.load(train_feature_file)
        train_feature_file.close()
        dev_feature_file = open(os.path.join(args.data_cache_path, "dev.pkl"), 'rb')
        dev_features = pickle.load(dev_feature_file)
        dev_feature_file.close()
        test_feature_file = open(os.path.join(args.data_cache_path, "test.pkl"), 'rb')
        test_features = pickle.load(test_feature_file)
        test_feature_file.close()
    else:
        train_features = processor.get_features_from_dataset(train_ds)
        dev_features = processor.get_features_from_dataset(dev_ds)
        test_features = processor.get_features_from_dataset(test_ds)
        features_dict = {"train": train_features, "dev": dev_features, "test": test_features}
        for name in features_dict.keys():
            saved_feature_file = open(os.path.join(args.data_cache_path, f"{name}.pkl"), "wb")
            pickle.dump(features_dict[name], saved_feature_file)
            saved_feature_file.close()

    if args.test_only:
        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            # num_labels=args.num_class,
        )
        config.gradient_checkpointing = True  # TODO

        model = BaselineModel(args, config)
        load_path = os.path.join(args.model_path, f"checkpoint_{args.test_checkpoint_id}.pkl")
        print(f"Loading NN model pretrained checkpoint from {load_path} ...")
        model.load_state_dict(torch.load(load_path))
        model.to(args.device)
        evaluate(args, model, test_features, tag='test')
        exit(0)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    else:
        print(f"args.model_path: {args.model_path} already exists, still go on? y/[n]")
        c = input()
        if c != "y":
            exit(10)

    model = BaselineModel(args, config)
    model.to(args.device)

    # if len(processor.new_tokens) > 0:
    #     model.encoder.resize_token_embeddings(len(tokenizer))  # TODO

    benchmarks = (
        ("train", train_features),
        ("dev", dev_features),
        ("test", test_features),
    )

    train(args, model, train_features, benchmarks,)

if __name__ == '__main__':
    main()

"""import torch
from torch import nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def prepare_model(args):
    pass


def prepare_trainer(args):
    pass

if __name__ == '__main__':
    args = None
    trainer = prepare_trainer(args)
    trainer.train()"""