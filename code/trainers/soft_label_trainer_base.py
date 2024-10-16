from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import cuda
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from losses.losses import My_ECELoss


class TrainLossEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = 999
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.last_epoch = 0

    def on_log(self, args, state, control, **kwargs):
        # differentiate train logs from eval logs
        train_logs = [
            log for log in state.log_history if "loss" in log and "eval_loss" not in log
        ]

        # check train logs not empty
        if not train_logs:
            # print("WARNING: train logs are empty.")
            return

        # check that there are enough logs in the list
        epoch = 0 if state.epoch is None else state.epoch
        if len(train_logs) < epoch or epoch == self.last_epoch:
            # print("WARNING: have not moved to the next epoch yet.")
            return

        # get train loss
        train_loss = train_logs[-1]["loss"]

        # update best loss or add to counter
        if train_loss < self.best_loss:
            self.best_loss = train_loss
            self.epochs_without_improvement = 0
            self.best_epoch = state.epoch
            control.should_save = True
        else:
            self.epochs_without_improvement += 1

        # if not improvement, early stop
        if self.epochs_without_improvement >= self.patience:
            control.should_training_stop = True

        self.last_epoch = epoch


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        weights = inputs.pop("sample_weight")  # one sample = 0.975

        labels = inputs.pop(
            "labels"
        )  # labels from data, one sample = [0.83333, 0.08333, 0.083333]

        outputs = model(**inputs)
        logits = outputs[0]  # model predictions

        loss_function = nn.CrossEntropyLoss(reduction="none")  # label smoothing is 0

        labels = labels.to(torch.float32)  # ensure labels are probabilites

        unreduced_loss = loss_function(
            logits, labels
        )  # difference between labels and model preds

        # apply weights to the loss
        weighted_loss = unreduced_loss * weights

        # compute the final loss as the mean of weighted losses
        final_loss = weighted_loss.mean()

        return (final_loss, outputs) if return_outputs else final_loss


class SoftLabelTrainer:
    def __init__(self, config, train_ds, test_ds):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.train_ds = train_ds
        self.test_ds = test_ds

    def initialize_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def initialize_network(self):
        return AutoModelForSequenceClassification.from_pretrained(
            self.config["model_name"], num_labels=len(self.config["labels2id"])
        ).to(self.device)

    def initialize_training_args(self, output_dir_name: str):
        warmup_ratio = float(self.config["warmup_epochs"]) / float(
            self.config["num_epochs"]
        )
        return TrainingArguments(
            output_dir=output_dir_name,
            remove_unused_columns=False,
            learning_rate=self.config["learning_rate"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            gradient_accumulation_steps=1,
            num_train_epochs=self.config["num_epochs"],
            warmup_ratio=warmup_ratio,
            optim="adamw_torch",
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            # logging_steps=1,
            save_strategy="epoch",
            save_total_limit=1,
            weight_decay=0.01,
            metric_for_best_model="loss",
            load_best_model_at_end=True,
            report_to=["wandb"],
        )

    def report_scores(
        self, labels, predictions, metric_group_prepend="overall", dict_of_metrics={}
    ):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="macro"
        )
        # The global f1_metrics
        dict_of_metrics[metric_group_prepend + "_f1_macro"] = f1
        dict_of_metrics[metric_group_prepend + "_precision"] = precision
        dict_of_metrics[metric_group_prepend + "_recall"] = recall

        if metric_group_prepend == "overall":
            f1_by_class = f1_score(labels, predictions, average=None).tolist()
            for i, f1 in enumerate(f1_by_class):
                dict_of_metrics[metric_group_prepend + "_f1_class_" + str(i)] = f1
        return dict_of_metrics

    def get_preds_from_logits(self, logits):
        ret = np.zeros(logits.shape)
        ret[np.arange(len(logits)), np.argmax(logits, axis=1)] = 1

        return ret

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        final_metrics = {}

        # Deduce predictions from
        # predictions = get_preds_from_logits(logits)
        predictions = self.get_preds_from_logits(logits)

        self.report_scores(
            labels,
            predictions,
            metric_group_prepend="overall",
            dict_of_metrics=final_metrics,
        )
        # to extend: report scores for each individual class

        ECE_module = My_ECELoss()
        ece = ECE_module(torch.from_numpy(logits), torch.from_numpy(labels))
        final_metrics["ece"] = ece

        return final_metrics

    def train(self):
        model = self.initialize_network()
        tokenizer = self.initialize_tokenizer()

        # ensure model recognises pad token
        model.config.pad_token_id = tokenizer.pad_token_id

        if self.config["output_dir_name"] == "automatic":
            output_dir_name = f"./models/{self.config['dataset_name']}/{self.config['project_name']}_{self.config['seed']}/fold_{self.config['fold']}"
        else:
            output_dir_name = (
                self.config["output_dir_name"] + f"/fold_{self.config['fold']}"
            )

        wandb.init(
            project=f"{self.config['project_name']}_{self.config['seed']}",
            name=f"fold-{self.config['fold']}",
        )

        training_args = self.initialize_training_args(output_dir_name)

        early_stopping_callback = TrainLossEarlyStoppingCallback()

        trainer = MyTrainer(
            model=model,
            args=training_args,
            train_dataset=self.train_ds,
            eval_dataset=self.test_ds,
            compute_metrics=self.compute_metrics,
            callbacks=[early_stopping_callback],
            tokenizer=tokenizer,
        )
        trainer.train()

        # log best epoch
        best_epoch = early_stopping_callback.best_epoch
        best_epoch_logs = [
            log for log in trainer.state.log_history if log.get("epoch") == best_epoch
        ]
        wandb.log({"best_epoch": early_stopping_callback.best_epoch})
        wandb.log(best_epoch_logs[-1])

        wandb.finish()
