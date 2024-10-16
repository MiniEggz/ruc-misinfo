"""
Main class for the classification pipeline.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from dataset_creation.high_agreement import gen_from_dataset

from config_parser import log_args, merge_config, parse_args

# Load the data
from preprocessing.preprocess import Preprocess
from stats.dataset_stats import get_label_distribution, output_stats
from trainers.soft_label_trainer_base import SoftLabelTrainer


def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def view_stats(config, annotations, preprocessor):
    # whole dataset
    whole_dataset_dist = get_label_distribution(annotations.df, config["labels2id"])
    output_stats(whole_dataset_dist, "Whole Dataset")

    con_threshold = 3
    # List of columns that COULD contain annotations EXCLUDING reannos
    col_format = ["user_$_label", "user_$_confidence"]
    annotator_cols = []
    for i in range(1, 7):
        for col_name in col_format:
            annotator_cols.append(col_name.replace("$", str(i)))

    # split into high agreement and low agreement/single annotated
    high_agreement, other = gen_from_dataset(
        annotations.df.copy(), con_threshold, annotator_cols
    )
    # high agreement
    high_agreement_dist = get_label_distribution(high_agreement, config["labels2id"])
    output_stats(high_agreement_dist, "High agreement")

    # other
    other_samples_dist = get_label_distribution(other, config["labels2id"])
    output_stats(other_samples_dist, "Other samples")

    for fold in range(5):
        train_df, test_df = preprocessor.return_fold(
            df=annotations.df.copy(),
            fold=fold,
            label_type="hard_label",
            claim_col=config["claim_col"],
            test_label_type="hard_label",
            df_only=True,
        )
        train_dist = get_label_distribution(train_df, config["labels2id"])
        test_dist = get_label_distribution(test_df, config["labels2id"])

        output_stats(train_dist, f"Fold {fold} train")
        output_stats(test_dist, f"Fold {fold} test")


def main():
    # Get arguments and log to the user
    args = parse_args()
    log_args(args)

    config_file = args.config if args.config else "config.yaml"

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    config = merge_config(config, args)
    config["project_name"] = Path(config_file).stem

    set_seed(config["seed"])

    print("Config:")
    print(json.dumps(config, indent=2))

    dataset_path = config["dataset_path"]
    preprocessor = Preprocess(
        config, dataset_path, config["model_name"]
    )  # use different tokenizer

    annotations = preprocessor.load_and_preprocess()

    if config["stats"]:
        view_stats(config, annotations, preprocessor)
        return

    if config["annotator_graph"]:
        annotations.display_annotator_graph()
        return

    # get train and test ds for the fold
    # make sure to change back to hard label
    train_ds, test_ds = preprocessor.return_fold(
        df=annotations.df.copy(),
        fold=config["fold"],
        label_type=config["train_label_method"],
        claim_col=config["claim_col"],
        test_label_type="hard_label",
    )

    # debug soft label generation
    if config["debug_mode"]:
        # convert dataset back to dataframe
        df = pd.DataFrame(train_ds)
        print("Training set shape:")
        print(df.shape)

        df = pd.DataFrame(test_ds)
        print("Test set shape:")
        print(df.shape)

    # training with Bert
    trainer = SoftLabelTrainer(config, train_ds, test_ds)
    trainer.train()


if __name__ == "__main__":
    main()
