"""
Preprocssor for data used to generate soft labels,
tokenise inputs, return specific folds.
"""

from datetime import datetime

# TODO: cleanup
import numpy as np
import pandas as pd
from dataset_creation.high_agreement import gen_from_dataset
from datasets import Dataset, concatenate_datasets
from sklearn.model_selection import KFold
from transformers import PreTrainedTokenizer

from preprocessing.annotator_reliability import Annotations
from preprocessing.soft_label_utils import convert_df_to_tokenized_ds


class Preprocess:
    def __init__(self, config, dataset_path, tokenizer_name):
        self.config = config
        self.dataset_path = dataset_path
        self.tokenizer_name = tokenizer_name

    def load_and_preprocess(self):
        """
        Generate soft labels for an annotated dataset
        and tokenize the claim triple-text pair

        Args:
            label_type (str): label type, either "hard_label" or
                "soft_label".

        Returns:
            Dataset: the dataset prepared for huggingface processing.
        """
        # initialise annotations
        annotations = Annotations(
            self.dataset_path,
            self.config["num_annos"],
            self.config["labels2id"],
            calibrate_confidence=self.config["calibrate_confidence"],
            agreement_metric="krippendorff",
            drop_unrepresentative=self.config["drop_unrepresentative"],
            merge_labels=self.config["merge_labels"],
        )

        print("Dataframe loaded successfully.")

        # optionally calculate annotator reliability
        if self.config["calculate_reliability"]:
            annotations.calculate_annotator_reliability(
                alpha=self.config["intra_weighting"],
                beta=self.config["inter_weighting"],
            )

        # generate final labels based on label type
        annotations.generate_final_labels_and_sample_weights()

        print("Labels generated successfully")

        return annotations

    # TODO: clean up
    def return_fold(
        self,
        df,
        fold,
        label_type,
        claim_col,
        test_label_type="hard_label",
        llama=False,
        df_only=False,
    ):
        """Get the train and test set for a given fold.

        Args:
            df (pd.DataFrame): data frame.
            fold (int): fold number (from 0-4).
            label_type (str): type of label, either "hard_label"
                or "soft_label".
            claim_col (str): name of the claim column, either
                "claim_title" or "claim_triple".
            include_train_data (bool): unused. TODO: remove
            llama (bool): whether using llama or not.

        TODO: fix type
        Returns:
            (ds, ds): tuple of train and test datasets.
        """

        # include method parameter for hard labels (used in test data?)

        # TODO: modify how high agreement gained
        con_threshold = 3
        # List of columns that COULD contain annotations EXCLUDING reannos
        col_format = ["user_$_label", "user_$_confidence"]
        annotator_cols = []
        for i in range(1, 7):
            for col_name in col_format:
                annotator_cols.append(col_name.replace("$", str(i)))

        # split into high agreement and low agreement/single annotated
        high_agreement, other = gen_from_dataset(df, con_threshold, annotator_cols)

        # get the 5-fold split for high agreement
        kf = KFold(n_splits=5, random_state=43, shuffle=True)
        splits = list(kf.split(high_agreement))
        train_indices, test_indices = splits[fold]

        # combine other and train split
        train_df = pd.concat([other, high_agreement.iloc[train_indices]], axis=0).copy()
        train_df.loc[:, "label"] = train_df[label_type]

        # get test set
        test_df = high_agreement.iloc[test_indices].copy()
        test_df.loc[:, "label"] = test_df[test_label_type]

        if df_only:
            return train_df, test_df

        # print(len(train_df))
        # print(len(test_df))

        # print(len(train_df) + len(test_df))

        # print("Train df")
        # print(train_df[["tweet_id", "soft_label", "hard_label", "label", "sample_weight"]].head())
        # print("Test df")
        # print(test_df[["tweet_id", "soft_label", "hard_label", "label", "sample_weight"]].head())
        # print("Done")

        if not llama:
            # TODO: change final_soft_label column names
            train_ds = convert_df_to_tokenized_ds(
                train_df, self.tokenizer_name, claim_col
            )
            test_ds = convert_df_to_tokenized_ds(
                test_df, self.tokenizer_name, claim_col
            )
        else:
            train_ds = Dataset.from_pandas(train_df)
            test_ds = Dataset.from_pandas(test_df)

        # kf = KFold(n_splits=5, random_state=43, shuffle=True)

        # for i, (train_index, test_index) in enumerate(kf.split(test_ds)):
        # # print(f"Fold {i}:")
        # if i == fold:
        # fin_train_index = train_index
        # fin_test_index = test_index

        # train_from_test = test_ds.select(fin_train_index)

        # # print(f"Train from test size: {train_from_test.shape}")

        # test_from_test = test_ds.select(fin_test_index)

        # train_merged = concatenate_datasets([train_from_test, train_ds])

        return train_ds, test_ds
