import re

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# TODO: check this is correct
def convert_confidence(confidence, num_classes=3):
    """
    Convert an annotator's confidence value in the range 1-5
    to the range 0-1.

    Parameters:
        confidence (int): the annotator's confidence in their primary label.

    Returns:
        int: confidence value in the range 0-1
    """
    return (1 / num_classes) + (
        ((num_classes - 1) / num_classes) * ((confidence - 1) / 4)
    )


def inverse_convert_confidence(confidence_as_prob):
    """
    Converts a probablity in range 0-1 to an annotator confidence value in the range 1-5.
    (Inverse of convert_confidence function. Necessary when doing confidence calibration.)
    Parameters:
        confidence_as_prob (float): the annotator's confidence in their primary label.

    Returns:
        int: confidence value in the range 1-5
    """
    # TODO: make more general for num annotators
    return (6 * confidence_as_prob) - 1


def create_user_soft_label(
    row, user_prefix, label_mapping, num_classes, calibrate_confidence=False
):
    """Generate the soft label of the given user.

    Args:
        row (pd.Series): a single row of the full annotation DataFrame.
        config (dict): pipeline configuration options.
        user_prefix (str): user's prefix in the form user_x or re_user_x.
        calibrate_confidence (bool): whether confidence calibration takes place.

    Returns:
        np.ndarray: vector of soft labels, with each dimension representing probability
                    of the sample fitting the class.
    """

    soft_label = np.zeros(num_classes)

    primary_label = row[f"{user_prefix}_label"]

    # user has not made an annotation
    if pd.isna(primary_label):
        return np.nan

    if calibrate_confidence:
        confidence_col = f"{user_prefix}_cal_confidence"
    else:
        confidence_col = f"{user_prefix}_confidence"

    # get stances and confidence from row
    secondary_label = row[f"{user_prefix}_secondary"]
    primary_confidence = int(row[confidence_col])

    # retrieve index of label
    primary_index = label_mapping[primary_label]

    # update primary label with the given confidence
    soft_label[primary_index] = convert_confidence(primary_confidence)

    if pd.isna(secondary_label) or secondary_label == "NotAva":
        # redistribute uniformly
        for i in range(len(soft_label)):
            if i != primary_index:
                soft_label[i] = (1 - soft_label[primary_index]) / (num_classes - 1)
    elif primary_label == secondary_label:
        # edge case where secondary has been changed
        # to be the same label - is equal to all
        # being redistributed to stance 2
        soft_label[primary_index] = 1
    else:
        # redistribute all remaining to stance 2
        secondary_index = label_mapping[secondary_label]
        soft_label[secondary_index] = 1 - soft_label[primary_index]

        # if secondary is higher than primary
        if soft_label[secondary_index] > soft_label[primary_index]:
            soft_label[secondary_index] = soft_label[primary_index]
            # case where secondary is bigger than primary, set secondary equal and redistribute rest to other label
            for i in range(len(soft_label)):
                remaining_probability = (
                    1 - soft_label[primary_index] - soft_label[secondary_index]
                )
                if i != primary_index and i != secondary_index:
                    soft_label[i] = remaining_probability / (num_classes - 2)

    return soft_label


def add_row_soft_labels(
    row, num_annotators, label_mapping, num_classes, calibrate_confidence=False
):
    """Add all possible soft labels to the given row.

    Parameters:
        row (pd.Series): a single row of the full annotation DataFrame.
        config (dict): pipeline configuration options.
        max_annotators (int): the number of annotators involved in the full dataframe.
        calibrate_confidence (bool): whether confidence calibration takes place.

    Returns:
        row [pd.Series]: Row containing generated soft labels.
    """
    for i in range(1, num_annotators + 1):
        valid_user_prefixes = [f"user_{i}"]

        if not calibrate_confidence:
            valid_user_prefixes.append(f"re_user_{i}")

        for prefix in valid_user_prefixes:
            row[f"{prefix}_soft_label"] = create_user_soft_label(
                row,
                prefix,
                label_mapping,
                num_classes,
                calibrate_confidence=calibrate_confidence,
            )

    return row


def add_soft_labels(
    df, num_annotators, label_mapping, num_classes, calibrate_confidence=False
):
    """Add soft labels to the dataframe and return the new dataframe.

    Args:
        df (pd.DataFrame): Pandas DataFrame.
        config (dict): dict containing experiment config.
        calibrate_confidence (bool): whether confidence calibration takes place.

    Returns:
        pd.DataFrame: dataframe containing generated soft labels for each user.
    """
    return df.apply(
        lambda row: add_row_soft_labels(
            row,
            num_annotators,
            label_mapping,
            num_classes,
            calibrate_confidence=calibrate_confidence,
        ),
        axis=1,
    )


def create_row_final_soft_label(
    row,
    reliability_dict,
    num_annotators=6,
):
    """Create final soft label for row with either one or two annotations.
       Adds this final soft label to the row "soft_label" and also
       calculates the final sample weight based on the reliability of the
       annotators that have annotated the sample.

    Args:
        row (pd.Series): single row of the annotation dataframe.
        reliability_dict (dict): dict containing username: reliability pairs.
        num_annotators (int): number of annotators on the project.

    Returns:
        pd.Series: the updated row containing columns of final_soft_label and sample_weight.
    """
    # TODO: convert to list comprehension?
    row_annotators = []
    # loop through each annotator
    for i in range(1, num_annotators + 1):
        prefix = f"user_{i}"
        # check if user has a soft label column that isn't NaN
        if isinstance(row[f"{prefix}_soft_label"], np.ndarray):
            # if so, append prefix to row_annotators list
            row_annotators.append(prefix)

    if len(row_annotators) == 0:
        raise ValueError(
            f"Row number {row.name} does not have annotations, remove this row from the dataset."
        )
    elif len(row_annotators) == 1:
        row["soft_label"] = row[f"{row_annotators[0]}_soft_label"]
        row["sample_weight"] = reliability_dict[row_annotators[0]]
    elif len(row_annotators) == 2:
        # weighted average of labels based on annotator reliability
        soft_label_1 = row[f"{row_annotators[0]}_soft_label"]
        reliability_1 = reliability_dict[row_annotators[0]]
        soft_label_2 = row[f"{row_annotators[1]}_soft_label"]
        reliability_2 = reliability_dict[row_annotators[1]]
        reliability_sum = reliability_1 + reliability_2
        row["soft_label"] = (reliability_1 / reliability_sum) * soft_label_1 + (
            reliability_2 / reliability_sum
        ) * soft_label_2
        row["sample_weight"] = reliability_sum / 2
    else:
        raise ValueError("This row has more than 2 annotators.")
    return row


def create_row_hard_label(row):
    """Creates the hard label for the row given there is already
       a soft label in place. This should always be the case.

    Args:
        row (pd.Series): row of the dataframe.

    Returns:
        pd.Series: modified row with the extra "hard_label"
            column.

    Raises:
        ValueError: where no soft label is available for this row -
            this should never be the case.
    """
    soft_label = row["soft_label"]

    # TODO: add check for soft labels

    soft_label = np.array(soft_label)

    hard_label = np.zeros_like(soft_label)

    # first occurence of max num
    hard_label[np.argmax(soft_label)] = 1

    row["hard_label"] = hard_label

    return row


# TODO: find way of separating soft and hard labels
# 3 columns, soft_label, hard_label, label
def generate_final_soft_labels(input_df, reliability_dict, num_annotators):
    """Generate soft labels for each row in an annotated DataFrame.
       Also generates the sample weightings based on annotator reliability scores.

    Parameters:
        config (dict):
        input_df: The annotated dataframe

    Returns:
        pd.DataFrame: final dataframe with soft labels and sample weightings.
    """
    df = input_df.copy()
    return df.apply(
        lambda row: create_row_final_soft_label(row, reliability_dict, num_annotators),
        axis=1,
    )


def generate_hard_labels(input_df, reliability_dict, num_annotators):
    """Generate hard labels for each row in an annotated DataFrame.

    Args:
        input_df (pd.DataFrame): The annotated DataFrame
        num_annotators (int): the number of annotators in the DataFrame

    Returns:
        pd.DataFrame: final DataFrame with hard labels.
    """
    df = input_df.copy()
    return df.apply(lambda row: create_row_hard_label(row), axis=1)


def count_num_annos(example, max_users=6):
    """Count the number of annotators in the csv file,
    assuming there is at least one annotator.

    Returns:
        int: the number of annotators
    """
    count = 0
    for i in range(1, max_users + 1):
        if not pd.isna(example[f"user_{i}_label"]):
            count += 1
    return count


# TODO: delete
# def load_and_process_dataframe(csv_path):
#     """
#     Load a Pandas Dataframe and append a column
#     for the number of annotators in each row.

#     Parameters:
#         csv_path (str): the path to the csv file to load

#     Returns:
#         A Pandas Dataframe of the csv data
#     """
#     df = pd.read_csv(csv_path)

#     # adding a num_annotators column
#     num_annos_list = []
#     for idx, row in df.iterrows():
#         num_annos_list.append(count_num_annos(row))

#     df["num_annotator"] = num_annos_list  # check if necessary
#     return df


# needs testing properly to ensure correct tokenization
def convert_df_to_tokenized_ds(
    df,
    tokenizer_name,
    claim_col,
    tweet_col="tweet_text",
    label_col="label",
    weight_col="sample_weight",
):
    """
    Tokenize a pair of claim and text and create a Huggingface Dataset object.

    Parameters:
        df (DataFrame): The Dataframe containing claim, text and soft labels
        tokenizer_name (str): The pre-trained tokenizer to apply
        claim_col (str): the column in the DataFrame containing the claim triple
        tweet_col (str): the column in the DataFrame containing tweet text

    Returns:
        Dataset: the Dataset prepared for training
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = Dataset.from_pandas(df)

    # use the title instead of the triple
    # claim_col = "claim_title"

    # Tokenize the text and retain the labels
    def tokenize_function(examples):
        # Tokenize the input pairs
        tokenized_inputs = tokenizer(
            examples[claim_col],
            examples[tweet_col],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        # Add the labels to the tokenized inputs
        tokenized_inputs["labels"] = examples[label_col]
        tokenized_inputs["sample_weight"] = examples[weight_col]

        return tokenized_inputs

    # Apply the tokenization and label inclusion
    ds = ds.map(tokenize_function, batched=True)
    columns_to_keep = ["input_ids", "attention_mask", "labels", "sample_weight"]
    columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]

    # Remove the unwanted columns
    ds = ds.remove_columns(columns_to_remove)

    print("Tokenization successful")

    return ds
