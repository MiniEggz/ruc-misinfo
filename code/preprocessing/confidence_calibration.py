"""
Implementation of confidence calibration from
Don't Waste a Single Annotation.
"""

import math

# %%
import numpy as np
import pandas as pd

from preprocessing.soft_label_utils import (
    convert_confidence,
    inverse_convert_confidence,
)

# define meta data, maybe change later?
NUM_OF_ANNOTATORS = 6
# NUM_OF_POSTS = 100
NUM_OF_LABELS = 3


# converting our data such that each annotation will be on its own row. (NON-ANNOTATIONS ARE STILL INCLUDED SO THERE ARE LOTS OF NANs)
def data_conversion(df):
    new_data = []
    for index, row in df.iterrows():
        tweet_info = row[:5].tolist()
        current_index = 5
        for i in range(6):
            user_info = [i + 1]
            user_info.extend(row[current_index : current_index + 5].tolist())
            combined_info = tweet_info + user_info
            new_data.append(combined_info)

            current_index = current_index + 5

    return pd.DataFrame(
        new_data,
        columns=[
            "tweet_id",
            "claim_time",
            "tweet_text",
            "claim_title",
            "claim_triple",
            "user_id",
            "user_note",
            "user_label",
            "user_secondary",
            "user_confidence",
            "user_duration_seconds",
        ],
    )


# part one of bayesian confidence calibration as described in DWASA


def part_one(df, confidence):
    prior = 1 / NUM_OF_LABELS  # TODO: maybe change to more accurate prior
    likelihood = confidence  # assumption from DWASA
    marge = prior * likelihood + (1 - prior) * (1 - likelihood)
    return (prior) * likelihood / marge


# Implements equations 3 and 4 from DWASA paper
def agreement(df, label, annotator, prior):
    # Posts annoteted by anotator
    anotatorSet = set(df[df["user_id"] == annotator]["tweet_id"])
    # Posts annotated by nonAnnotator
    nonAnotatorSet = set(df[df["user_id"] != annotator]["tweet_id"])
    # Posts annotated by both
    coAnotatorSet = anotatorSet.intersection(nonAnotatorSet)

    # Posts annoteted by anotator with label
    anotatorLabelSet = set(
        df[(df["user_id"] == annotator) & (df["user_label"] == label)]["tweet_id"]
    )
    # Posts annotated by nonAnnotator with label
    nonAnotatorLabelSet = set(
        df[(df["user_id"] != annotator) & (df["user_label"] == label)]["tweet_id"]
    )
    # Posts union of both
    coAnotatorLabelSet = anotatorLabelSet.intersection(nonAnotatorLabelSet)

    # posts where non annotator1 dont give label
    nonAnotatorNonLabelSet = set(
        df[(df["user_id"] != annotator) & (df["user_label"] != label)]["tweet_id"]
    )

    # eq 3 from paper
    numer1 = len((coAnotatorSet.intersection(coAnotatorLabelSet)))
    denom1 = len(
        coAnotatorSet.intersection(set(df[(df["user_label"] == label)]["tweet_id"]))
    )

    # prevents division by 0
    if numer1 < 3 or denom1 < 3:
        eq1 = prior
    else:
        eq1 = numer1 / denom1

    # eq 4 from paper
    numer2 = len((anotatorLabelSet.intersection(nonAnotatorNonLabelSet)))
    denom2 = len(
        coAnotatorSet.intersection(
            set(
                df[(df["user_id"] != annotator) & (df["user_label"] != label)][
                    "tweet_id"
                ]
            )
        )
    )

    # prevents division by 0
    if numer2 < 3 or denom2 < 3:
        eq2 = prior
    else:
        eq2 = numer2 / denom2

    return eq1, eq2


# part two of bayesian confidence calibration as described in DWASA (equation 2 from paper)
def part_two(user_label, user_id, df, prior):
    # print(prior)
    liklihood, nonLiklihood = agreement(df, user_label, user_id, prior)
    return prior * liklihood / (prior * liklihood + (1 - prior) * nonLiklihood)


def run_conf_cal(df, save_to_csv=False):
    new_df = data_conversion(
        df
    )  # new_df is one annotation per row. used for equations 3 and 4

    # read through df and calculate confidence
    for index, row in df.iterrows():
        for i in range(1, NUM_OF_ANNOTATORS + 1):
            norm_confidence = convert_confidence(row[f"user_{i}_confidence"])
            prior = part_one(df, norm_confidence)
            norm_con_cal = part_two(row[f"user_{i}_label"], i, new_df, prior)
            con_cal = inverse_convert_confidence(norm_con_cal)

            # debugging
            # if not math.isnan(norm_con_cal):
            #     print(norm_confidence, norm_con_cal, con_cal)

            if con_cal >= 1.0:
                df.at[index, f"user_{i}_cal_confidence"] = con_cal
            elif not math.isnan(con_cal):
                df.at[index, f"user_{i}_cal_confidence"] = 1.0

    if save_to_csv:
        df.to_csv("../../data/data_with_confidence_calibration.csv", index=False)
    return df


if __name__ == "__main__":
    df = pd.read_csv("../../data/UkraineMisinfoWithReannotation.csv")
    new_df = run_conf_cal(df, save_to_csv=False)
