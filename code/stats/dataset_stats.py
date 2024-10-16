import numpy as np
import pandas as pd


def get_label_distribution(df, label_mapping):
    """Get the label distribution within a given dataframe.

    Args:
        df (pd.DataFrame): DataFrame containing either the whole dataset
            or a subset (such as 'high agreement' or 'other').
        label_mapping (dict): a mapping of label names to their corresponding
            number - e.g. {"misinfo": 0, "other": 1}; as a hard label misinfo
            appears [1, 0] and other [0, 1].
    """
    # get label counts
    label_indices = df["hard_label"].apply(np.argmax)
    label_counts = label_indices.value_counts()

    # add to dict
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    return {
        reverse_label_mapping[label]: count for label, count in label_counts.items()
    }


def output_stats(dist, title):
    print("------------")
    print(f"{title}:")
    total = 0
    for k, v in dist.items():
        print(f"{k}: {v}")
        total += v
    print(f"Total: {total}")
