import pandas as pd

# TODO: this needs to be added as a preprocessing step instead

def find_annotated_columns(series):
    """
    Finds the column names that have been annotated.

    Params:
        - pd.series - a pandas series (or equivalent) of a row from the
         misinfo csv, containing only the annotated columns.

    Returns:
        - list - containing the names of the columns that have annotations.
    """
    # Convert NAN and not NAN values to booleans.
    annotator_booleans = series.notna()

    column_names = annotator_booleans.index.to_list()
    values = annotator_booleans.to_list()
    annotated_columns = [col for col, val in zip(column_names, values) if val]

    return annotated_columns


# TODO: needs to be made more general
def print_ratios(samples_df, annotator_cols):
    """
    Prints the ratios of misinfo, debunk, and other in the high
    agreement dataframe.

    Params:
        - pd.dataframe - a pandas dataframe of high agreementannotated samples,
        with labels 'misinfo', 'debunk', and 'other'.
        - list - a list of the names of the columns containing the annotations.
    """
    annotated_cols = samples_df[annotator_cols]
    # As samples are high-agreement, first label will be the same as
    # all following labels (that aren't NAN).
    first_labels = annotated_cols.fillna(method="bfill", axis=1).iloc[:, 0]
    total_labels = len(first_labels)

    # Count label occurances:
    label_counts = first_labels.value_counts().to_dict()

    output = f"""
    -----------------------------
    High-agreement size: {total_labels}
    
    Label data:
        - Misinfo: 
            - Count {label_counts["misinfo"]}
            - Percent { (label_counts["misinfo"] / total_labels) * 100 }
        - Debunk: 
            - Count {label_counts["debunk"]}
            - Percent { (label_counts["debunk"] / total_labels) * 100 }
        - Other: 
            - Count {label_counts["other"]}
            - Percent { (label_counts["other"] / total_labels) * 100 }
    -----------------------------
    """

    print(output)


# TODO: don't want these to be separate, should be generated on the fly
def gen_from_dataset(samples, con_threshold, annotator_cols):
    """
    Function to seperate a dataset into two seperate datasets, one containing
    high annotator agreement and the other containing the rest of the annotations.

    Params:
        - pd.dataframe - the dataframe of samples to be seperated.
        - int - the confidence threshold for an annotation to be considered as
            possibly high agreement.
        - list - a list of the names of the columns containing the annotations.

    Returns:
        - tuple (pd.dataframe, pd.dataframe) - a tuple containing the high annotator
        agreement data and the non-high annotator agreement data, both as pandas
        dataframes.
    """
    high_agreement_samples = samples.copy()
    other_samples = samples.copy()
    low_agreement_indices = []

    for i, sample in samples.iterrows():
        annotated_cols = find_annotated_columns(sample[annotator_cols])

        # Can't calculate agreement with just 1 (or 0) annotators.
        if len(annotated_cols) <= 2:
            low_agreement_indices.append(i)
            continue

        # annotated_sample = sample[annotated_cols]

        # Seperating labels and confidence columns.
        annotated_label_cols = [col for col in annotated_cols if "label" in col]
        annotated_sample_labels = sample[annotated_label_cols]

        annotated_con_cols = [col for col in annotated_cols if "confidence" in col]
        annotated_sample_cons = sample[annotated_con_cols]
        cons_below_threshold = annotated_sample_cons[
            annotated_sample_cons < con_threshold
        ]
        # If labels are not equal, then remove from high agreement
        if len(annotated_sample_labels.unique()) != 1:
            low_agreement_indices.append(i)
            continue
        # If at least one confidence below the threshold, then remove from
        # high agreement
        elif len(cons_below_threshold) > 0:
            low_agreement_indices.append(i)
            continue

    # Remove non-high agreement samples.
    high_agreement_samples = high_agreement_samples.drop(low_agreement_indices)

    # Getting all non-high agreement samples.
    # merged_samples = pd.merge(samples, high_agreement_samples, how="outer", indicator=True)
    # other_samples = merged_samples.loc[merged_samples._merge == "left_only"].drop("_merge", axis=1)
    row_indices_in_sample = [i for i in range(len(samples.index))]
    high_agreement_indices = [
        i for i in row_indices_in_sample if i not in low_agreement_indices
    ]

    other_samples = other_samples.drop(high_agreement_indices)

    return (high_agreement_samples, other_samples)


def gen_from_file(sample_path, con_threshold, annotator_cols):
    """
    Function that loads the samples from a file and then calls gen_from_dataset on them.

    Params:
        - sample_path (String): a file path to the csv containing the annotated data.
        - con_threshold (Int): the confidence threshold for an annotation to be considered as
            possibly high agreement.
        - annotator_cols (List): a list of the names of the columns containing the annotations.

    Returns:
        - tuple (pd.dataframe, pd.dataframe) - a tuple containing the high annotator
        agreement data and the non-high annotator agreement data, both as pandas
        dataframes.
    """
    samples = pd.read_csv(sample_path)
    return gen_from_dataset(samples, con_threshold, annotator_cols)


# TODO: can maybe adapt this
if __name__ == "__main__":
    # Thresholds used to filter 'high agreemnet' samples.
    con_threshold = 3

    # List of columns that COULD contain annotations.
    col_format = ["user_$_label", "user_$_confidence"]
    annotator_cols = []
    for i in range(1, 7):
        for col_name in col_format:
            annotator_cols.append(col_name.replace("$", str(i)))

    sample_path = "../../data/UkraineMisinfoWithReannotation.csv"
    high_agreement_samples, other_samples = gen_from_file(
        sample_path, con_threshold, annotator_cols
    )

    # Evaluting results.
    print_ratios(high_agreement_samples, annotator_cols)

    # Saving as seperate csvs.
    save_path = "../../data"
    high_agreement_samples.to_csv(f"{save_path}/HighAgreement.csv")
    other_samples.to_csv(f"{save_path}/NonHighAgreement.csv")
