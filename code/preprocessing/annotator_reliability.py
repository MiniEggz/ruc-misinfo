import re

import krippendorff
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

try:
    from preprocessing.confidence_calibration import run_conf_cal
    from preprocessing.soft_label_utils import (
        add_soft_labels,
        generate_final_soft_labels,
        generate_hard_labels,
    )
except ImportError:
    # if running from a script
    from confidence_calibration import run_conf_cal
    from soft_label_utils import (
        add_soft_labels,
        generate_final_soft_labels,
        generate_hard_labels,
    )

DATASET_PATH = "../../data/UkraineMisinfoWithReannotation.csv"


def check_user_format(user_x):
    """Check that user is in the correct format of
    the string user_{some_number} or re_user_{some_number}.
    """
    return bool(re.match(r"(re_)?user_\d+", user_x))


def pairwise_nominal_krippendorff_agreement(
    pair_df, heading_1, heading_2, label_mapping  # TODO: test this works
):
    """Get the nominal annotator agreement, given two headings for each
       annotator column containing their primary label for each sample.

    Args:
        pair_df (pd.DataFrame): dataframe filtered to contain only the samples
                                that allow agreement calculations.
        heading_1 (str): heading of the first column required to calculate agreement.
        heading_2 (str): heading of the second column required to calculate agreement.
        label_mapping (dict): mapping of labels to numeric values.

    Returns:
        float: level of agreement. TODO: figure out how to best normalise for next calculations.
    """
    if pair_df[heading_1].isna().any() or pair_df[heading_2].isna().any():
        raise ValueError(
            "One or both of the columns given contain NaN values; the column names may be incorrect or there is an issue with the data."
        )

    # convert string categories to numeric values
    pair_df[heading_1 + "_numeric"] = pair_df[heading_1].map(label_mapping)
    pair_df[heading_2 + "_numeric"] = pair_df[heading_2].map(label_mapping)

    # turn the pair dataframe into 2d array for Krippendorff calculation
    krippendorff_format_data = (
        pair_df[[heading_1 + "_numeric", heading_2 + "_numeric"]].to_numpy().T
    )
    return krippendorff.alpha(
        reliability_data=krippendorff_format_data, level_of_measurement="nominal"
    )


# TODO: this will need generalising - won't always be 3d if more / less labels
def is_soft_label(x, num_classes=3):
    """Check that a value is a soft label (nd vector).

    Args:
        x: some value.
        num_classes: num classes for n-d vector.

    Returns:
        bool: whether x is a nd numpy vector.
    """
    return isinstance(x, np.ndarray) and x.shape == (num_classes,)


def headings_contain_soft_labels(df, heading_1, heading_2, num_classes=3):
    """Check that the given headings contain nothing but soft labels.

    Args:
        df (pd.DataFrame): the dataframe containing data to be worked with.
        heading_1 (str): first heading containing soft labels.
        heading_2 (str): second heading containing soft labels.

    Returns:
        bool: whether headings contain only soft labels.
    """
    checks = df[[heading_1, heading_2]].applymap(
        lambda x: is_soft_label(x, num_classes)
    )
    return checks.all(axis=None)


def cosine_similarity(vector_a, vector_b):
    """Calculate the cosine similarity between two vectors.

    Args:
        vector_a (np.ndarray)
        vector_b (np.ndarray)

    Returns:
        float: cosine similarity between the two vectors.
    """
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


def pairwise_cosine_similarity(pair_df, heading_1, heading_2, num_classes=3):
    """Calculate the pairwise cosine similarity between two columns of soft labels.

    Args:
        pair_df (pd.DataFrame): data frame containing annotation data.
        heading_1 (str): heading of first column containing soft labels.
        heading_2 (str): heading of second column containing soft labels.

    Returns:
        float: average cosine similarity between the two sets of soft labels.
    """
    if pair_df[heading_1].isna().any() or pair_df[heading_2].isna().any():
        raise ValueError(
            "One or both of the columns given contain NaN values; the column names may be incorrect or there is an issue with the data."
        )

    if not headings_contain_soft_labels(pair_df, heading_1, heading_2, num_classes):
        raise Exception(
            "There has been an issue in generating the soft labels in the dataset."
        )

    cosine_similarities = pair_df.apply(
        lambda row: cosine_similarity(row[heading_1], row[heading_2]), axis=1
    )
    return np.sum(cosine_similarities) / len(cosine_similarities)


def retrieve_pair_annotations(df, user_x, user_y):
    """Get the subset of the dataframe annotated by users
       x and y.

    Args:
        df (pd.DataFrame): the whole dataset.
        user_x (str): name of the user in the form user_x.
        user_x (str): name of the user in the form user_y.

    Returns:
        pd.DataFrame: copy of the reduced subset containing
                      only samples annotated by both users.
    """
    if not check_user_format(user_x):
        raise ValueError(
            "User x parameters must be in the form user_x or re_user_x, where x is some number."
        )
    if not check_user_format(user_y):
        raise ValueError(
            "User y parameters must be in the form user_y or re_user_y, where y is some number."
        )

    return df[df[f"{user_x}_label"].notna() & df[f"{user_y}_label"].notna()].copy()


def pairwise_agreement(
    df, user_x, user_y, label_mapping, metric="krippendorff", num_classes=3
):
    """Get the pairwise annotator agreement given the full dataframe.

    Args:
        df (pd.DataFrame): full dataframe containing the whole dataset.
        user_x (str): name of the user in the form user_x.
        user_y (str): name of the user in the form user_y.
        metric (str): agreement metric to use for inter-/intra-annotator agreement:
                      * krippendorff: the nominal krippendorff similarity metric on
                                      hard labels only.
                      * cosine: the cosine similarity metric to be used on soft labels.

    Returns:
        float: agreement between user_x and user_y.
    """
    pair_df = retrieve_pair_annotations(df, user_x, user_y)
    if metric == "krippendorff":
        return pairwise_nominal_krippendorff_agreement(
            pair_df, user_x + "_label", user_y + "_label", label_mapping
        )
    elif metric == "cosine":
        return pairwise_cosine_similarity(
            pair_df,
            user_x + "_soft_label",
            user_y + "_soft_label",
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"The metric {metric} was not recognised.")


class Annotations:
    """Class to hold all annotation information for the EffiARA annotation
    framework. Methods include inter- and intra- annotator agreement
    calculations, as well the overall reliability calculation and other
    utilities.
    """

    def __init__(
        self,
        dataset_path,
        num_annotators,
        label_mapping,
        calibrate_confidence=False,
        agreement_metric="krippendorff",
        drop_unrepresentative=False,
        merge_labels=None,
    ):
        # set instance variables
        self.num_annotators = num_annotators
        self.label_mapping = label_mapping
        self.num_classes = len(label_mapping)
        self.calibrate_confidence = calibrate_confidence
        self.agreement_metric = agreement_metric
        self.drop_unrepresentative = drop_unrepresentative
        self.merge_labels = merge_labels

        # load dataset
        self.df = pd.read_csv(dataset_path)
        if self.drop_unrepresentative:
            self.df = self.df[self.df["representative"] == "y"]
            self.df.reset_index(drop=True, inplace=True)

        # TODO: add checks on dataset?

        # merge labels
        self.replace_labels()

        # generate user soft labels
        self.df = add_soft_labels(
            self.df, self.num_annotators, self.label_mapping, self.num_classes
        )

        # calculate intra annotator (independent of confidence calibration)
        self.G = self.init_annotator_graph()
        self.calculate_intra_annotator_agreement()

        if self.calibrate_confidence:
            self.df = run_conf_cal(self.df)
            # need to regenerate soft labels using calibrated confidence
            self.df = add_soft_labels(
                self.df,
                self.num_annotators,
                self.label_mapping,
                self.num_classes,
                calibrate_confidence=True,
            )

        # calculate inter_annotator agreement
        self.calculate_inter_annotator_agreement()

    def replace_labels(self):
        """Merge labels. Uses find and replace so do not switch labels e.g.
        {"misinfo": ["debunk"], "debunk": ["misinfo", "other"]}.
        """
        if not self.merge_labels:
            return

        # TODO: add type checks?
        for replacement, to_replace in self.merge_labels.items():
            for label in to_replace:
                for i in range(1, self.num_annotators + 1):
                    # each label col
                    label_col = f"user_{i}_label"
                    re_label_col = "re_" + label_col
                    secondary_col = f"user_{i}_secondary"
                    re_secondary_col = "re_" + secondary_col

                    # find and replace in each col
                    self.df[label_col] = self.df[label_col].replace(label, replacement)
                    self.df[secondary_col] = self.df[secondary_col].replace(
                        label, replacement
                    )
                    self.df[re_label_col] = self.df[re_label_col].replace(
                        label, replacement
                    )
                    self.df[re_secondary_col] = self.df[re_secondary_col].replace(
                        label, replacement
                    )

    def generate_final_labels_and_sample_weights(self):
        """Generate the final labels and sample weights for the dataframe."""
        self.df = generate_final_soft_labels(
            self.df, self.get_reliability_dict(), self.num_annotators
        )

        self.df = generate_hard_labels(
            self.df, self.get_reliability_dict(), self.num_annotators
        )

    def init_annotator_graph(self):
        """Initialise the annotator graph with an initial reliability of 1.
        This means each annotator will initially be weighted equally.
        """
        G = nx.Graph()
        for i in range(1, self.num_annotators + 1):
            G.add_node(f"user_{i}", reliability=1)
        return G

    def normalise_edge_property(self, property):
        """Normalise an edge property to have a mean of 1.

        Args:
            property (str): the name of the edge property to normalise.
        """
        total = sum(edge[property] for _, _, edge in self.G.edges(data=True))
        num_edges = self.G.number_of_edges()

        avg = total / num_edges
        if avg < 0:
            raise ValueError(
                "Mean value must be greater than zero, high agreement/reliability will become low and vice versa."
            )

        for _, _, edge in self.G.edges(data=True):
            edge[property] /= avg

    def normalise_node_property(self, property):
        """Normalise a node property to have a mean of 1.

        Args:
            property (str): the name of the node property to normalise.
        """
        total = sum(node[property] for _, node in self.G.nodes(data=True))
        num_nodes = self.G.number_of_nodes()

        avg = total / num_nodes
        if avg < 0:
            raise ValueError(
                "Mean value must be greater than zero, high agreement/reliability will become low and vice versa."
            )

        for node in self.G.nodes():
            self.G.nodes[node][property] /= avg

    def calculate_inter_annotator_agreement(self):
        """Calculate the inter-annotator agreement between each
        pair of annotators. Each agreement value will be
        represented on the edges of the graph between nodes
        that are representative of each annotator.
        """
        inter_annotator_agreement_scores = {}
        for i in range(self.num_annotators):
            # get the current annotators and 2 linked
            current_annotator = f"user_{i+1}"
            link_1_annotator = f"user_{(i+1) % 6 + 1}"
            link_2_annotator = f"user_{(i+2) % 6 + 1}"

            # update inter annotator agreement scores
            inter_annotator_agreement_scores[(current_annotator, link_1_annotator)] = (
                pairwise_agreement(
                    self.df,
                    current_annotator,
                    link_1_annotator,
                    self.label_mapping,
                    metric=self.agreement_metric,
                    num_classes=self.num_classes,
                )
            )
            inter_annotator_agreement_scores[(current_annotator, link_2_annotator)] = (
                pairwise_agreement(
                    self.df,
                    current_annotator,
                    link_2_annotator,
                    self.label_mapping,
                    metric=self.agreement_metric,
                    num_classes=self.num_classes,
                )
            )

            # add all agreement scores to the graph
            for users, score in inter_annotator_agreement_scores.items():
                self.G.add_edge(users[0], users[1], agreement=score)

    def calculate_intra_annotator_agreement(self):
        """Calculate intra-annotator agreement."""
        for i in range(self.num_annotators):
            user_name = f"user_{i+1}"
            re_user_name = f"re_user_{i+1}"
            self.G.nodes[user_name]["intra_agreement"] = pairwise_agreement(
                self.df,
                user_name,
                re_user_name,
                self.label_mapping,
                metric=self.agreement_metric,
                num_classes=self.num_classes,
            )

    def calculate_avg_inter_annotator_agreement(self):
        """Calculate each annotator's average agreement using
        using a weighted average from the annotators around
        them. The average is weighted by the overall reliability
        score of each annotator.
        """
        for node in self.G.nodes():
            edges = self.G.edges(node, data=True)
            # get weighted avg agreement
            weighted_agreement_sum = 0
            weights_sum = 0
            for _, target, edge in edges:
                weight = self.G.nodes[target]["reliability"]
                weights_sum += weight
                weighted_agreement_sum += weight * edge["agreement"]
            self.G.nodes[node]["avg_inter_agreement"] = (
                weighted_agreement_sum / weights_sum if weights_sum else 0
            )

    def calculate_annotator_reliability(self, alpha=0.5, beta=0.5, epsilon=0.001):
        """Recursively calculate annotator reliability, using
           intra-annotator agreement, inter-annotator agreement,
           or a mixture, controlled by the alpha and beta parameters.
           Alpha and Beta must sum to 1.0.

        Args:
            alpha (float): value between 0 and 1, controlling weight of intra-annotator agreement.
            beta (float): value between 0 and 1, controlling weight of inter-annotator agreement.
            epsilon (float): controls the maximum change from the last iteration to indicate convergence.
        """
        if alpha + beta != 1:
            raise ValueError("Alpha and Beta must sum to 1.0.")

        if alpha < 0 or alpha > 1 or beta < 0 or beta > 1:
            raise ValueError("Alpha and beta values must be between 0 and 1.")

        # keep updating until convergence
        max_change = np.inf
        while abs(max_change) > epsilon:
            print("Running iteration.")
            previous_reliabilties = {
                node: data["reliability"] for node, data in self.G.nodes(data=True)
            }

            # calculate the new inter annotator agreement scores
            self.calculate_avg_inter_annotator_agreement()

            # update reliability
            for _, node in self.G.nodes(data=True):
                node["reliability"] = (
                    alpha * node["intra_agreement"] + beta * node["avg_inter_agreement"]
                )
            self.normalise_node_property("reliability")

            # find largest change as a marker
            max_change = max(
                [
                    abs(self.G.nodes[node]["reliability"] - previous_reliabilties[node])
                    for node in self.G.nodes()
                ]
            )

    def get_user_reliability(self, username):
        """Get the reliability of a given annotator.

        Args:
            username (str): username of the annotator.

        Returns:
            float: reliability score of the annotator.
        """
        return self.G.nodes[username]["reliability"]

    def get_reliability_dict(self):
        """Get a dictionary of reliability scores per username.

        Returns:
            dict: dictionary of key=username, value=reliability.
        """
        return {node: self.G.nodes[node]["reliability"] for node in self.G.nodes()}

    def display_annotator_graph(self):
        """Display the annotation graph."""
        plt.figure(figsize=(12, 12))
        pos = nx.circular_layout(self.G, scale=0.9)

        node_size = 3000
        nx.draw_networkx_nodes(self.G, pos, node_size=node_size)
        nx.draw_networkx_edges(self.G, pos)
        labels = {node: node[-1] for node in self.G.nodes()}
        nx.draw_networkx_labels(
            self.G, pos, labels=labels, font_color="white", font_size=24
        )

        # add inter-annotator agreement to edges
        edge_labels = {
            (u, v): f"{d['agreement']:.3f}" for u, v, d in self.G.edges(data=True)
        }
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=24)

        # adjust text pos for intra-annotator agreement
        for node, (x, y) in pos.items():
            if x == 0:
                align = "center"
                if y > 0:
                    y_offset = 0.15
                else:
                    y_offset = -0.15
            elif y == 0:
                align = "center"
                y_offset = 0 if x > 0 else -0.15
            elif x > 0:
                align = "left"
                y_offset = 0.15 if y > 0 else -0.15
            else:
                align = "right"
                y_offset = 0.15 if y > 0 else -0.15

            plt.text(
                x,
                y + y_offset,
                s=f"{self.G.nodes[node]['intra_agreement']:.3f}",
                horizontalalignment=align,
                verticalalignment="center",
                fontdict={"color": "black", "size": 24},
            )

        # legend for reliability
        reliability_scores = {
            node: data["reliability"] for node, data in self.G.nodes(data=True)
        }
        reliability_text = "Reliability:\n\n" + "\n".join(
            [f"{node}: {score:.3f}" for node, score in reliability_scores.items()]
        )
        plt.text(
            0.05,
            0.95,
            reliability_text,
            transform=plt.gca().transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=12,
            color="black",
        )

        # plot
        plt.axis("off")
        plt.show()

    def __str__(self):
        return_string = ""
        for node, attrs in self.G.nodes(data=True):
            return_string += f"Node {node} has the following attributes:\n"
            for attr, value in attrs.items():
                return_string += f"  {attr}: {value}\n"
            return_string += "\n"
        return return_string
