import argparse
import sys


def parse_args():
    """
    Get relevant arguments for configuring settings
    for training the model.


    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Whether to only display dataset statistics.",
    )
    parser.add_argument(
        "--annotator_graph",
        action="store_true",
        help="Wehther to only display annotator graph.",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="File path to the config yaml"
    )
    parser.add_argument(
        "--calibrate_confidence",
        action="store_true",
        help="Whether to adjust the users given confidence",
    )
    parser.add_argument(
        "--calculate_reliability",
        action="store_true",
        help="Whether to calculate annotator reliability",
    )
    parser.add_argument(
        "--keep_unrepresentative",
        action="store_true",
        help="Whether to keep unrepresentative samples (where the triple annotation does not represent the full claim)",
    )
    parser.add_argument(
        "--claim_col",
        type=str,
        help="Column to take the claim from - either 'claim_title' or 'claim_triple'.",
    )
    parser.add_argument(
        "--train_label_method",
        type=str,
        help="Which label method to use 'hard_label' or 'soft_label'.",
    )
    parser.add_argument(
        "--inter_weighting",
        type=float,
        default=None,
        help="Weight for inter-annotator agreement (requires --calculate_reliability=True)",
    )
    parser.add_argument(
        "--intra_weighting",
        type=float,
        default=None,
        help="Weight for intra-annotator agreement (requires --calculate_reliability=True)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Dataset fold (between 0 and 4)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for training and evaluation results",
    )

    args = parser.parse_args()

    return args


def log_args(args):
    print("Training Configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")


def merge_config(config, args):
    if args.claim_col:
        config["claim_col"] = args.claim_col
    if args.train_label_method:
        config["train_label_method"] = args.train_label_method
    if args.inter_weighting is not None:
        config["inter_weighting"] = args.inter_weighting
    if args.intra_weighting is not None:
        config["intra_weighting"] = args.intra_weighting
    if args.keep_unrepresentative:
        config["drop_unrepresentative"] = False
    if args.output_dir:
        config["output_dir_name"] = args.output_dir
    if args.calibrate_confidence:
        config["calibrate_confidence"] = True
    if args.fold:
        config["fold"] = args.fold
    if args.calculate_reliability:
        config["calculate_reliability"] = args.calculate_reliability
    config["stats"] = args.stats
    config["annotator_graph"] = args.annotator_graph

    return config
