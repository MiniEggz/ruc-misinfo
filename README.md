# RUC-MCD: Russo-Ukrainian Conflict Knowledge-Based Misinformation Classification Dataset

## Description
RUC-MCD enables misinformation classification, in the form of semantic entailment from an evidenced false claim.
This repository contains both the dataset (data/UkraineMisinfoDatasetNoTweetText.csv), which must be appended with
X posts using the X api and the tweet IDs stored in the dataset, and the classification pipeline utilising soft label training, as well as 
calibrating the confidence given by annotators and metrics for annotator agreement.

To get started, clone this repository and install the required packages with 'pip install -r requirements.txt'.

## Usage

### Configuration
To run experiments, first navigate to the root directory of this project. In this
directory, there will be a config.yaml. Modify this to change any training arguments
you may want to adjust. Key arguments include `seed`, `labels2id`, `merge_labels`,
`dataset_path`, `calculate_reliability`, `inter_weighting`, `intra_weighting`,
`train_label_method`. There are then model specific arguments for the
`AutoModelForSequenceClassification`; these arguments include:
`model_name`, `num_epochs`, `warmup_epochs`, `learning_rate`, `batch_size`.

`seed`: sets the seed for all experiments, allowing for repeatability.

`labels2id`: structured in the format: `{"label1": 0, "label2": 1, ..., "labeln": n-1}`.

`merge_labels`: this allows you to merge any of the labels, if reducing the number
                of classes is desired. In the experiments run in this study, the labels
                'debunk' and 'other' were merged into the 'other' class.

`dataset_path`: provides the path to the dataset.

`calibrate_confidence`: whether to use confidence calibration, as in Wu et al. (2023).

`calculate_reliability`: whether to calculate annotator reliability using EffiARA.

`inter_weighting`: the weighting of inter-annotator agreement in the annotator
reliability calculation.

`train_label_method`: either using `hard_label` or `soft_label`.

All model parameters are as expected.

If you would like to have a number of different configuration files, you can name
the config file as you wish and add it as a command line argument when running
any experiments. The `--config` argument will default to `config.yaml`.

The name of your config file will also contribute to your run name on WandB.

### Running experiments
To run experiments, first ensure all requirements are satisfied. Then run the command:
```
python code/main.py --config="path_to_config.yaml" fold={0-4}
```
to run a specific fold.

To adjust any arguments from the command line, use `code/main.py -h` for a summary.

For dataset statistics and no run, you can pass `--stats` to view the split of each fold
and the split of the whole dataset. If you would like to view the annotator reliability
graph at the current configuration, you can run `--annotator_graph`.

Example:
```
python code/main.py --stats
```

```
python code/main.py --config="path_to_config.yaml" --annotator_graph
```

## Results

| Model | Label Type | Confidence Calibration | Reliability | Reliability Type |   F1-Macro   |  ECE  |
|-------|------------|------------------------|-------------|------------------|--------------|-------|
| BERT  | hard       | ✗                      | ✗           | N/A              | 0.699 (0.05) | 0.151 (0.02)     |
| Llama | hard       | ✗                      | ✗           | N/A              | 0.738 (0.02) | 0.116 (0.01)     |
| BERT  | hard       | ✗                      | ✓           | inter            | 0.698 (0.05) | 0.144 (0.02)     |
| Llama | hard       | ✗                      | ✓           | inter            | 0.726 (0.04) | 0.121 (0.02)     |
| BERT  | hard       | ✗                      | ✓           | intra            | 0.690 (0.07) | 0.152 (0.03)     |
| Llama | hard       | ✗                      | ✓           | intra            | 0.751 (0.05) | 0.106 (0.02)     |
| BERT  | hard       | ✗                      | ✓           | inter+intra      | 0.677 (0.12) | 0.119 (0.03)     |
| Llama | hard       | ✗                      | ✓           | inter+intra      | 0.726 (0.07) | 0.111 (0.02)     |
| BERT  | soft       | ✗                      | ✗           | N/A              | 0.691 (0.07) | 0.071 (0.01)     |
| Llama | soft       | ✗                      | ✗           | N/A              | 0.730 (0.09) | 0.093 (0.02)     |
| BERT  | soft       | ✗                      | ✓           | inter            | 0.728 (0.04) | 0.072 (0.02)     |
| Llama | soft       | ✗                      | ✓           | inter            | 0.724 (0.06) | 0.094 (0.01)     |
| BERT  | soft       | ✗                      | ✓           | intra            | 0.722 (0.07) | 0.079 (0.01)     |
| Llama | soft       | ✗                      | ✓           | intra            | 0.732 (0.07) | 0.079 (0.01)     |
| BERT  | soft       | ✗                      | ✓           | inter+intra      | 0.740 (0.06) | 0.077 (0.02)     |
| Llama | soft       | ✗                      | ✓           | inter+intra      | 0.756 (0.07) | 0.092 (0.01)     |
| BERT  | soft       | ✓                      | ✗           | N/A              | 0.627 (0.03) | 0.116 (0.01)     |
| Llama | soft       | ✓                      | ✗           | N/A              | 0.638 (0.07) | 0.124 (0.01)     |
