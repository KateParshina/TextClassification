import pandas as pd
from pathlib import Path
import argparse
from classifiers import ANNClassifier, DTClassifier, RFClassifier
from preprocess import AnnPreProcessor
from dataset import DataSet
from environment import create_folder, create_json_file
from evaluation import create_metrics_df, plot_metrics, plot_cm, plot_nn_model

"""
Mian file to train classifiers
Actions:
1) Open dataset file, check if it is stable, generate dataset
2) Split dataset into train and test
3) preprocess test and train datasets
4) train models with train dataset
5) evaluate models with test dataset and save statistic
"""

# TODO: add logger

TARGET = "target"
parser = argparse.ArgumentParser(description='TRAIN PROCESS')
parser.add_argument('-d', metavar='-d', type=str, help='path to dataset file with text data and labels', required=True)
parser.add_argument('-lc', metavar='-lc', type=str, help='name of label column in dataset file', default="")
parser.add_argument('-fc', metavar='-fc', type=str, help='name of feature column in dataset file', default="")
args = parser.parse_args()


def load_dataset(path: str, label_column="label", feature_column="text"):
    if not Path(path).exists():
        return

    dataframe = pd.read_csv(path)

    if any(column not in dataframe.columns for column in [label_column, feature_column]):
        return

    labels = set(dataframe[label_column])
    label_target_map = {label: index for index, label in enumerate(labels)}

    dataframe[TARGET] = dataframe.apply(lambda row: label_target_map[row[label_column]], axis=1)

    dataset = DataSet(dataframe, feature_column=feature_column, label_column=label_column)

    return dataset, label_target_map


def train_process():
    dataset_local_path = args.d
    label_column_name = args.lc
    feature_column_name = args.fc

    if not (dataset_local_path and Path(dataset_local_path).exists()):
        raise ValueError(f"Path {dataset_local_path} does not exist")

    dataset, l_t_m = load_dataset(dataset_local_path, label_column=label_column_name,
                                  feature_column=feature_column_name)
    if dataset is None:
        raise ValueError("failed to load dataset")

    classes = set(dataset.y_train)
    dataset_name = Path(dataset_local_path).stem
    create_folder(dataset_name)
    classifiers = [
        (DTClassifier(), AnnPreProcessor()),
        (ANNClassifier(), AnnPreProcessor()),
        (RFClassifier(), AnnPreProcessor())
    ]

    create_json_file(l_t_m, f"{dataset_name}/label_target_map.json")

    train_results = {}
    comparing_metrics = {}
    for classifier, preprocessor in classifiers:
        processed_data = preprocessor.preprocess_train_data(dataset, dataset_name)
        classifier.run_preprocessed_data(data=processed_data, name=dataset_name)

        train_results[classifier.name] = {"cls": classifier.get_local_models_paths(),
                                          "prp": preprocessor.get_local_models_paths()}

        metrics = classifier.evaluate_train_process(processed_data)
        comparing_metrics[classifier.name] = metrics

        plot_cm(classes=list(classes), save=True, output_path=f"{dataset_name}/CM-{classifier.name}.jpg",
                cm=metrics["cm"])
        # plot_nn_model(classifier.model, f"{dataset_name}/model-{classifier.name}.png")

    comparing_df = create_metrics_df(comparing_metrics)

    plot_metrics(comparing_df, output_path=f"{dataset_name}/evaluation.jpg", save=True)

    create_json_file(train_results, f"{dataset_name}/train-results.json")


if __name__ == '__main__':
    train_process()
