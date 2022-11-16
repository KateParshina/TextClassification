import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.model_selection import train_test_split


TARGET = "target"


class DataSet:
    def __init__(self, data: pd.DataFrame, feature_column: str, label_column: str):
        self.data = data
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.target = TARGET
        self.feature = feature_column
        self.label = label_column
        self.label_target_map = {}
        self.split()

    def split(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.data[self.feature], self.data[self.target],
                                                            test_size=0.33, random_state=42)
        self.x_train = X_train
        self.y_train = Y_train
        self.x_test = X_test
        self.y_test = Y_test

        labels_map = {row[TARGET]: row[self.label] for _, row in self.data.iterrows()}
        self.label_target_map = labels_map


@dataclass
class ProcessedDataSet:
    x_train: np.array
    y_train: np.array
    x_test: np.array
    y_test: np.array
    model_path: str
