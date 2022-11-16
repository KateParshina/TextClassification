import visualkeras
import pandas as pd
import seaborn as sn
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def plot_nn_model(model, output_path: str):
    visualkeras.layered_view(model, legend=True, draw_volume=True, to_file=output_path)


def plot_cm(cm, classes: list, save=True, output_path="CM.png"):
    df = pd.DataFrame(cm, index=classes, columns=classes)
    df = df.round(4)
    ax = sn.heatmap(df, annot=True, fmt="g")
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    if save:
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()


def plot_metrics(metrics_df: pd.DataFrame, output_path="metrics.jpg", save=True):
    fig = px.imshow(metrics_df,
                    labels=dict(x="Metrics", y="Classifiers", color="Score"),
                    x=metrics_df.columns,
                    y=metrics_df.transpose().columns,
                    text_auto=True,
                    range_color=[0, 1],
                    color_continuous_scale=["salmon", "palegoldenrod", "mediumseagreen"],
                    )
    fig.update_layout(
        font=dict(
            family="Courier New, monospace",
            size=26,
            color="RebeccaPurple"
        )
    )
    fig.update_traces(colorbar_bordercolor="#444", xgap=4, ygap=4)
    fig.update_xaxes(side="top")
    if save:
        fig.write_image(output_path, width=2200, height=1300)
    else:
        fig.show()


def create_metrics_df(metrics: dict):
    possible_columns = ["accuracy", "f1_score", "recall", "precision", "roc_auc"]

    df = pd.DataFrame(metrics)
    df = df.transpose()
    provided_column = df.columns
    actual_column = [c for c in possible_columns if c in provided_column]

    metrics_df = df[actual_column]

    return metrics_df.round(4)


class Evaluator:
    def __init__(self, predicted, actual):
        self.predicted = predicted
        self.actual = actual

    def confusion_matrix(self):
        return confusion_matrix(self.actual, self.predicted, normalize='true')

    def accuracy(self):
        return accuracy_score(self.actual, self.predicted, normalize=True)

    def f1_score(self):
        return f1_score(self.actual, self.predicted, average="micro")

    def recall(self):
        return recall_score(self.actual, self.predicted, average="micro")

    def precision(self):
        return precision_score(self.actual, self.predicted, average="micro")

    def roc_auc(self):
        return roc_auc_score(self.actual, self.predicted, average="micro", multi_class="ovr")

    def get_metrics(self):
        return {"accuracy": self.accuracy(),
                "f1_score": self.f1_score(),
                "recall": self.recall(),
                "precision": self.precision(),
                # "roc_auc": self.roc_auc(),
                "cm": self.confusion_matrix(),
                }