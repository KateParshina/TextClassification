import numpy as np
import tensorflow as tf
import environment as envr

from evaluation import Evaluator
from dataset import ProcessedDataSet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


TF_LOCAL_MODEL_PATH = "my_model.h5"
TF_CHECKPOINT_PATH = "training_1/cp.ckpt"
TF_TRAIN_HISTORY_PATH = "/trainHistoryDict"


class ANNClassifier:
    """
    ANNClassifier describes classification model based on ANN
    To define classification model use build_model() or load_model() functions
    """
    def __init__(self):
        self.name = "ann-classifier"
        self.model = None
        self.history = None

        self.history_path = TF_TRAIN_HISTORY_PATH
        self.modelH5_path = TF_LOCAL_MODEL_PATH
        self.checkpoint_path = TF_CHECKPOINT_PATH

    def train(self, X_train, Y_train, X_test=None, Y_test=None, save=False):
        """
        Trains model with provided train dataset
        :param X_train: samples of train data
        :param Y_train: labels of train samples
        :param X_test: samples of validation data
        :param Y_test: labels of validation samples
        :param save: Bool value. If True, training process details will be saved to the local storage
        """
        validation_data = None
        if X_test is not None and Y_test is not None:
            validation_data = (X_test, Y_test)

        callback = None
        if save:
            # TensorBoard call command: tensorboard --logdir self.checkpoint_path
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.checkpoint_path)
            callback = [tensorboard_callback]

        train_history = self.model.fit(X_train, Y_train,
                                       validation_data=validation_data,
                                       epochs=50,
                                       batch_size=254,
                                       callbacks=callback)
        self.history = train_history

        if save:
            self.save(output_path=self.modelH5_path)
            self.save_history(train_history, output_path=self.history_path)

    def load_model(self, model_path=None):
        """
        Loads local H5 file and identifies instance's model
        :param model_path: model local path
        """
        if not model_path:
            model_path = self.modelH5_path
        loaded_model = tf.keras.models.load_model(model_path)
        self.model = loaded_model

    def predict(self, samples_to_predict):
        """
        Predicts model outputs
        :param samples_to_predict: list of samples to predict
        :return: list of predictions for each sample
        """
        return self.model.predict(samples_to_predict)

    def predict_single_sample(self, sample):
        """
        Predicts single sample model output
        :param sample: sample of test data
        :return: sparse categorical classes for sample
        """
        return self.predict([sample])[0]

    @staticmethod
    def get_class(prediction):
        """
        Converts sparse categorical matrix to class value
        :param prediction: list of predicted classes values
        :return: index of class with the highest probability
        """
        return np.argmax(prediction)

    def save(self, output_path: str):
        """
        Saves pretrained model to local storage
        :param output_path: local path to save model
        """
        self.model.save(output_path)

    def build_model(self, classes_number=5, units=10, input_dimension=10):
        """
        Builds, identifies, compiles model architecture
        :param classes_number:
        :param units:
        :param input_dimension:
        :return:
        """
        i = tf.keras.layers.Input(shape=input_dimension)
        x = tf.keras.layers.Dense(units, activation="relu")(i)
        x = tf.keras.layers.Dense(classes_number)(x)

        model = tf.keras.models.Model(i, x)
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=["accuracy"]
        )

        self.model = model

    @staticmethod
    def save_history(history: tf.keras.callbacks.History, output_path: str):
        """
        Saves dict history of train process
        :param history: trained history from model.fit()
        :param output_path: path to save history locally
        """
        envr.create_pkl_file(data=history.history, output_path=output_path)

    def load_history(self, history_path=None):
        """
        Loads locally saved train dict history and assigns to instance history value
        :param history_path: path to local file
        """
        if not history_path:
            history_path = self.history_path

        self.history = envr.load_pkl_file(history_path)

    def run_preprocessed_data(self, data: ProcessedDataSet, name: str):
        K = len(set(data.y_train))
        M = 300
        D = data.x_train.shape[1]

        self.checkpoint_path = f"{name}/{self.name}-checkpoints"
        self.history_path = f"{name}/{self.name}-trainHistoryDict"
        self.modelH5_path = f"{name}/{self.name}-modelH5.h5"

        self.build_model(classes_number=K, units=M, input_dimension=D)
        self.train(data.x_train, data.y_train, data.x_test, data.y_test, save=True)

    def evaluate_train_process(self, data: ProcessedDataSet):
        # PREDICT DATA WITH X_TEST
        predictions = self.predict(data.x_test)
        predictions = [self.get_class(p) for p in predictions]

        # CREATE CONFUSION MATRIX, ACCURACY, F1 SCORE, RECALL, PRECISION, ROC, AUC
        evaluation = Evaluator(predictions, data.y_test)
        metrics = evaluation.get_metrics()

        # SAVE IMAGES, RETURN VALUES TO SAVE
        return metrics

    def get_local_models_paths(self):
        return {"history": self.history_path,
                "checkpoints": self.checkpoint_path,
                "model": self.modelH5_path}


class CNNClassifier:
    pass


class DTClassifier:
    def __init__(self):
        self.model = None
        self.name = "dt-classifier"
        self.model_path = ""

    def build_model(self):
        self.model = DecisionTreeClassifier()

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def save(self, output_path: str):
        envr.create_pkl_file(self.model, output_path=output_path)

    def load_model(self, model_path):
        self.model = envr.load_pkl_file(model_path)

    def predict(self, samples_to_predict):
        return self.model.predict(samples_to_predict)

    def predict_single_sample(self, sample):
        """
        Predicts single sample model output
        :param sample: sample of test data
        :return: sparse categorical classes for sample
        """
        return self.predict([sample])[0]

    def run_preprocessed_data(self, data: ProcessedDataSet, name: str):
        self.build_model()
        self.train(x_train=data.x_train, y_train=data.y_train)

        self.model_path = f"{name}/{self.name}-model"
        self.save(self.model_path)

    def get_local_models_paths(self):
        return {"model": self.model_path}

    def evaluate_train_process(self, data: ProcessedDataSet):
        # PREDICT DATA WITH X_TEST
        predictions = self.predict(data.x_test)

        # CREATE CONFUSION MATRIX, ACCURACY, F1 SCORE, RECALL, PRECISION, ROC, AUC
        evaluation = Evaluator(predictions, data.y_test)
        metrics = evaluation.get_metrics()

        # SAVE IMAGES, RETURN VALUES TO SAVE
        return metrics


class RFClassifier:
    def __init__(self):
        self.model = None
        self.name = "rf-classifier"
        self.model_path = ""

    def build_model(self):
        self.model = RandomForestClassifier()

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def save(self, output_path: str):
        envr.create_pkl_file(self.model, output_path=output_path)

    def load_model(self, model_path):
        self.model = envr.load_pkl_file(model_path)

    def predict(self, samples_to_predict):
        return self.model.predict(samples_to_predict)

    def predict_single_sample(self, sample):
        """
        Predicts single sample model output
        :param sample: sample of test data
        :return: sparse categorical classes for sample
        """
        return self.predict([sample])[0]

    def run_preprocessed_data(self, data: ProcessedDataSet, name: str):
        self.build_model()
        self.train(x_train=data.x_train, y_train=data.y_train)

        self.model_path = f"{name}/{self.name}-model"
        self.save(self.model_path)

    def get_local_models_paths(self):
        return {"model": self.model_path}

    def evaluate_train_process(self, data: ProcessedDataSet):
        predictions = self.predict(data.x_test)

        evaluation = Evaluator(predictions, data.y_test)
        metrics = evaluation.get_metrics()

        return metrics
