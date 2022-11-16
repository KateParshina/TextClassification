import pickle

from pathlib import Path
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from dataset import DataSet, ProcessedDataSet
from sklearn.feature_extraction.text import TfidfVectorizer


STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()
TARGET = "target"
OUTPUT_FOLDER = ""


class Vectorizer:
    def __init__(self, corpus=None, w2vec_model_path="word2vec.model"):
        if Path(w2vec_model_path).exists():
            self.model = Word2Vec.load(w2vec_model_path)
        elif corpus is not None:
            self.model = Word2Vec(sentences=corpus, window=3, min_count=5, vector_size=50)
            self.model.save(w2vec_model_path)

    def vectorize_token(self, token):
        if token in self.model.wv:
            vector = self.model.wv.get_vector(token)

            return vector

    def vectorise_seq(self, sequence: list):
        vector_seq = [self.vectorize_token(t) for t in sequence]

        return [res for res in vector_seq if res is not None]


class AnnPreProcessor:
    def __init__(self):
        self.name = "ann-prp"
        self.tokenizer = None
        self.vectorizer = None
        self.vectorizer_model_path = None

    def preprocess_train_data(self, dataset: DataSet, name: str):
        self.build_vectorizer()
        self.train_vectorizer(dataset.x_train)

        vectorizer_model_path = f"{name}/{self.name}.pk"
        self.save_vectorizer(vectorizer_model_path)

        train_data = self.vectorize(dataset.x_train)
        test_data = self.vectorize(dataset.x_test)
        self.vectorizer_model_path = vectorizer_model_path

        return ProcessedDataSet(train_data, dataset.y_train,
                                test_data, dataset.y_test,
                                model_path=vectorizer_model_path)

    def build_vectorizer(self):
        vectorizer = TfidfVectorizer(stop_words="english")
        self.vectorizer = vectorizer

    def train_vectorizer(self, train_data):
        self.vectorizer.fit(train_data)

    def save_vectorizer(self, output_path: str):
        with open(output_path, 'wb') as fin:
            pickle.dump(self.vectorizer, fin)

    def load_vectorizer(self, path: str):
        with open(path, "rb") as file_pi:
            vectorizer = pickle.load(file_pi)
        self.vectorizer = vectorizer

    def vectorize(self, samples):
        return self.vectorizer.transform(samples).toarray()

    def vectorize_raw_text(self, text: str):
        vector = self.vectorize([text])[0]
        return vector

    def filter(self, samples):
        return [self.filter_single_sample(text) for text in samples]

    def filter_single_sample(self, text: str):
        tokens = word_tokenize(text.lower())
        filtered_text = " ".join(t for t in tokens if t not in STOP_WORDS)
        return filtered_text

    def get_local_models_paths(self):
        return {"vectorizer": self.vectorizer_model_path}


class DTPreprocessor:
    def __init__(self):
        self.name = "dt-prp"
        self.tokenizer = None
        self.vectorizer = None
        self.vectorizer_model_path = None

    def preprocess_train_data(self, dataset: DataSet):
        # filter text from stop words
        # create features tokenization with TFIDF
        # save tokenizer
        pass


