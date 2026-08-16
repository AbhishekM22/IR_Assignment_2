import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


class DocumentClassifier:
    """TF-IDF + Linear SVM document classification pipeline."""

    def __init__(self):
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("svm", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42)))
        ])
        self.is_trained = False
        self._initialize_dummy_train()

    def _initialize_dummy_train(self):
        """Bootstrap classifier with default sample classes so UI works out-of-the-box."""
        sample_texts = [
            "information retrieval search engine tf-idf bm25 indexing",
            "machine learning deep neural networks classification svm",
            "python software engineering backend web development api",
            "data science statistics evaluation metrics precision recall"
        ]
        sample_labels = ["IR & Search", "Machine Learning", "Software Dev", "Data Science"]
        self.train(sample_texts, sample_labels)

    def train(self, texts, labels):
        """Trains the Linear SVM model on labeled data."""
        self.model.fit(texts, labels)
        self.is_trained = True

    def predict(self, text: str):
        """Predicts class label and probability scores for input text."""
        if not self.is_trained:
            raise ValueError("Classifier model is not trained yet.")

        predicted_class = self.model.predict([text])[0]
        probs = self.model.predict_proba([text])[0]
        classes = self.model.classes_

        score_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}
        return predicted_class, score_dict