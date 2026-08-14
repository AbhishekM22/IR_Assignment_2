import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class TextMiner:

    def __init__(self, max_features: int = 3000):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=max_features, ngram_range=(1, 2)
        )

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def process_corpus(self, df: pd.DataFrame):
        df["clean_content"] = df["content"].apply(self.clean_text)
        tfidf_matrix = self.vectorizer.fit_transform(df["clean_content"])
        feature_names = self.vectorizer.get_feature_names_out()
        return df, tfidf_matrix, feature_names