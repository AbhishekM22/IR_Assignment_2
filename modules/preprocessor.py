import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class TextPreprocessor:
    """Handles cleaning raw text and building numerical TF-IDF feature matrices."""

    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=max_features, ngram_range=(1, 2)
        )

    @staticmethod
    def clean_text(text: str) -> str:
        """Standardizes and cleans text inputs."""
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Strip special characters
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def fit_transform(self, df: pd.DataFrame):
        """Cleans dataframe content and returns fitted TF-IDF matrix."""
        print("[+] Preprocessing and vectorizing news text...")
        df["clean_text"] = df["content"].apply(self.clean_text)
        tfidf_matrix = self.vectorizer.fit_transform(df["clean_text"])
        print(
            f"[✔] TF-IDF Matrix shape created: {tfidf_matrix.shape} (Articles x Features)"
        )
        return df, tfidf_matrix