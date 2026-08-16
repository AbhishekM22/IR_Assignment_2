import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class TextPreprocessor:
    """Handles cleaning raw text, corpus stats, topic modeling, and building numerical TF-IDF feature matrices."""

    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=max_features, ngram_range=(1, 2)
        )

    @staticmethod
    def clean_text(text: str) -> str:
        """Standardizes and cleans text inputs."""
        if not isinstance(text, str):
            return ""
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

    # =========================================================
    # NEW MODULE EXTENSIONS FOR PREPROCESSING TAB
    # =========================================================

    @staticmethod
    def process_sample_text(text: str):
        """Pipeline Explorer helper: step-by-step text transformation."""
        tokens = text.lower().split()
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
        cleaned_words = [w for w in cleaned.split() if len(w) > 2]
        stemmed = [w[:4] if len(w) > 4 else w for w in cleaned_words]
        return tokens, " ".join(cleaned_words), " ".join(stemmed)

    @staticmethod
    def get_corpus_stats(df: pd.DataFrame):
        """Calculates dynamic corpus statistics from input DataFrame."""
        text_column = "clean_text" if "clean_text" in df.columns else "content"
        docs = df[text_column].astype(str).apply(TextPreprocessor.clean_text).tolist()
        
        num_docs = len(docs)
        all_words = " ".join(docs).split()
        total_tokens = len(all_words)
        vocab_size = len(set(all_words))
        avg_doc_len = int(total_tokens / num_docs) if num_docs > 0 else 0

        return {
            "documents": num_docs,
            "total_tokens": total_tokens,
            "vocab_size": vocab_size,
            "avg_doc_len": avg_doc_len
        }

    @staticmethod
    def extract_topics(df: pd.DataFrame, n_topics: int = 4, top_words: int = 5):
        """Performs Latent Dirichlet Allocation (LDA) for topic modeling."""
        text_column = "clean_text" if "clean_text" in df.columns else "content"
        docs = df[text_column].astype(str).apply(TextPreprocessor.clean_text)

        tf_vectorizer = CountVectorizer(stop_words="english", max_features=1000)
        tf_matrix = tf_vectorizer.fit_transform(docs)
        feature_names = tf_vectorizer.get_feature_names_out()

        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(tf_matrix)

        topics_data = []
        for topic_idx, topic in enumerate(lda.components_):
            keywords = ", ".join([feature_names[i] for i in topic.argsort()[:-top_words - 1:-1]])
            topics_data.append({
                "Topic ID": f"Topic {topic_idx + 1}",
                "Top Keywords": keywords
            })

        return pd.DataFrame(topics_data)