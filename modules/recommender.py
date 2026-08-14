import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class NewsRecommender:

    def __init__(self, tfidf_matrix, df: pd.DataFrame):
        self.tfidf_matrix = tfidf_matrix
        self.df = df

    def recommend(self, doc_id: str, top_k: int = 5):
        doc_idx = self.df[self.df["doc_id"] == doc_id].index[0]
        target_vec = self.tfidf_matrix[doc_idx]

        sims = cosine_similarity(target_vec, self.tfidf_matrix).flatten()
        similar_indices = sims.argsort()[::-1]
        similar_indices = [i for i in similar_indices if i != doc_idx][:top_k]

        recs = self.df.iloc[similar_indices].copy()
        recs["similarity_score"] = sims[similar_indices]
        return recs