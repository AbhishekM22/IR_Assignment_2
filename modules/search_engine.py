import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class SearchEngine:

    def __init__(self, vectorizer, tfidf_matrix, df: pd.DataFrame):
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.df = df

    def search(self, query: str, top_k: int = 10):
        clean_q = query.lower()
        query_vec = self.vectorizer.transform([clean_q])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = scores.argsort()[::-1][:top_k]
        results = self.df.iloc[top_indices].copy()
        results["relevance_score"] = scores[top_indices]
        return results[results["relevance_score"] > 0]

    def compute_pagerank(self, num_nodes: int = 50):
        """Simulate document citation/link graph and compute PageRank."""
        sub_matrix = self.tfidf_matrix[:num_nodes]
        sim_matrix = cosine_similarity(sub_matrix, sub_matrix)

        # Build graph thresholded on similarity > 0.15
        G = nx.Graph()
        for i in range(num_nodes):
            G.add_node(self.df.iloc[i]["doc_id"])

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if sim_matrix[i][j] > 0.15:
                    G.add_edge(
                        self.df.iloc[i]["doc_id"],
                        self.df.iloc[j]["doc_id"],
                        weight=float(sim_matrix[i][j]),
                    )

        pr_scores = nx.pagerank(G)
        return pr_scores, G