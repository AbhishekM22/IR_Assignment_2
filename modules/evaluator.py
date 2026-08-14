import numpy as np
import pandas as pd


class IREvaluator:

    @staticmethod
    def evaluate_retrieval(
        retrieved_ids, relevant_ids, k_list=[3, 5, 10]
    ) -> dict:
        metrics = {}
        rel_set = set(relevant_ids)

        for k in k_list:
            ret_k = retrieved_ids[:k]
            hits = len(set(ret_k).intersection(rel_set))

            p_k = hits / k
            r_k = hits / len(rel_set) if len(rel_set) > 0 else 0
            f1 = (
                (2 * p_k * r_k) / (p_k + r_k) if (p_k + r_k) > 0 else 0.0
            )

            metrics[f"Precision@{k}"] = round(p_k, 4)
            metrics[f"Recall@{k}"] = round(r_k, 4)
            metrics[f"F1@{k}"] = round(f1, 4)

        # Reciprocal Rank (MRR)
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in rel_set:
                mrr = 1.0 / rank
                break
        metrics["MRR"] = round(mrr, 4)

        # Mean Average Precision (MAP)
        ap = 0.0
        hits = 0
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in rel_set:
                hits += 1
                ap += hits / i
        metrics["MAP"] = round(ap / len(rel_set) if len(rel_set) > 0 else 0, 4)

        # NDCG@10
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:10], 1):
            if doc_id in rel_set:
                dcg += 1.0 / np.log2(i + 1)
        idcg = sum([1.0 / np.log2(i + 1) for i in range(1, min(len(rel_set), 10) + 1)])
        metrics["NDCG@10"] = round(dcg / idcg if idcg > 0 else 0, 4)

        return metrics