from collections import defaultdict
import pandas as pd


class InvertedIndexer:

    def __init__(self):
        self.index = defaultdict(list)
        self.metadata = {}

    def build_index(self, df: pd.DataFrame):
        self.index.clear()
        self.metadata.clear()

        for idx, row in df.iterrows():
            doc_id = row["doc_id"]
            # Save metadata separately
            self.metadata[doc_id] = {
                "title": row["title"],
                "category": row["category"],
                "url": row["url"],
                "length": len(row["clean_content"].split()),
            }

            # Inverted Index creation
            terms = set(row["clean_content"].split())
            for term in terms:
                self.index[term].append(doc_id)

        return len(self.index), len(self.metadata)