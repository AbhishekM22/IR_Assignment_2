import hashlib
import pandas as pd
from datasets import load_dataset


class AGNewsCrawler:
    """Handles fetching AG News, crawling depth simulation, and deduplication."""

    def __init__(self):
        self.category_map = {
            1: "World",
            2: "Sports",
            3: "Business",
            4: "Sci/Tech",
        }

    def fetch_ag_news(self, num_samples: int = 1000) -> tuple[pd.DataFrame, int]:
        """Fetch AG News dataset with fallback mechanism and clean duplicate documents."""
        try:
            # 1. Attempt loading using full HuggingFace repository ID
            print("[+] Fetching dataset from HuggingFace...")
            dataset = load_dataset(
                "fancyzhx/ag_news", split=f"train[:{num_samples}]"
            )
            df = pd.DataFrame(dataset)

            # Convert 0-indexed labels to 1-indexed for mapping consistency
            df["label_id"] = df["label"] + 1
            df["category"] = df["label_id"].map(self.category_map)
            df["content"] = df["text"]

        except Exception as e:
            print(
                f"[!] HuggingFace fetch failed ({e}). Falling back to direct raw CSV..."
            )
            # 2. Fallback: Direct pandas fetch from raw repository mirror
            fallback_url = "https://raw.githubusercontent.com/mhjabreel/CharICL/master/data/ag_news_csv/train.csv"
            df = pd.read_csv(
                fallback_url,
                names=["label_id", "title_raw", "content"],
                nrows=num_samples,
            )
            df["category"] = df["label_id"].map(self.category_map)

        # Clean Title & Add Metadata
        df["doc_id"] = [f"DOC_{i+1:04d}" for i in range(len(df))]
        df["url"] = [
            f"https://news.org/article/{i+1}" for i in range(len(df))
        ]
        df["title"] = df["content"].apply(
            lambda x: (
                x.split(" - ")[0]
                if isinstance(x, str) and " - " in x
                else str(x)[:40] + "..."
            )
        )

        # Duplicate Handling via MD5 Hash
        df["doc_hash"] = df["content"].apply(
            lambda x: hashlib.md5(str(x).encode()).hexdigest()
        )
        initial_len = len(df)
        df = df.drop_duplicates(subset=["doc_hash"]).reset_index(drop=True)

        return df, initial_len - len(df)