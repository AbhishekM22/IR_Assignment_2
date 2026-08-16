# Information Retrieval & News Recommender System

**BITS Pilani WILP – M.Tech Data Science and Engineering**  
**Course:** Information Retrieval (Assignment 2)  
**Group:** 11  

---

## 📌 Project Overview
This repository contains an end-to-end Information Retrieval (IR) and News Recommender System built on an active corpus derived from the **AG News Dataset**. Deployed as a Streamlit Web Application within the BITS Virtual Lab environment, the system integrates:

1. **Web Crawling & Extraction:** Ingests document records, enforces cryptographic MD5 hashing to deduplicate raw payloads, and isolates metadata schema (`doc_id`, `category`, `url`, `title`).
2. **Text Preprocessing & Inverted Indexing:** Performs tokenization, lowercasing, stop-word filtering, and Porter stemming. Builds an inverted index mapping **9,878 unique terms** across **1,400 indexed documents**.
3. **Multi-Engine Ranked Search:** Dual relevance scoring combining probabilistic **Okapi BM25** and **TF-IDF** vector space models, integrated with graph-based **PageRank** centrality visualization.
4. **Content-Based Recommendation:** Computes item-to-item TF-IDF cosine similarity matrices to surface relevant articles without user history.
5. **System Telemetry & Evaluation:** Calculates real-time IR effectiveness metrics including Precision@K, Recall@K, F1@K, MAP, MRR, and NDCG@10[cite: 2].

---

## 🏗️ Architecture & Pipeline Flow
[ Crawling & Extraction ]
│
▼
[ Text Mining & Preprocessing ]  (Normalizes tokens, drops noise)
│
▼
[ Inverted Indexing ]           (Builds sub-linear lookup postings)
│
▼
[ Search & PageRank ]            (Evaluates BM25 relevance & authority)
│
▼
[ Content Recommendation ]     (Expands candidate discovery via TF-IDF)

---

## 🚀 Key Performance Benchmarks

* **Indexing Throughput:** ~1,200 documents/second[cite: 2]
* **Average Query Latency:** ~14.2 ms[cite: 2]
* **Memory Footprint:** ~142 MB RAM[cite: 2]
* **Retrieval Scores:** MAP = 1.0000 | MRR = 1.0000 | NDCG@10 = 1.0000 | F1@10 = 0.8889[cite: 2]

---

## 🛠️ Installation & Local Setup

### Prerequisites
* Python 3.9 or higher
* `pip` package manager

### 1. Clone Repository & Install Dependencies
```bash
git clone [https://github.com/your-username/ir-news-recommender.git](https://github.com/your-username/ir-news-recommender.git)
cd ir-news-recommender
pip install -r requirements.txt