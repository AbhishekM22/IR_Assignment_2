# Information Retrieval & News Recommender System

**BITS Pilani WILP – M.Tech Data Science and Engineering**  
**Course:** Information Retrieval (Assignment 2)  
**Group:** 11

---

## 📌 Project Brief
This project implements an end-to-end Information Retrieval (IR) and Content-Based News Recommendation System deployed via a Streamlit web interface in the BITS Virtual Lab environment[cite: 2]. Built on the **AG News Dataset** (1,400 indexed documents across 4 categories)[cite: 2], the system processes raw web feeds, performs MD5 deduplication, constructs an inverted index over a 9,878-term vocabulary, and evaluates candidate search results using Okapi BM25, TF-IDF, and graph-based PageRank authority scoring[cite: 2].

---

## 🚀 Quick Start Guide

### 1. Prerequisites
* Python 3.9 or higher
* `pip` package manager

### 2. Install Dependencies
Run the following command in your terminal to install all required libraries:

```bash
pip install -r requirements.txt


### 3. Run the application
streamlit run app.py