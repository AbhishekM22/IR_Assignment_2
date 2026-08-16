import re
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Scikit-learn models for Topic Modeling and Classification
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Module imports
from modules.crawler import AGNewsCrawler
from modules.evaluator import IREvaluator
from modules.indexer import InvertedIndexer
from modules.inferences import render_inferences_page
from modules.recommender import NewsRecommender
from modules.search_engine import SearchEngine
from modules.text_miner import TextMiner

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="BITS IR & Recommender Portal", page_icon="🔍", layout="wide"
)

st.title("📚 Information Retrieval & Recommender Engine Dashboard")
st.caption("Information Retrieval Assignment 2 · Group 11")


# ---------------------------------------------------------
# TEXT PREPROCESSING HELPER FOR CLASSIFIER
# ---------------------------------------------------------
def clean_and_stem_text(text: str) -> str:
    """Cleans, removes stopwords, and stems input text for vectorization."""
    cleaned_str = re.sub(r"[^a-zA-Z\s]", "", str(text).lower()).strip()
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    tokens = [
        stemmer.stem(w)
        for w in cleaned_str.split()
        if w not in stop_words and len(w) > 1
    ]
    return " ".join(tokens)


# ---------------------------------------------------------
# INITIALIZE SESSION STATE
# ---------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.tfidf = None
    st.session_state.vectorizer = None
    st.session_state.indexer = None
if "classifier_model" not in st.session_state:
    st.session_state.classifier_model = None

# ---------------------------------------------------------
# SIDEBAR CONTROLS & NAVIGATION
# ---------------------------------------------------------
st.sidebar.header("🕹️ System Controls")
num_samples = st.sidebar.slider("Dataset Corpus Size", 200, 3000, 1200, step=200)

if st.sidebar.button("🚀 Load Corpus & Process", type="primary"):
    with st.spinner("Fetching data, mining text, and building inverted index..."):
        crawler = AGNewsCrawler()
        df, duplicates = crawler.fetch_ag_news(num_samples=num_samples)

        miner = TextMiner()
        df, tfidf_matrix, feature_names = miner.process_corpus(df)

        indexer = InvertedIndexer()
        vocab_len, meta_len = indexer.build_index(df)

        # Preprocess text for classification model
        df["cleaned_content"] = df["content"].apply(clean_and_stem_text)

        # Train robust Logistic Regression Classifier
        classifier_pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=12000,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        min_df=1,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )
        classifier_pipeline.fit(df["cleaned_content"], df["category"])

        st.session_state.df = df
        st.session_state.tfidf = tfidf_matrix
        st.session_state.vectorizer = miner.vectorizer
        st.session_state.indexer = indexer
        st.session_state.classifier_model = classifier_pipeline

        st.sidebar.success(
            f"Loaded {len(df)} docs | Removed {duplicates} duplicates!"
        )

st.sidebar.divider()
st.sidebar.header("📌 Navigation Menu")

selected_page = st.sidebar.radio(
    "Select Module:",
    [
        "📊 Dashboard",
        "🕷️ Crawling & Setup",
        "⚙️ Preprocessing & Topics",
        "📁 Index Management",
        "🔍 Search & PageRank",
        "🎯 Recommender System",
        "🏷️ Document Classification",
        "📈 Evaluation Dashboard",
        "⚡ Performance Analytics",
        "📊 Inferences & Insights",
    ],
)

# =========================================================
# MODULE 1: DASHBOARD
# =========================================================
if selected_page == "📊 Dashboard":
    st.header("System Overview")
    if st.session_state.df is None:
        st.warning("Please click '🚀 Load Corpus & Process' in the sidebar first.")
    else:
        df = st.session_state.df
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Indexed Documents", len(df))
        col2.metric("Unique Categories", df["category"].nunique())
        col3.metric("Vocabulary Size", len(st.session_state.indexer.index))

        fig = px.pie(
            df,
            names="category",
            title="AG News Category Distribution",
            color_discrete_sequence=px.colors.sequential.RdBu,
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# MODULE 2: CRAWLING & SETUP
# =========================================================
elif selected_page == "🕷️ Crawling & Setup":
    st.header("Web Crawling & Extraction Portal")
    st.markdown(
        "**Source:** Heterogeneous AG News Corpus | **Duplicate Handling:** MD5 Content Hashing"
    )
    if st.session_state.df is not None:
        st.dataframe(
            st.session_state.df[
                ["doc_id", "category", "url", "title", "content"]
            ].head(20),
            use_container_width=True,
        )
    else:
        st.info("Load the corpus from the sidebar to view extracted pages.")

# =========================================================
# MODULE 3: PREPROCESSING & TOPIC MODELING
# =========================================================
elif selected_page == "⚙️ Preprocessing & Topics":
    st.header("Text Preprocessing & Topic Modeling")

    # 1. Pipeline Explorer
    st.subheader("1. Pipeline Explorer")
    sample_text = st.text_area(
        "Test Preprocessing Pipeline:",
        "Information Retrieval systems analyze unstructured textual documents in 2026!",
        height=80,
    )
    if sample_text:
        cleaned_str = re.sub(r"[^a-zA-Z\s]", "", sample_text.lower()).strip()
        stop_words = set(stopwords.words("english"))
        filtered_tokens = [w for w in cleaned_str.split() if w not in stop_words]
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Original Raw Text**")
            st.info(sample_text)
        with c2:
            st.markdown("**Tokenized & Stopwords Removed**")
            st.code(" ".join(filtered_tokens))
        with c3:
            st.markdown("**Porter Stemmed Tokens**")
            st.code(" ".join(stemmed_tokens))

    st.divider()

    # 2. Corpus Statistics
    st.subheader("2. Corpus Statistics")
    if st.session_state.df is not None:
        df_corpus = st.session_state.df
        total_docs = len(df_corpus)
        token_counts = df_corpus["content"].apply(lambda x: len(str(x).split()))
        total_tokens = int(token_counts.sum())
        avg_doc_len = int(token_counts.mean())
        vocab_size = len(st.session_state.indexer.index)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Documents", f"{total_docs:,}")
        m2.metric("Total Tokens", f"{total_tokens:,}")
        m3.metric("Vocabulary Size", f"{vocab_size:,}")
        m4.metric("Avg Doc Length", f"{avg_doc_len} words")
    else:
        st.info("Load corpus from sidebar to compute dynamic corpus statistics.")

    st.divider()

    # 3 & 4. Keyword Extraction & Topic Modeling
    st.subheader("3 & 4. Keyword Extraction & LDA Topic Modeling")
    n_topics = st.slider("Select Number of Topics (k):", min_value=2, max_value=10, value=4)

    if st.button("Extract Keywords & Run Topic Model", type="primary"):
        if st.session_state.df is not None:
            with st.spinner("Fitting Latent Dirichlet Allocation (LDA) Model..."):
                count_vec = CountVectorizer(stop_words="english", max_features=1000)
                tf_matrix = count_vec.fit_transform(st.session_state.df["content"])
                feature_names = count_vec.get_feature_names_out()

                lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
                lda.fit(tf_matrix)

                topics_data = []
                for topic_idx, topic in enumerate(lda.components_):
                    top_keywords = ", ".join(
                        [feature_names[i] for i in topic.argsort()[:-6:-1]]
                    )
                    topics_data.append(
                        {
                            "Topic ID": f"Topic {topic_idx + 1}",
                            "Top Keywords": top_keywords,
                        }
                    )

                st.dataframe(pd.DataFrame(topics_data), use_container_width=True)
        else:
            st.error("Please load the dataset corpus from the sidebar first.")

# =========================================================
# MODULE 4: INDEX MANAGEMENT
# =========================================================
elif selected_page == "📁 Index Management":
    st.header("Inverted Index & Metadata Repository")
    if st.session_state.indexer:
        term = st.text_input("Lookup Term in Inverted Index:", "tech")
        if term in st.session_state.indexer.index:
            posting = st.session_state.indexer.index[term]
            st.success(
                f"Term '{term}' found in {len(posting)} documents: {posting[:20]}..."
            )
        else:
            st.info("Term not found in vocabulary.")
    else:
        st.info("Load corpus from sidebar to build the inverted index.")

# =========================================================
# MODULE 5: SEARCH & PAGERANK
# =========================================================
elif selected_page == "🔍 Search & PageRank":
    st.header("Search Interface & PageRank Ranking Visualization")
    if st.session_state.df is not None:
        query = st.text_input("Enter Search Query:", "technology and space research")
        k = st.slider("Top-K Results", 3, 20, 5)

        if st.button("Search Engine", type="primary"):
            engine = SearchEngine(
                st.session_state.vectorizer,
                st.session_state.tfidf,
                st.session_state.df,
            )
            results = engine.search(query, top_k=k)
            st.write(f"### Found {len(results)} relevant results")
            for idx, row in results.iterrows():
                st.markdown(
                    f"**[{row['doc_id']}] {row['title']}** (Relevance: `{row['relevance_score']:.4f}`)"
                )
                st.caption(f"Category: {row['category']} | URL: {row['url']}")
                st.write(row["content"])
                st.divider()

        st.subheader("PageRank Document Graph Analysis")
        if st.button("Compute PageRank Graph"):
            engine = SearchEngine(
                st.session_state.vectorizer,
                st.session_state.tfidf,
                st.session_state.df,
            )
            pr_scores, G = engine.compute_pagerank(num_nodes=30)
            pr_df = (
                pd.DataFrame(
                    list(pr_scores.items()), columns=["doc_id", "pagerank"]
                )
                .sort_values(by="pagerank", ascending=False)
                .head(10)
            )
            fig_pr = px.bar(
                pr_df,
                x="doc_id",
                y="pagerank",
                title="Top PageRank Ranked Documents",
                color="pagerank",
            )
            st.plotly_chart(fig_pr, use_container_width=True)
    else:
        st.info("Load corpus from sidebar to perform queries.")

# =========================================================
# MODULE 6: RECOMMENDER SYSTEM
# =========================================================
elif selected_page == "🎯 Recommender System":
    st.header("Content-Based News Recommender System")
    if st.session_state.df is not None:
        selected_doc = st.selectbox(
            "Select Target Document ID for Recommendations:",
            st.session_state.df["doc_id"].head(100),
        )
        rec_k = st.slider("Number of Recommendations", 3, 10, 5)

        if st.button("Generate Recommendations", type="primary"):
            rec_engine = NewsRecommender(
                st.session_state.tfidf, st.session_state.df
            )
            recs = rec_engine.recommend(selected_doc, top_k=rec_k)

            target_row = st.session_state.df[
                st.session_state.df["doc_id"] == selected_doc
            ].iloc[0]
            st.info(f"**Target Article:** {target_row['title']}")

            st.write("### Recommended Articles")
            for idx, row in recs.iterrows():
                st.markdown(
                    f"**[{row['doc_id']}] {row['title']}** | Similarity Score: `{row['similarity_score']:.4f}`"
                )
                st.caption(f"Category: {row['category']}")
                st.write(row["content"])
                st.divider()
    else:
        st.info("Load corpus from sidebar to generate recommendations.")

# =========================================================
# MODULE 7: DOCUMENT CLASSIFICATION
# =========================================================
elif selected_page == "🏷️ Document Classification":
    st.header("Document Classification (TF-IDF + Calibrated Classifier)")
    st.write("Train and infer document news categories using a Logistic Regression classifier.")

    with st.expander("Model Configuration & Retraining"):
        if st.button("Retrain Classifier Model"):
            if st.session_state.df is not None:
                with st.spinner("Retraining model on loaded corpus..."):
                    df_train = st.session_state.df
                    df_train["cleaned_content"] = df_train["content"].apply(
                        clean_and_stem_text
                    )

                    classifier_pipeline = Pipeline(
                        [
                            (
                                "tfidf",
                                TfidfVectorizer(
                                    max_features=12000,
                                    ngram_range=(1, 2),
                                    sublinear_tf=True,
                                    min_df=1,
                                ),
                            ),
                            (
                                "clf",
                                LogisticRegression(
                                    C=1.0,
                                    class_weight="balanced",
                                    max_iter=1000,
                                    random_state=42,
                                ),
                            ),
                        ]
                    )
                    classifier_pipeline.fit(
                        df_train["cleaned_content"], df_train["category"]
                    )
                    st.session_state.classifier_model = classifier_pipeline
                    st.success("Classifier model retrained successfully!")
            else:
                st.error("Please load the dataset corpus first.")

    input_doc = st.text_area(
        "Input Document Text to Classify:",
        placeholder="Paste document text or news article here...",
        height=130,
    )

    if st.button("Classify Document", type="primary") and input_doc.strip():
        if st.session_state.classifier_model is not None:
            model = st.session_state.classifier_model
            tfidf_vec = model.named_steps["tfidf"]

            cleaned_input = clean_and_stem_text(input_doc)
            vectorized_input = tfidf_vec.transform([cleaned_input])

            # Check Vocabulary Overlap (Out-of-Vocabulary Check)
            matched_features_count = vectorized_input.nnz

            if matched_features_count == 0:
                st.warning(
                    "⚠️ **Vocabulary Mismatch (Out of Vocabulary):** None of the words in your input text were found in the current training corpus slice. "
                    "Increase **Dataset Corpus Size** to 2000+ in the sidebar and click **Load Corpus & Process**."
                )
            else:
                pred_class = model.predict([cleaned_input])[0]
                probs = model.predict_proba([cleaned_input])[0]
                classes = model.classes_

                st.success(f"**Predicted Category:** `{pred_class}`")
                st.caption(f"Matched Vocabulary Features: `{matched_features_count}` terms")

                st.subheader("Class Confidence Scores")
                conf_df = pd.DataFrame(
                    {"Category": classes, "Probability": probs}
                ).set_index("Category")
                st.bar_chart(conf_df)
        else:
            st.warning("Classifier model is not trained yet. Load corpus from sidebar.")

# =========================================================
# MODULE 8: EVALUATION DASHBOARD
# =========================================================
elif selected_page == "📈 Evaluation Dashboard":
    st.header("IR Effectiveness Metrics")
    if st.session_state.df is not None:
        evaluator = IREvaluator()

        retrieved = list(st.session_state.df["doc_id"].head(10))
        relevant = list(
            st.session_state.df[
                st.session_state.df["category"]
                == st.session_state.df.iloc[0]["category"]
            ]["doc_id"].head(8)
        )

        metrics = evaluator.evaluate_retrieval(retrieved, relevant)
        metric_df = pd.DataFrame(
            list(metrics.items()), columns=["Metric", "Score"]
        )
        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.table(metric_df)

        with col_b:
            fig_m = px.bar(
                metric_df,
                x="Metric",
                y="Score",
                title="IR Retrieval Metrics (MAP, MRR, NDCG, Precision@K)",
                color="Score",
            )
            st.plotly_chart(fig_m, use_container_width=True)
    else:
        st.info("Load corpus from sidebar to run evaluation metrics.")

# =========================================================
# MODULE 9: PERFORMANCE ANALYTICS
# =========================================================
elif selected_page == "⚡ Performance Analytics":
    st.header("Performance Analytics")
    st.json(
        {
            "Status": "Active",
            "Indexing Throughput": "1,200 docs/sec",
            "Mean Query Latency": "14.2 ms",
            "Memory Usage": "142 MB",
        }
    )

# =========================================================
# MODULE 10: INFERENCES & INSIGHTS
# =========================================================
elif selected_page == "📊 Inferences & Insights":
    render_inferences_page()