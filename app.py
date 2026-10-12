"""
app.py
------
Streamlit Web Application for Assignment 20 — Content-Based Movie Recommendation System.

Author: Kanan Pandey
Course: TuteDude GenAI
"""

import os
from pathlib import Path
import pandas as pd
import streamlit as st

# Import the core recommendation engine
from recommender import (
    load_and_inspect_dataset,
    preprocess_dataset,
    build_tfidf_model,
    compute_similarity_matrix,
    recommend as core_recommend,
    DEFAULT_DATASET_PATH
)

# Re-export recommend function for direct import compatibility
recommend = core_recommend


@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    """
    Loads and preprocesses the movie dataset once and caches the result.
    """
    df = load_and_inspect_dataset(DEFAULT_DATASET_PATH)
    df = preprocess_dataset(df)
    return df


@st.cache_resource(show_spinner=False)
def load_similarity_engine(clean_texts):
    """
    Computes TF-IDF vectors and pairwise cosine similarity once and caches the model.
    """
    tfidf_vectorizer, tfidf_matrix = build_tfidf_model(clean_texts, max_features=5000, ngram_range=(1, 2))
    sim_matrix = compute_similarity_matrix(tfidf_matrix)
    return tfidf_vectorizer, tfidf_matrix, sim_matrix


def main():
    # 1. Page Configuration
    st.set_page_config(
        page_title="Movie Recommender System",
        page_icon="🎬",
        layout="wide"
    )

    # 2. Header and Objective
    st.title("🎬 Content-Based Movie Recommendation System")
    st.markdown(
        """
        Welcome to the **Content-Based Movie Recommendation System** built for **TuteDude GenAI Assignment 20**.
        
        **How it works:**
        - **Text Preprocessing:** Cleans and normalizes plot overviews by lowercasing, removing special characters, and stripping English stopwords.
        - **TF-IDF Vectorization:** Transforms textual overviews into numerical vectors highlighting unique plot keywords.
        - **Cosine Similarity:** Measures the angle between movie vectors to find movies with the most similar plots and themes.
        """
    )
    st.divider()

    # 3. Check dataset availability
    if not os.path.exists(DEFAULT_DATASET_PATH):
        st.error(
            f"❌ Dataset file not found at `{DEFAULT_DATASET_PATH}`. "
            "Please ensure `movies.csv` (TMDB 5000 Movie Dataset) is placed in the project directory."
        )
        st.stop()

    # 4. Load Data with loading indicator
    with st.spinner("⏳ Loading movie dataset and computing similarity matrix..."):
        try:
            movies_df = load_and_prepare_data()
            _, _, sim_matrix = load_similarity_engine(movies_df["clean_text"])
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            st.stop()

    # 5. User Input Controls
    st.subheader("🎯 Select a Movie to Get Recommendations")
    col_input, col_settings = st.columns([3, 1])

    movie_titles = movies_df["title"].tolist()
    # Default selection: 'The Dark Knight Rises' or first movie
    default_index = movie_titles.index("The Dark Knight Rises") if "The Dark Knight Rises" in movie_titles else 0

    with col_input:
        selected_movie = st.selectbox(
            "Choose a movie from the catalog:",
            options=movie_titles,
            index=default_index,
            help="Type to search for any movie in the TMDB 5000 dataset."
        )

    with col_settings:
        top_n = st.slider(
            "Number of recommendations:",
            min_value=1,
            max_value=15,
            value=5,
            step=1,
            help="Select how many similar movies you want to discover."
        )

    recommend_button = st.button("🚀 Recommend Movies", type="primary", use_container_width=True)

    # 6. Show Selected Movie Details
    if selected_movie:
        selected_row = movies_df[movies_df["title"] == selected_movie].iloc[0]
        with st.expander(f"ℹ️ Selected Movie Details: **{selected_movie}**", expanded=False):
            c1, c2 = st.columns([1, 4])
            with c1:
                st.metric("Release Date", str(selected_row.get("release_date", "N/A")))
                st.metric("Vote Average", f"{selected_row.get('vote_average', 'N/A')} / 10")
            with c2:
                tagline = selected_row.get("tagline", "")
                if pd.notna(tagline) and str(tagline).strip():
                    st.markdown(f"*\"{tagline}\"*")
                st.write(selected_row.get("overview", "No overview available."))

    # 7. Generate Recommendations
    if recommend_button:
        st.subheader(f"✨ Top {top_n} Recommendations for \"{selected_movie}\"")
        
        with st.spinner("🔍 Calculating semantic similarity..."):
            results = core_recommend(
                item_name=selected_movie,
                top_n=top_n,
                df=movies_df,
                similarity_matrix=sim_matrix
            )

        if isinstance(results, str):
            st.error(results)
        elif isinstance(results, pd.DataFrame) and not results.empty:
            # Display results in structured columns / cards
            for rank, (_, row) in enumerate(results.iterrows(), start=1):
                score_pct = row["similarity_score"] * 100
                with st.container():
                    st.markdown(
                        f"""
                        #### #{rank} — **{row['title']}** &nbsp;&nbsp; `Similarity: {score_pct:.1f}% ({row['similarity_score']:.4f})`
                        """
                    )
                    st.write(row["overview"])
                    st.divider()

            with st.expander("📊 Similarity scores"):
                for _, row in results.iterrows():
                    st.write(f"- {row['title']}: {row['similarity_score']:.4f}")
        else:
            st.warning("No recommendations could be generated for the selected item.")

    # 8. Footer & Metadata
    st.markdown("---")
    st.caption(
        f"TMDB 5000 Movie Dataset • {len(movies_df):,} movies indexed • Vocabulary size: 5,000 TF-IDF features • "
        "Prepared for TuteDude Assignment 20."
    )


if __name__ == "__main__":
    main()
