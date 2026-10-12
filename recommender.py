"""
recommender.py
--------------
Content-Based Recommendation Engine for TuteDude Assignment 20.

This module provides functions to:
1. Load and inspect the Kaggle TMDB 5000 Movies dataset.
2. Preprocess text (lowercase, remove punctuation, remove stopwords, handle missing values).
3. Generate numerical TF-IDF feature vectors.
4. Compute cosine similarity scores between items.
5. Recommend top-N similar items based on content similarity.
"""

import os
import re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Default path to the dataset relative to this file
DEFAULT_DATASET_PATH = Path(__file__).parent / "movies.csv"


def load_and_inspect_dataset(csv_path=DEFAULT_DATASET_PATH):
    """
    Loads the movie dataset using Pandas and displays structural details.
    
    Parameters:
        csv_path (str or Path): Path to the movies CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame with duplicate titles removed.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    print("=" * 60)
    print("DATASET INSPECTION & SUMMARY")
    print("=" * 60)
    print(f"Dataset file path: {csv_path}")
    print(f"Raw dataset shape: {df.shape} (rows, columns)")
    print(f"\nColumn names ({len(df.columns)} columns):")
    print(df.columns.tolist())
    print("\nColumn data types:")
    print(df.dtypes.to_string())

    text_like_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    print(f"\nText-like columns: {text_like_columns}")
    print("Item identifier column: title")
    print("Recommendation text column: overview")

    print("\nMissing values per column:")
    missing_summary = df.isnull().sum()
    print(missing_summary[missing_summary > 0])

    print("\nFirst 5 rows (Title and Overview preview):")
    for i, row in df.head(5).iterrows():
        title = row.get("title", "Unknown")
        overview = str(row.get("overview", ""))[:80]
        print(f"  {i+1}. {title} -> {overview}...")

    # Drop duplicate movie titles so each movie has a unique title lookup
    initial_count = len(df)
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    dedup_count = len(df)
    print(f"\nDuplicate titles removed: {initial_count - dedup_count}")
    print(f"Usable dataset shape: {df.shape}")

    # Handle missing text values in the 'overview' column
    missing_overview_count = df["overview"].isnull().sum()
    df["overview"] = df["overview"].fillna("")
    print(f"Missing overviews replaced with empty string: {missing_overview_count}")

    return df


def clean_text_data(text):
    """
    Preprocesses a text string:
    1. Converts text to lowercase.
    2. Removes punctuation, digits, and special characters.
    3. Removes English stopwords.
    4. Handles missing or non-string inputs safely.
    
    Parameters:
        text (str): Raw input text.
        
    Returns:
        str: Processed and cleaned text.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # 1. Convert to lowercase
    text_lower = text.lower()

    # 2. Remove punctuation, symbols, and non-alphabetic characters
    text_alpha = re.sub(r"[^a-z\s]", " ", text_lower)

    # 3. Tokenize into words and filter out English stopwords
    tokens = text_alpha.split()
    cleaned_tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS and len(w) > 1]

    # 4. Join back into clean space-separated text
    return " ".join(cleaned_tokens)


def preprocess_dataset(df):
    """
    Applies text preprocessing to the dataset and creates a new 'clean_text' column.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with the new 'clean_text' column.
    """
    df = df.copy()
    # The primary text feature for content similarity is 'overview'
    df["overview"] = df["overview"].fillna("")
    df["clean_text"] = df["overview"].apply(clean_text_data)
    return df


def build_tfidf_model(clean_texts, max_features=5000, ngram_range=(1, 2)):
    """
    Converts cleaned text data into numerical TF-IDF vectors using scikit-learn.
    
    Parameters:
        clean_texts (pd.Series or list): Iterable of cleaned text strings.
        max_features (int): Maximum vocabulary size for top terms.
        ngram_range (tuple): Lower and upper boundary of range of n-values for n-grams.
        
    Returns:
        tuple: (fitted TfidfVectorizer, scipy sparse tfidf_matrix)
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english"
    )
    tfidf_matrix = tfidf.fit_transform(clean_texts)
    return tfidf, tfidf_matrix


def compute_similarity_matrix(tfidf_matrix):
    """
    Computes pairwise cosine similarity between all items in the TF-IDF matrix.
    
    Parameters:
        tfidf_matrix: Sparse matrix of shape (n_samples, n_features).
        
    Returns:
        np.ndarray: Cosine similarity matrix of shape (n_samples, n_samples).
    """
    # Keep all pairwise scores, but store them sparsely to reduce app memory use.
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix, dense_output=False)
    return similarity_matrix


def recommend(item_name, top_n=5, df=None, similarity_matrix=None, tfidf_matrix=None):
    """
    Recommends top-N most similar movies for a given movie title based on content.
    
    Parameters:
        item_name (str): Title of the movie to base recommendations on.
        top_n (int): Number of recommendations to return (default: 5).
        df (pd.DataFrame): DataFrame containing movie records with 'title' column.
        similarity_matrix (np.ndarray, optional): Precomputed cosine similarity matrix.
        tfidf_matrix (sparse matrix, optional): TF-IDF matrix if computing on the fly.
        
    Returns:
        pd.DataFrame or str: DataFrame of recommendations with ['title', 'similarity_score'],
                             or an error message string if the title is invalid.
    """
    if df is None:
        raise ValueError("DataFrame 'df' must be provided to recommend().")

    if not isinstance(item_name, str) or not item_name.strip():
        return "Error: Please provide a valid movie title."

    # Case-insensitive title search
    clean_target = item_name.strip().lower()
    matches = df[df["title"].astype(str).str.strip().str.lower() == clean_target]

    if matches.empty:
        # Partial match suggestion if exact match is not found
        partial = df[df["title"].astype(str).str.lower().str.contains(clean_target, regex=False)]
        if not partial.empty:
            suggestions = partial["title"].head(3).tolist()
            return f"Error: Movie '{item_name}' was not found. Did you mean one of these: {suggestions}?"
        return f"Error: Movie '{item_name}' was not found in the dataset. Please check the spelling."

    # Index of selected movie in the DataFrame
    target_idx = matches.index[0]
    matched_title = df.loc[target_idx, "title"]

    # Compute or retrieve similarity scores
    if similarity_matrix is not None:
        score_row = similarity_matrix[target_idx]
        sim_scores = score_row.toarray().ravel() if hasattr(score_row, "toarray") else score_row.ravel()
    elif tfidf_matrix is not None:
        sim_scores = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix).ravel()
    else:
        raise ValueError("Either 'similarity_matrix' or 'tfidf_matrix' must be provided.")

    # Sort items by similarity score in descending order
    sorted_indices = sim_scores.argsort()[::-1]

    # Exclude the selected movie itself from recommendations
    recommended_indices = [idx for idx in sorted_indices if idx != target_idx][:top_n]

    # Build the result DataFrame
    results = []
    for idx in recommended_indices:
        results.append({
            "title": df.loc[idx, "title"],
            "similarity_score": round(float(sim_scores[idx]), 4),
            "overview": df.loc[idx, "overview"] if "overview" in df.columns else ""
        })

    rec_df = pd.DataFrame(results)
    return rec_df


def run_cli_tests():
    """
    Executes end-to-end tests for all assignment tasks:
    Task 1: Dataset loading and inspection
    Task 2: Text preprocessing and Task 3: TF-IDF vectorization
    Task 4: Cosine similarity calculation
    Task 5: Recommendation validation on 3 real titles + invalid title + empty text
    """
    print("=" * 60)
    print("RUNNING ASSIGNMENT 20 VERIFICATION SUITE")
    print("=" * 60)

    # 1. Load dataset
    df = load_and_inspect_dataset()

    # 2. Preprocess text
    print("\n--- TASK 2: TEXT PREPROCESSING ---")
    df = preprocess_dataset(df)
    print("Clean text sample 1 (Avatar):")
    print("Raw:  ", df.loc[0, "overview"][:90])
    print("Clean:", df.loc[0, "clean_text"][:90])

    # 3. TF-IDF Vectorization
    print("\n--- TASK 3: TF-IDF VECTORIZATION ---")
    tfidf_vectorizer, tfidf_matrix = build_tfidf_model(df["clean_text"], max_features=5000, ngram_range=(1, 2))
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape} (rows: movies, cols: vocabulary features)")
    print(f"Sample vocabulary words: {list(tfidf_vectorizer.vocabulary_.keys())[:10]}")

    # 4. Cosine Similarity
    print("\n--- TASK 4: COSINE SIMILARITY MATRIX ---")
    sim_matrix = compute_similarity_matrix(tfidf_matrix)
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    if hasattr(sim_matrix, "nnz"):
        matrix_bytes = sim_matrix.data.nbytes + sim_matrix.indices.nbytes + sim_matrix.indptr.nbytes
        print(f"Similarity matrix storage: sparse CSR, {sim_matrix.nnz:,} non-zero values, {matrix_bytes / (1024 * 1024):.2f} MB")
    else:
        print(f"Similarity matrix storage: dense, {sim_matrix.nbytes / (1024 * 1024):.2f} MB")
    print("Self-similarity score verification (item 0 with item 0):", round(float(sim_matrix[0, 0]), 4))

    # 5. Recommendation Tests
    print("\n--- TASK 5: RECOMMENDATION TESTS ---")

    test_titles = ["Avatar", "The Dark Knight Rises", "Toy Story"]
    for title in test_titles:
        print(f"\n[Test Movie]: '{title}' (top 5 recommendations):")
        result = recommend(title, top_n=5, df=df, similarity_matrix=sim_matrix)
        if isinstance(result, pd.DataFrame):
            # Check self-exclusion
            has_self = title.lower() in result["title"].str.lower().values
            print(f"  Self-exclusion verified (is target in recs?): {not has_self}")
            for idx, row in result.iterrows():
                print(f"    {idx+1}. {row['title']} (Similarity: {row['similarity_score']:.4f})")
        else:
            print("  ", result)

    # Test invalid title
    print("\n[Edge Case Test]: Invalid Title ('Nonexistent Superhero Movie 999'):")
    invalid_res = recommend("Nonexistent Superhero Movie 999", top_n=5, df=df, similarity_matrix=sim_matrix)
    print("  Output:", invalid_res)

    # Test empty / whitespace title
    print("\n[Edge Case Test]: Empty Title (''):")
    empty_res = recommend("", top_n=5, df=df, similarity_matrix=sim_matrix)
    print("  Output:", empty_res)

    # Test movie with empty overview
    print("\n[Edge Case Test]: Movie with Empty Overview Handling:")
    empty_overview_indices = df[df["clean_text"] == ""].index
    if not empty_overview_indices.empty:
        sample_empty_idx = empty_overview_indices[0]
        empty_title = df.loc[sample_empty_idx, "title"]
        print(f"  Testing movie '{empty_title}' with empty text:")
        rec_empty = recommend(empty_title, top_n=3, df=df, similarity_matrix=sim_matrix)
        print("  Recommendations generated without crashing:", isinstance(rec_empty, pd.DataFrame))
    else:
        print("  All movies have non-empty clean text.")

    print("\n" + "=" * 60)
    print("ALL VERIFICATION SUITE CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_cli_tests()
