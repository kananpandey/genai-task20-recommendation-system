# Assignment 20 — Content-Based Movie Recommendation System

**Student:** Kanan Pandey  
**Dataset:** TMDB 5000 Movie Dataset  
**Dataset source:** [Kaggle — TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

This project builds a content-based movie recommender from plot overviews. It cleans the text, creates TF-IDF vectors, compares movies with cosine similarity, and provides a Streamlit interface.

## Assignment checklist

| Task | Requirement | Project implementation |
|---|---|---|
| 1 | Load and understand the CSV | recommender.py reports shape, columns, first five records, data types, missing values, and the item/text columns. |
| 2 | Preprocess recommendation text | Fills missing overviews, lowercases, removes punctuation/numbers/special characters, normalizes whitespace, removes English stopwords, and stores the result in clean_text. |
| 3 | TF-IDF vectorization | Uses TfidfVectorizer(max_features=5000, ngram_range=(1, 2)) and prints the matrix shape. |
| 4 | Cosine similarity | Computes and stores the pairwise cosine similarity matrix; the implementation keeps it sparse to reduce memory use. |
| 5 | Recommendation function | recommend(item_name, top_n=5, ...) ranks similar movies and excludes the selected movie. The CLI demonstrates three different titles. |
| 6 | Simple app interface | app.py provides a movie selector, recommendation-count control, generate button, and readable results. |
| 7 | Git and GitHub | Complete: all nine project files are committed on main in [kananpandey/genai-task20-recommendation-system](https://github.com/kananpandey/genai-task20-recommendation-system). The repository is private; grant reviewer access or change visibility if the course requires it. |
| 8 | Render deployment | Complete: the Streamlit service is deployed from the GitHub main branch. |
| 9 | Final validation | Complete: the public app opened successfully and generated recommendations for Avatar and Toy Story. Each result list excluded the selected movie. |

## Dataset details

- **Name:** TMDB 5000 Movie Dataset
- **Kaggle page:** [https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Included file:** movies.csv
- **Raw size:** 4,803 rows × 20 columns
- **Item identifier:** title
- **Text feature:** overview
- Three duplicate movie titles are removed before title lookup; blank overviews are filled with an empty string.

The dataset file is included so the project runs without downloading data at startup. Keep the Kaggle attribution when redistributing it.

## Files

- app.py — Streamlit app
- recommender.py — data inspection, cleaning, TF-IDF, similarity, and recommendation logic
- movies.csv — included dataset
- requirements.txt — pinned Python dependencies
- render.yaml and .python-version — Render service/runtime configuration
- AUDIT_REPORT.md — implementation and verification record

## Run locally

Use Python 3.11.9. From this folder:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python recommender.py
streamlit run app.py
~~~

The app runs locally at http://localhost:8501.

## How the recommendations work

1. Missing plot overviews become empty strings.
2. Text is lowercased; non-letter characters (including punctuation, digits, and symbols) become spaces; whitespace is normalized; English stopwords are removed.
3. The cleaned text is transformed with unigram and bigram TF-IDF features, capped at 5,000 features.
4. Cosine similarity compares every movie's TF-IDF vector. Similarity scores range from 0 (no shared weighted terms) to 1 (identical direction).
5. The recommendation function sorts by score and excludes the selected title from its own results.

This is a content-based method: it uses plot text and does not use collaborative filtering.

## GitHub repository

The project files are committed on the main branch in [kananpandey/genai-task20-recommendation-system](https://github.com/kananpandey/genai-task20-recommendation-system). This repository is private. If a reviewer needs access, grant access to them or change the repository visibility if the course requires it.

## Render deployment

**Live application:** [https://genai-task20-recommendation-system.onrender.com](https://genai-task20-recommendation-system.onrender.com)

The service is connected to the GitHub repository's main branch. Render builds it with the included requirements and starts the Streamlit app on Render's assigned port. The deployed page was checked in a browser, and its recommendation flow was exercised with multiple movies.

The service uses Render's free instance. After a period without activity, it may spin down; the first visit after that can take 50 seconds or more to respond.

## Verification

See AUDIT_REPORT.md for the requirement-by-requirement review, local run results, and public app validation.
