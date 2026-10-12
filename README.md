# Assignment 20 — Content-Based Movie Recommendation System

**Student:** Kanan Pandey  
**Dataset:** TMDB 5000 Movie Dataset  
**Dataset source:** [Kaggle — TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

This project builds a content-based movie recommender from plot overviews. It cleans the text, creates TF-IDF vectors, compares movies with cosine similarity, and provides a small Streamlit interface.

## Assignment checklist

| Task | Requirement | Project implementation |
|---|---|---|
| 1 | Load and understand the CSV | recommender.py reports shape, columns, first five records, data types, missing values, and the item/text columns. |
| 2 | Preprocess recommendation text | Fills missing overviews, lowercases, removes punctuation/numbers/special characters, normalizes whitespace, removes English stopwords, and stores the result in clean_text. |
| 3 | TF-IDF vectorization | Uses TfidfVectorizer(max_features=5000, ngram_range=(1, 2)) and prints the matrix shape. |
| 4 | Cosine similarity | Computes and stores the pairwise cosine similarity matrix; the implementation keeps it sparse to reduce memory use. |
| 5 | Recommendation function | recommend(item_name, top_n=5, ...) ranks similar movies and excludes the selected movie. The CLI demonstrates three different titles. |
| 6 | Simple app interface | app.py provides a movie selector, recommendation-count control, generate button, and readable results. |
| 7 | Git and GitHub | Local Git setup is prepared. Create/push the repository under the student's GitHub account after that account is selected in the GitHub connection. |
| 8 | Render deployment | render.yaml defines the Streamlit web service and start command. Connect the GitHub repository to Render to deploy. |
| 9 | Final validation | Local command-line recommendations are checked. Add the live Render URL here after deployment: **Pending deployment**. |

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

## GitHub setup

After confirming that the GitHub connection is using the student's kananpandey account, create a repository named GenAI-Task20-Kanan-Pandey, then set it as the origin remote and push the main branch. Do not push this project through a different account.

## Render deployment

The included render.yaml uses the repository root, installs requirements.txt, and starts Streamlit on Render's assigned port. In Render, create a Blueprint from the GitHub repository and wait for the service to finish deploying. Then open the public URL, test the selector and recommendation button, and replace **Pending deployment** in this README with the working URL.

## Verification

See AUDIT_REPORT.md for the checks run locally and any remaining external steps.
