# Assignment 20 — Content-Based Movie Recommendation System

**Student:** Kanan Pandey  
**Dataset:** TMDB 5000 Movie Dataset  
**Kaggle source:** [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

This project recommends movies with plot overviews. It cleans the overview text, converts it into TF-IDF vectors, compares movies with cosine similarity, and provides a Streamlit interface.

## Assignment coverage

| Task | Requirement | Implementation |
|---|---|---|
| 1 | Load and understand the dataset | `recommender.py` prints the dataset shape, columns, first five rows, data types, missing values, and the selected item and text columns. |
| 2 | Preprocess recommendation text | Fills missing overviews, lowercases text, removes punctuation, numbers, and special characters, normalizes whitespace, removes English stopwords, and stores the result in `clean_text`. |
| 3 | Apply TF-IDF vectorization | Uses `TfidfVectorizer(max_features=5000, ngram_range=(1, 2))` and prints the matrix dimensions. |
| 4 | Calculate cosine similarity | Computes pairwise cosine similarity and stores the matrix sparsely to reduce memory use. |
| 5 | Generate recommendations | `recommend(item_name, top_n=5)` ranks similar movies and excludes the selected movie. The CLI demonstrates three titles. |
| 6 | Build a simple interface | `app.py` provides a movie selector, recommendation-count control, button, and results. |
| 7 | Use Git and GitHub | Project source is maintained in the [GitHub repository](https://github.com/kananpandey/genai-task20-recommendation-system). |
| 8 | Deploy on Render | Render service configuration is included in `render.yaml`. |
| 9 | Include the deployed application link | The public application URL is listed below. |

## Dataset

- **Name:** TMDB 5000 Movie Dataset
- **Kaggle page:** [https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Included file:** `movies.csv`
- **Raw size:** 4,803 rows × 20 columns
- **Item identifier:** `title`
- **Text feature:** `overview`

The project removes three duplicate titles and replaces missing overviews with empty strings. The dataset is included so the application can run without downloading data at startup. Keep the Kaggle attribution when redistributing it.

## Project files

- `app.py` — Streamlit application
- `recommender.py` — dataset inspection, text cleaning, TF-IDF, similarity, and recommendation logic
- `movies.csv` — dataset
- `requirements.txt` — pinned Python dependencies
- `render.yaml` and `.python-version` — Render service and runtime configuration

## Run locally

Use Python 3.11.9. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python recommender.py
streamlit run app.py
```

The app runs locally at http://localhost:8501.

## How recommendations work

1. Missing plot overviews are replaced with empty strings.
2. Text is lowercased, non-letter characters are replaced with spaces, whitespace is normalized, and English stopwords are removed.
3. The cleaned text is transformed into unigram and bigram TF-IDF features, with a maximum vocabulary size of 5,000.
4. Cosine similarity compares the direction of two TF-IDF vectors. Because the vectors are normalized, the score focuses on how their weighted terms overlap rather than on the length of each overview. This makes it a useful measure for comparing plot descriptions.
5. Recommendations are sorted by similarity score, and the selected movie is excluded from its own results.

This is a content-based method: it uses plot text and does not use collaborative filtering.

## GitHub repository

[Open the public project repository](https://github.com/kananpandey/genai-task20-recommendation-system). The project files are committed on the `main` branch.

## Render deployment

**Application URL:** [https://genai-task20-recommendation-system.onrender.com](https://genai-task20-recommendation-system.onrender.com)

The app provides a movie selector and a **Recommend Movies** button. Select a movie and click the button to view similar titles. A selected movie is excluded from its own recommendations. Render's free instance may spin down after inactivity, so the first visit may take longer to load.
