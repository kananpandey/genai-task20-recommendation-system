# Assignment 20 — Verification Report

**Project:** Content-Based Movie Recommendation System  
**Student:** Kanan Pandey  
**Verification date:** 2026-10-12

## Requirement-by-requirement review

| Task | Result | Evidence / remaining action |
|---|---|---|
| 1. Load and understand dataset | Pass | Pandas inspection prints raw shape, columns, first five rows, data types, nulls, and identifies title as the item identifier and overview as the text feature. |
| 2. Text preprocessing | Pass | Missing overviews are filled; the clean_text column is lowercased, stripped of punctuation/numbers/special characters, whitespace-normalized, and filtered for English stopwords. |
| 3. TF-IDF | Pass | Uses TfidfVectorizer with max_features=5000 and ngram_range=(1, 2); the command-line run prints the resulting matrix dimensions. |
| 4. Cosine similarity | Pass | Pairwise cosine similarity is calculated and retained as a sparse matrix; its dimensions are printed. |
| 5. Recommendation function | Pass | The CLI calls the function for Avatar, The Dark Knight Rises, and Toy Story, and checks that each selected item is excluded from its recommendations. |
| 6. Streamlit interface | Automated UI flow verified; browser/live verification pending | Streamlit AppTest confirmed the selector, recommendation-count slider, button, no premature results, and recommendations after clicking. A public browser deployment remains pending. |
| 7. GitHub repository | Pending account selection/push | Local files are ready for Git. The GitHub connector reports ShivamGrover-05 as its active login, so no push was made to the wrong account. Select/connect kananpandey before publishing. |
| 8. Render deployment | Configuration prepared; deployment pending | render.yaml defines the service, build command, and Streamlit start command. A Render service still needs to be created from the correct GitHub repository. |
| 9. Deployed app validation | Pending deployment | No public Render URL is available yet; once deployed, verify the URL and recommendation flow, then record that URL in the README. |

## Dataset facts checked

- Source: [Kaggle — TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- Included CSV: movies.csv
- Raw dataset dimensions: **4,803 rows × 20 columns**
- Raw null values reported: **3 in overview**
- Duplicate titles removed by the project: **3**
- Expected working rows after title deduplication: **4,800**
- Identifier: title
- Content feature: overview

## Local verification procedure

Run:

~~~powershell
python recommender.py
~~~

The CLI should inspect and preprocess the data, print TF-IDF and similarity dimensions, show recommendations for three titles, and exercise invalid-title, blank-title, and empty-overview handling. The sparse similarity matrix is used to reduce memory consumption while still representing all pairwise cosine scores.

The Streamlit app can be started with:

~~~powershell
streamlit run app.py
~~~

A successful command start is not the same as a browser or production deployment check. The live URL and end-to-end checks remain pending until deployment is completed.

### Run result

The verification command was executed successfully with Python 3.11.9. It reported a raw dataset shape of (4,803, 20), removed three duplicate titles, replaced the three missing overviews, produced a TF-IDF matrix of (4,800, 5,000), and produced a sparse cosine-similarity matrix of (4,800, 4,800) with 6,518,148 stored non-zero scores (about 74.61 MB). It returned five recommendations for Avatar, The Dark Knight Rises, and Toy Story, excluded each selected movie from its own result list, and handled invalid, blank, and empty-overview cases.

The Streamlit application itself and a public Render deployment have not yet been browser-tested.
## Submission readiness

The core assignment implementation and deployment configuration are present. The GitHub publication, Render service, live URL, and final deployed-app validation are external steps that remain incomplete until the correct GitHub identity and Render account are available.
