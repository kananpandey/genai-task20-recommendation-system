# Assignment 20 — Verification Report

**Project:** Content-Based Movie Recommendation System  
**Student:** Kanan Pandey  
**Verification date:** 2026-10-12

## Requirement-by-requirement review

| Task | Result | Evidence |
|---|---|---|
| 1. Load and understand dataset | Pass | Pandas inspection prints raw shape, columns, first five rows, data types, nulls, and identifies title as the item identifier and overview as the text feature. |
| 2. Text preprocessing | Pass | Missing overviews are filled; clean_text is lowercased, stripped of punctuation/numbers/special characters, whitespace-normalized, and filtered for English stopwords. |
| 3. TF-IDF | Pass | Uses TfidfVectorizer with max_features=5000 and ngram_range=(1, 2); the command-line run prints the resulting matrix dimensions. |
| 4. Cosine similarity | Pass | Pairwise cosine similarity is calculated and retained as a sparse matrix; its dimensions are printed. |
| 5. Recommendation function | Pass | The CLI calls the function for Avatar, The Dark Knight Rises, and Toy Story, and checks that each selected item is excluded from its recommendations. |
| 6. Streamlit interface | Pass | The public app exposes a movie selector, recommendation-count slider, button, and readable recommendation cards. Live browser checks generated results for multiple selections. |
| 7. GitHub repository | Pass | The project is committed on main in Kanan Pandey's private repository: [kananpandey/genai-task20-recommendation-system](https://github.com/kananpandey/genai-task20-recommendation-system). If course reviewers need direct access, grant it or change the repository visibility according to course rules. |
| 8. Render deployment | Pass | Render reports the service as live at [https://genai-task20-recommendation-system.onrender.com](https://genai-task20-recommendation-system.onrender.com). It deployed from main; the first deployment built commit 68ed737. |
| 9. Deployed app validation | Pass | The public page loaded. Clicking Recommend Movies for Avatar and Toy Story returned five recommendations each; neither selected title appeared in its own results. The default The Dark Knight Rises view also displayed five recommendations. |

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

The CLI inspects and preprocesses the data, prints TF-IDF and similarity dimensions, shows recommendations for three titles, and exercises invalid-title, blank-title, and empty-overview handling. The sparse similarity matrix reduces memory use while representing pairwise cosine scores.

The Streamlit app can be started with:

~~~powershell
streamlit run app.py
~~~

## Local run result

The verification command was executed successfully with Python 3.11.9. It reported a raw dataset shape of (4,803, 20), removed three duplicate titles, replaced the three missing overviews, produced a TF-IDF matrix of (4,800, 5,000), and produced a sparse cosine-similarity matrix of (4,800, 4,800) with 6,518,148 stored non-zero scores (about 74.61 MB). It returned five recommendations for Avatar, The Dark Knight Rises, and Toy Story, excluded each selected movie from its own result list, and handled invalid, blank, and empty-overview cases.

## Public deployment validation

- Render deployment status: **Deploy succeeded | Live**
- Public URL: [https://genai-task20-recommendation-system.onrender.com](https://genai-task20-recommendation-system.onrender.com)
- The page initially showed Streamlit's running state while loading the dataset and similarity data, then displayed the selector and recommendation controls.
- Avatar produced five results, beginning with Apollo 18; Avatar was not among the results.
- Toy Story produced five results, beginning with Toy Story 3; Toy Story was not among the results.
- The default The Dark Knight Rises view showed five recommendations and did not include The Dark Knight Rises itself.
- Render's free instance may spin down after inactivity, so a later first visit can take longer to respond.

## Submission readiness

The assignment implementation is complete, the repository is published on GitHub, and the public Render app and recommendation interaction were verified. The repository is private, so confirm whether the course reviewer needs GitHub access; the project ZIP can also be submitted through the course portal.
