import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from csp_filter import apply_csp_filter, sanitize_dataframe
from heuristic import compute_heuristic_score
from clustering import cluster_movies
from ann_model import predict_ratings
from explain import add_explanations


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "movies.csv")


def load_movies():
    # Try UTF-8 first, fall back to latin-1 for Windows-saved files
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding="latin-1")

    # Always sanitize column names and values to remove hidden quotes
    df = sanitize_dataframe(df)
    return df


def get_recommendations(genre, min_rating, max_runtime, min_year, top_n=5):
    try:
        df = load_movies()
    except Exception as e:
        return pd.DataFrame(), f"Could not load movies.csv: {e}"

    try:
        filtered = apply_csp_filter(df, genre, min_rating, max_runtime, min_year)
    except ValueError as e:
        return pd.DataFrame(), str(e)

    if filtered.empty:
        return pd.DataFrame(), (
            "No movies matched your constraints. "
            "Try lowering the minimum rating, increasing the runtime limit, "
            "or choosing an earlier release year."
        )

    filtered = compute_heuristic_score(filtered, genre)
    filtered = cluster_movies(filtered, n_clusters=min(3, len(filtered)))
    filtered = predict_ratings(filtered)
    filtered = add_explanations(filtered, genre)
    filtered = filtered.sort_values("heuristic_score", ascending=False)

    top = filtered.head(top_n).reset_index(drop=True)
    return top, None
