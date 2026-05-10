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
    """
    Load movies.csv with UTF-8 encoding (latin-1 fallback for Windows-saved files).
    Always sanitizes column names and numeric types after loading.
    """
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding="latin-1")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Make sure movies.csv exists inside the dataset/ folder."
        )

    df = sanitize_dataframe(df)
    return df


def get_recommendations(genre, min_rating, max_runtime, min_year, top_n=8, country="Any"):
    """
    Full AI recommendation pipeline:

      Step 1 — Load dataset
      Step 2 — CSP filtering (hard constraints)
      Step 3 — Heuristic scoring (soft ranking)
      Step 4 — K-Means clustering (grouping)
      Step 5 — ANN rating prediction
      Step 6 — Explainable AI (natural-language reasons)
      Step 7 — Sort by heuristic score and return top N

    Returns (DataFrame, error_message).
    If error_message is not None, the DataFrame will be empty.
    """
    try:
        df = load_movies()
    except Exception as e:
        return pd.DataFrame(), f"Could not load movies.csv — {e}"

    try:
        filtered = apply_csp_filter(df, genre, min_rating, max_runtime, min_year, country)
    except ValueError as e:
        return pd.DataFrame(), str(e)
    except Exception as e:
        return pd.DataFrame(), f"Filter error — {e}"

    if filtered.empty:
        return pd.DataFrame(), (
            "No movies matched your filters. "
            "Try lowering the minimum rating, raising the runtime limit, "
            "choosing an earlier year, or selecting 'Any' for genre / country."
        )

    try:
        filtered = compute_heuristic_score(filtered, genre)
    except Exception as e:
        return pd.DataFrame(), f"Heuristic scoring failed — {e}"

    try:
        filtered = cluster_movies(filtered, n_clusters=min(3, len(filtered)))
    except Exception as e:
        filtered["cluster"] = 0

    try:
        filtered = predict_ratings(filtered)
    except Exception as e:
        filtered["ann_predicted_rating"] = filtered["rating"]

    try:
        filtered = add_explanations(filtered, genre)
    except Exception as e:
        filtered["explanation"] = "Recommended based on your preferences."

    filtered = filtered.sort_values("heuristic_score", ascending=False)
    top = filtered.head(top_n).reset_index(drop=True)
    return top, None


def get_country_list():
    """
    Return a sorted list of all countries in the dataset for the sidebar filter.
    Falls back to an empty list if the dataset cannot be loaded.
    """
    try:
        df = load_movies()
        if "country" not in df.columns:
            return []
        countries = sorted(df["country"].dropna().astype(str).unique().tolist())
        return [c for c in countries if c.strip() and c != "Unknown"]
    except Exception:
        return []
