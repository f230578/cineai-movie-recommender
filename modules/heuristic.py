import pandas as pd


def compute_heuristic_score(df, preferred_genre):
    scores = []
    current_year = 2024

    for _, row in df.iterrows():
        score = 0.0

        # Genre match bonus (30 pts if match, 10 pts otherwise)
        genre_score = 30.0 if row["genre"] == preferred_genre else 10.0
        score += genre_score

        # IMDb rating scaled to 40 pts max
        rating_score = (float(row["rating"]) / 10.0) * 40.0
        score += rating_score

        # Popularity scaled to 20 pts max
        popularity_score = (float(row["popularity"]) / 100.0) * 20.0
        score += popularity_score

        # Recency bonus: newer = more points
        years_old = current_year - int(row["year"])
        if years_old <= 3:
            recency_bonus = 10.0
        elif years_old <= 10:
            recency_bonus = 5.0
        else:
            recency_bonus = 0.0
        score += recency_bonus

        scores.append(round(score, 2))

    df = df.copy()
    df["heuristic_score"] = scores
    return df
