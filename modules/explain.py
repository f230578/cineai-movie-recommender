def generate_explanation(row, preferred_genre):
    parts = []

    if row["genre"] == preferred_genre:
        parts.append(f"it matches your preferred genre ({preferred_genre})")
    else:
        parts.append(f"it is a highly rated {row['genre']} film worth exploring")

    rating = float(row["rating"])
    if rating >= 8.5:
        parts.append(f"it has an exceptional IMDb rating of {rating}")
    elif rating >= 7.5:
        parts.append(f"it has a strong IMDb rating of {rating}")
    else:
        parts.append(f"it has a decent IMDb rating of {rating}")

    if float(row["popularity"]) >= 90:
        parts.append("it is extremely popular among audiences")
    elif float(row["popularity"]) >= 75:
        parts.append("it is well-liked by many viewers")

    current_year = 2024
    age = current_year - int(row["year"])
    if age <= 3:
        parts.append(f"it is a very recent release from {row['year']}")
    elif age <= 10:
        parts.append(f"it is a modern film from {row['year']}")

    if "heuristic_score" in row and float(row["heuristic_score"]) >= 80:
        parts.append("it scored highly in our AI ranking system")

    if not parts:
        return "This movie fits your preferences based on AI analysis."

    return "This movie was recommended because " + ", and ".join(parts) + "."


def add_explanations(df, preferred_genre):
    df = df.copy()
    df["explanation"] = df.apply(
        lambda row: generate_explanation(row, preferred_genre), axis=1
    )
    return df
