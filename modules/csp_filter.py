import pandas as pd


def sanitize_dataframe(df):
    # Strip single quotes, double quotes, and whitespace from ALL column names.
    # Some CSV editors (especially on Windows) wrap column names in single quotes
    # like 'rating' instead of rating — this causes KeyError when accessing df["rating"].
    df = df.copy()
    df.columns = [str(c).strip().strip("'").strip('"').strip() for c in df.columns]

    # Also clean up string cell values (genres, titles, etc.)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.strip("'").str.strip('"').str.strip()

    # Make sure numeric columns are numeric
    for col in ["rating", "runtime", "year", "popularity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rating"] = df.get("rating", pd.Series(dtype=float)).fillna(0.0)
    df["runtime"] = df.get("runtime", pd.Series(dtype=float)).fillna(999)
    df["year"] = df.get("year", pd.Series(dtype=float)).fillna(0)
    df["popularity"] = df.get("popularity", pd.Series(dtype=float)).fillna(0)

    return df


def apply_csp_filter(df, genre, min_rating, max_runtime, min_year):
    df = sanitize_dataframe(df)

    required_cols = ["title", "genre", "year", "rating", "runtime", "popularity"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Your movies.csv is missing these columns: {missing}\n"
            f"Columns found in file: {list(df.columns)}\n"
            "Please make sure the CSV header row has no extra quotes or spaces."
        )

    filtered = df.copy()

    if genre != "Any":
        filtered = filtered[filtered["genre"] == genre]

    filtered = filtered[filtered["rating"] >= min_rating]
    filtered = filtered[filtered["runtime"] <= max_runtime]
    filtered = filtered[filtered["year"] >= min_year]

    return filtered.reset_index(drop=True)


def get_constraint_summary(genre, min_rating, max_runtime, min_year):
    lines = []
    if genre != "Any":
        lines.append(f"Genre = {genre}")
    lines.append(f"Rating ≥ {min_rating}")
    lines.append(f"Runtime ≤ {max_runtime} min")
    lines.append(f"Year ≥ {min_year}")
    return lines
