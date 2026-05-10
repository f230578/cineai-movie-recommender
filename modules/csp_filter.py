import pandas as pd


def sanitize_dataframe(df):
    """
    Strip hidden single/double quotes from column names and string cells.
    Some CSV editors on Windows wrap column names like 'rating' instead of rating.
    Also coerces numeric columns so they are always usable as numbers.
    """
    df = df.copy()
    df.columns = [str(c).strip().strip("'").strip('"').strip() for c in df.columns]

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.strip("'").str.strip('"').str.strip()

    for col in ["rating", "runtime", "year", "popularity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rating"]     = df.get("rating",     pd.Series(dtype=float)).fillna(0.0)
    df["runtime"]    = df.get("runtime",    pd.Series(dtype=float)).fillna(999)
    df["year"]       = df.get("year",       pd.Series(dtype=float)).fillna(0)
    df["popularity"] = df.get("popularity", pd.Series(dtype=float)).fillna(0)

    # Fill country if missing
    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unknown")

    return df


def apply_csp_filter(df, genre, min_rating, max_runtime, min_year, country="Any"):
    """
    CSP (Constraint Satisfaction Problem) filtering.

    Each user preference becomes a hard constraint:
      - genre      : movie genre must match (unless 'Any')
      - min_rating : movie rating must be >= this value
      - max_runtime: movie runtime must be <= this value
      - min_year   : movie release year must be >= this value
      - country    : production country must match (unless 'Any')

    A movie is only kept if ALL active constraints are satisfied simultaneously.
    This is the core of a CSP — every variable (movie) must satisfy every constraint.
    """
    df = sanitize_dataframe(df)

    required_cols = ["title", "genre", "year", "rating", "runtime", "popularity"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Your movies.csv is missing these columns: {missing}\n"
            f"Columns found: {list(df.columns)}\n"
            "Check that the CSV header has no extra quotes or spaces."
        )

    filtered = df.copy()

    # Constraint 1 — Genre
    if genre and genre != "Any":
        filtered = filtered[filtered["genre"].str.strip() == genre.strip()]

    # Constraint 2 — Minimum rating
    filtered = filtered[filtered["rating"] >= float(min_rating)]

    # Constraint 3 — Maximum runtime
    filtered = filtered[filtered["runtime"] <= float(max_runtime)]

    # Constraint 4 — Minimum release year
    filtered = filtered[filtered["year"] >= float(min_year)]

    # Constraint 5 — Country of production
    if country and country != "Any" and "country" in filtered.columns:
        filtered = filtered[filtered["country"].str.strip() == country.strip()]

    return filtered.reset_index(drop=True)


def get_constraint_summary(genre, min_rating, max_runtime, min_year, country="Any"):
    """Return a list of human-readable active constraint strings for the UI."""
    lines = []
    if genre and genre != "Any":
        lines.append(f"Genre = {genre}")
    lines.append(f"Rating ≥ {min_rating}")
    lines.append(f"Runtime ≤ {max_runtime} min")
    lines.append(f"Year ≥ {min_year}")
    if country and country != "Any":
        lines.append(f"Country = {country}")
    return lines
