import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_movies(df, n_clusters=3):
    df = df.copy()

    if len(df) < n_clusters:
        df["cluster"] = 0
        return df

    features = df[["rating", "runtime", "popularity"]].astype(float).values

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    k = min(n_clusters, len(df))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    df["cluster"] = labels
    return df


def get_cluster_label(cluster_id):
    labels = {
        0: "Hidden Gems",
        1: "Mainstream Hits",
        2: "Cult Classics",
        3: "Blockbusters",
        4: "Art House"
    }
    return labels.get(int(cluster_id), f"Group {cluster_id}")
