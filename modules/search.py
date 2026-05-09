import networkx as nx
import heapq
from collections import deque


def build_movie_graph(df):
    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_node(row["title"], genre=row["genre"], rating=float(row["rating"]))

    movies = df.to_dict("records")
    for i in range(len(movies)):
        for j in range(i + 1, len(movies)):
            m1 = movies[i]
            m2 = movies[j]
            same_genre = m1["genre"] == m2["genre"]
            close_rating = abs(float(m1["rating"]) - float(m2["rating"])) <= 0.5
            if same_genre or close_rating:
                weight = round(abs(float(m1["rating"]) - float(m2["rating"])), 2)
                G.add_edge(m1["title"], m2["title"], weight=weight)

    return G


def bfs_search(graph, start_node, max_visits=10):
    if start_node not in graph:
        return []
    visited = []
    queue = deque([start_node])
    seen = set([start_node])
    while queue and len(visited) < max_visits:
        node = queue.popleft()
        visited.append(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return visited


def dfs_search(graph, start_node, max_visits=10):
    if start_node not in graph:
        return []
    visited = []
    stack = [start_node]
    seen = set()
    while stack and len(visited) < max_visits:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            visited.append(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in seen:
                    stack.append(neighbor)
    return visited


def astar_search(graph, start_node, df, max_visits=10):
    if start_node not in graph:
        return []
    rating_map = dict(zip(df["title"], df["rating"].astype(float)))
    open_set = [(0.0, start_node)]
    g_score = {start_node: 0.0}
    visited_order = []
    seen = set()
    while open_set and len(visited_order) < max_visits:
        _, current = heapq.heappop(open_set)
        if current in seen:
            continue
        seen.add(current)
        visited_order.append(current)
        for neighbor in graph.neighbors(current):
            edge_w = graph[current][neighbor].get("weight", 0.5)
            tentative_g = g_score.get(current, 0) + edge_w
            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                heuristic = 10.0 - rating_map.get(neighbor, 5.0)
                heapq.heappush(open_set, (tentative_g + heuristic, neighbor))
    return visited_order


def get_search_results(df, algorithm="BFS"):
    if df.empty:
        return [], None
    graph = build_movie_graph(df)
    start = df.sort_values("rating", ascending=False).iloc[0]["title"]
    if algorithm == "BFS":
        path = bfs_search(graph, start)
    elif algorithm == "DFS":
        path = dfs_search(graph, start)
    else:
        path = astar_search(graph, start, df)
    return path, graph
