import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import sys
import os
import html
import warnings

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules"))

try:
    from recommender import get_recommendations, load_movies, get_country_list
    from csp_filter import get_constraint_summary
    from search import get_search_results
    from clustering import get_cluster_label
    MODULES_OK = True
    _import_err_msg = ""
except Exception as _import_err:
    MODULES_OK = False
    _import_err_msg = str(_import_err)


# ════════════════════════════════════════════════════════════════
#  SESSION STATE — initialise once per browser session
#  Results are stored here so sorting widgets can re-order them
#  without re-running the expensive AI pipeline.
# ════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "results":        None,   # DataFrame returned by the pipeline
        "pipeline_error": None,   # error string if pipeline failed
        "sort_by":        "AI Score",
        "sort_dir":       "↓ High → Low",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CineAI — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --bg-deep:    #080a0f;
    --bg-card:    #10131a;
    --gold:       #c9a84c;
    --gold-light: #e8c97a;
    --red:        #c0392b;
    --text-main:  #e8e8e8;
    --text-muted: #8a8f9e;
    --border:     rgba(201,168,76,0.18);
    --radius:     14px;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-deep) !important;
    font-family: 'Outfit', sans-serif;
    color: var(--text-main);
}
[data-testid="stSidebar"] {
    background-color: #0c0f17 !important;
    border-right: 1px solid var(--border);
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 52px 24px 36px;
    background: radial-gradient(ellipse at 50% 0%, rgba(201,168,76,0.10) 0%, transparent 65%);
    border-bottom: 1px solid var(--border);
    margin-bottom: 36px;
}
.hero-eyebrow {
    font-weight: 500; font-size: 0.78rem; letter-spacing: 0.28em;
    text-transform: uppercase; color: var(--gold); margin-bottom: 12px;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3rem, 7vw, 5.5rem);
    letter-spacing: 0.06em; line-height: 1; color: #fff; margin: 0 0 16px;
}
.hero-title span { color: var(--gold); }
.hero-sub {
    color: var(--text-muted); font-size: 1rem; font-weight: 300;
    max-width: 520px; margin: 0 auto; line-height: 1.6;
}

/* ── Sidebar ── */
.sidebar-section {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.2em;
    text-transform: uppercase; color: var(--gold);
    margin: 24px 0 10px; padding-bottom: 6px; border-bottom: 1px solid var(--border);
}
.constraint-chip {
    display: inline-block;
    background: rgba(201,168,76,0.10); border: 1px solid rgba(201,168,76,0.25);
    color: var(--gold-light); border-radius: 20px;
    padding: 3px 11px; font-size: 0.75rem; margin: 3px 2px;
}

/* ── Stat strip ── */
.stat-strip { display: flex; gap: 12px; margin: 0 0 28px; flex-wrap: wrap; }
.stat-box {
    flex: 1; min-width: 110px;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 16px 18px; text-align: center;
}
.stat-number { font-family: 'Bebas Neue', sans-serif; font-size: 2.2rem; color: var(--gold); line-height: 1; }
.stat-label  { color: var(--text-muted); font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase; margin-top: 4px; }

/* ── Section headings ── */
.section-heading {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.6rem;
    letter-spacing: 0.1em; color: #fff; margin: 0 0 6px;
}
.section-heading span { color: var(--gold); }

/* ── Sort bar ── */
.sort-info {
    display: inline-block;
    background: rgba(201,168,76,0.08); border: 1px solid rgba(201,168,76,0.2);
    border-radius: 8px; padding: 6px 14px; font-size: 0.8rem;
    color: var(--gold-light); margin-bottom: 16px; letter-spacing: 0.03em;
}

/* ── Movie grid (CSS grid — no Streamlit columns needed) ── */
.movie-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-bottom: 8px;
}
@media (max-width: 820px) { .movie-grid { grid-template-columns: 1fr; } }

/* ── Movie card ── */
.movie-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--gold);
    border-radius: var(--radius);
    padding: 20px 20px 16px 20px;
    display: flex; flex-direction: column; gap: 8px;
}
.card-header   { display: flex; align-items: flex-start; gap: 14px; }
.card-rank     {
    font-family: 'Bebas Neue', sans-serif; font-size: 2.2rem;
    color: rgba(201,168,76,0.28); line-height: 1;
    flex-shrink: 0; letter-spacing: 0.05em; padding-top: 2px; min-width: 40px;
}
.card-title-block { flex: 1; }
.card-title    { font-weight: 700; font-size: 1.05rem; color: #fff; line-height: 1.3; margin-bottom: 2px; }
.card-year     { font-size: 0.82rem; color: var(--gold); font-weight: 500; margin-left: 6px; }
.card-stars    { color: var(--gold); font-size: 0.85rem; letter-spacing: 0.06em; }

/* ── Pills ── */
.meta-pills { display: flex; flex-wrap: wrap; gap: 6px; }
.pill       { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.09); border-radius: 20px; padding: 3px 11px; font-size: 0.76rem; color: var(--text-muted); }
.pill b     { color: var(--text-main); }
.pill-gold  { background: rgba(201,168,76,0.12); border-color: rgba(201,168,76,0.3);  color: var(--gold-light); }
.pill-green { background: rgba(46,213,115,0.08); border-color: rgba(46,213,115,0.2);  color: #6ee7a0; }
.pill-blue  { background: rgba(26,115,232,0.10); border-color: rgba(26,115,232,0.3);  color: #7ab3f5; }

/* ── Score bar ── */
.score-row   { display: flex; align-items: center; gap: 10px; }
.score-track { flex: 1; height: 4px; background: rgba(255,255,255,0.07); border-radius: 4px; overflow: hidden; }
.score-fill  { height: 100%; border-radius: 4px; background: linear-gradient(90deg, var(--gold), var(--gold-light)); }
.score-num   { font-family: 'Bebas Neue', sans-serif; font-size: 1rem; color: var(--gold); letter-spacing: 0.05em; white-space: nowrap; }

/* ── Explanation ── */
.card-explain {
    color: var(--text-muted); font-size: 0.82rem; font-style: italic;
    line-height: 1.5; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.05);
}

/* ── Charts ── */
.chart-card  { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; margin-bottom: 16px; }
.chart-title { font-family: 'Bebas Neue', sans-serif; font-size: 1.1rem; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 14px; }

/* ── Empty / error ── */
.empty-state { text-align: center; padding: 60px 20px; border: 1px dashed rgba(201,168,76,0.2); border-radius: var(--radius); margin: 20px 0; }
.empty-icon  { font-size: 3rem; margin-bottom: 12px; }
.empty-title { font-family: 'Bebas Neue', sans-serif; font-size: 1.6rem; letter-spacing: 0.08em; color: var(--text-main); margin-bottom: 8px; }
.empty-sub   { color: var(--text-muted); font-size: 0.92rem; line-height: 1.6; }

.err-banner  {
    background: rgba(192,57,43,0.12); border: 1px solid rgba(192,57,43,0.35);
    border-left: 4px solid var(--red); border-radius: var(--radius);
    padding: 16px 20px; color: #f0a89e; font-size: 0.9rem; line-height: 1.6; margin-bottom: 20px;
}
.err-banner b { color: #ff8a80; }

/* ── Button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--gold) 0%, #a07830 100%) !important;
    color: #0a0c10 !important; font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important; font-size: 1rem !important;
    letter-spacing: 0.06em !important; border: none !important;
    border-radius: 10px !important; padding: 14px 28px !important;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88 !important; }
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-family: 'Outfit', sans-serif !important; font-size: 0.82rem !important;
    font-weight: 500 !important; color: var(--text-muted) !important; letter-spacing: 0.04em !important;
}
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
div[data-testid="stSpinner"] p { color: var(--gold) !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: #2a2e3a; border-radius: 4px; }

.footer {
    text-align: center; padding: 32px 0 16px;
    color: rgba(138,143,158,0.4); font-size: 0.72rem;
    letter-spacing: 0.12em; text-transform: uppercase;
    border-top: 1px solid rgba(255,255,255,0.04); margin-top: 48px;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════

def safe_load_genres():
    if not MODULES_OK:
        return ["Any","Action","Comedy","Drama","Horror","Romance","Sci-Fi","Thriller"]
    try:
        df = load_movies()
        genres = sorted(df["genre"].dropna().astype(str).unique().tolist())
        return ["Any"] + [g for g in genres if g.strip()]
    except Exception:
        return ["Any","Action","Comedy","Drama","Horror","Romance","Sci-Fi","Thriller"]


def safe_load_countries():
    if not MODULES_OK:
        return ["Any"]
    try:
        return ["Any"] + get_country_list()
    except Exception:
        return ["Any"]


def star_string(rating):
    try:
        filled = max(0, min(5, round(float(rating) / 2)))
        return "★" * filled + "☆" * (5 - filled)
    except Exception:
        return "☆☆☆☆☆"


def score_pct(score):
    try:
        return min(100.0, max(0.0, float(score)))
    except Exception:
        return 0.0


def dark_fig(w=6, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#10131a")
    ax.set_facecolor("#10131a")
    ax.tick_params(colors="#8a8f9e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2e3a")
    return fig, ax


def apply_sort(df: pd.DataFrame, sort_by: str, sort_dir: str) -> pd.DataFrame:
    """
    Re-order the results DataFrame by the chosen column and direction.
    Works purely on the cached DataFrame — never re-runs the AI pipeline.
    """
    col_map = {
        "AI Score":   "heuristic_score",
        "Rating":     "rating",
        "Runtime":    "runtime",
        "Year":       "year",
        "Popularity": "popularity",
    }
    col = col_map.get(sort_by, "heuristic_score")
    ascending = (sort_dir == "↑ Low → High")

    if col not in df.columns:
        return df
    try:
        out = df.copy()
        out[col] = pd.to_numeric(out[col], errors="coerce")
        return out.sort_values(col, ascending=ascending, na_position="last").reset_index(drop=True)
    except Exception:
        return df


def build_card(rank, title, year, genre_v, rating, runtime,
               popularity, h_score, cluster_nm, ann_str,
               country_v, explanation, stars, bar_pct) -> str:
    """Return one movie card as an escaped HTML string."""
    e = html.escape  # shorthand

    return f"""
<div class="movie-card">
  <div class="card-header">
    <div class="card-rank">{rank:02d}</div>
    <div class="card-title-block">
      <div class="card-title">{e(str(title))}<span class="card-year">{int(year)}</span></div>
      <div class="card-stars">{e(str(stars))}</div>
    </div>
  </div>
  <div class="meta-pills">
    <span class="pill">🎭 <b>{e(str(genre_v))}</b></span>
    <span class="pill">⭐ <b>{float(rating):.1f}</b> / 10</span>
    <span class="pill">⏱️ <b>{int(runtime)} min</b></span>
    <span class="pill">🔥 <b>{int(popularity)}</b></span>
    <span class="pill pill-blue">🌍 <b>{e(str(country_v))}</b></span>
    <span class="pill pill-gold">🤖 {e(str(ann_str))}</span>
    <span class="pill pill-green">📂 {e(str(cluster_nm))}</span>
  </div>
  <div class="score-row">
    <div class="score-track">
      <div class="score-fill" style="width:{bar_pct:.1f}%;"></div>
    </div>
    <div class="score-num">AI {float(h_score):.0f}</div>
  </div>
  <div class="card-explain">💡 {e(str(explanation))}</div>
</div>"""


def render_cards(df: pd.DataFrame):
    """Build the full 2-column CSS grid of movie cards and render in one st.markdown call."""
    blocks = []
    for i, row in df.reset_index(drop=True).iterrows():
        try:
            ann_raw = row.get("ann_predicted_rating", None)
            blocks.append(build_card(
                rank        = i + 1,
                title       = row.get("title",           "Unknown"),
                year        = row.get("year",            0),
                genre_v     = row.get("genre",           "—"),
                rating      = float(row.get("rating",    0)),
                runtime     = float(row.get("runtime",   0)),
                popularity  = float(row.get("popularity",0)),
                h_score     = float(row.get("heuristic_score", 0)),
                cluster_nm  = get_cluster_label(int(row.get("cluster", 0))),
                ann_str     = f"{float(ann_raw):.1f}" if ann_raw is not None else "—",
                country_v   = row.get("country", "—") if "country" in row.index else "—",
                explanation = row.get("explanation", "Recommended based on your preferences."),
                stars       = star_string(float(row.get("rating", 0))),
                bar_pct     = score_pct(row.get("heuristic_score", 0)),
            ))
        except Exception as err:
            blocks.append(
                f'<div class="movie-card" style="border-left-color:#c0392b;">'
                f'Card #{i+1} error: {html.escape(str(err))}</div>'
            )

    st.markdown(
        '<div class="movie-grid">' + "".join(blocks) + '</div>',
        unsafe_allow_html=True,
    )


def render_stats(df: pd.DataFrame):
    try:
        avg_r   = round(float(df["rating"].astype(float).mean()), 1)
        avg_pop = int(df["popularity"].astype(float).mean())
        yr_min  = int(df["year"].min())
        yr_max  = int(df["year"].max())
        yr_span = f"{yr_min} – {yr_max}" if yr_min != yr_max else str(yr_min)
        n_gen   = df["genre"].nunique()
    except Exception:
        avg_r, avg_pop, yr_span, n_gen = "—", "—", "—", "—"

    st.markdown(f"""
<div class="stat-strip">
  <div class="stat-box"><div class="stat-number">{len(df)}</div><div class="stat-label">Picks for You</div></div>
  <div class="stat-box"><div class="stat-number">{avg_r}</div><div class="stat-label">Avg Rating</div></div>
  <div class="stat-box"><div class="stat-number">{avg_pop}</div><div class="stat-label">Avg Popularity</div></div>
  <div class="stat-box"><div class="stat-number" style="font-size:1.3rem;padding-top:8px;">{yr_span}</div><div class="stat-label">Year Range</div></div>
  <div class="stat-box"><div class="stat-number">{n_gen}</div><div class="stat-label">Genres</div></div>
</div>""", unsafe_allow_html=True)


def render_charts(df: pd.DataFrame):
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown('<div class="chart-card"><div class="chart-title">AI SCORE vs IMDB RATING</div>',
                    unsafe_allow_html=True)
        try:
            fig, ax = dark_fig(5.5, 3.8)
            xs = df["rating"].astype(float).values
            ys = df["heuristic_score"].astype(float).values
            ax.scatter(xs, ys, c=ys, cmap="YlOrRd", s=130, zorder=5,
                       edgecolors="#2a2e3a", linewidths=0.8,
                       vmin=max(ys.min()-5,0), vmax=min(ys.max()+5,100))
            for _, r in df.iterrows():
                ax.annotate(str(r["title"])[:13],
                            (float(r["rating"]), float(r["heuristic_score"])),
                            fontsize=6.2, color="#8a8f9e",
                            textcoords="offset points", xytext=(5,3))
            ax.set_xlabel("IMDb Rating", color="#8a8f9e", fontsize=8)
            ax.set_ylabel("AI Score",    color="#8a8f9e", fontsize=8)
            ax.grid(True, color="#1e2230", linewidth=0.6, zorder=0)
            st.pyplot(fig); plt.close(fig)
        except Exception as e:
            st.caption(f"Chart unavailable: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card"><div class="chart-title">CLUSTER BREAKDOWN</div>',
                    unsafe_allow_html=True)
        try:
            cc = df["cluster"].value_counts()
            if cc.empty: raise ValueError("empty")
            labels  = [get_cluster_label(int(c)) for c in cc.index]
            palette = ["#c9a84c","#e8c97a","#2d6a4f","#1a73e8","#9c27b0","#c0392b","#16a085"]
            fig2, ax2 = dark_fig(5.5, 3.8)
            wedges, _, ats = ax2.pie(cc.values, labels=None, autopct="%1.0f%%",
                                     colors=palette[:len(cc)], startangle=90,
                                     wedgeprops={"linewidth":2,"edgecolor":"#10131a"},
                                     pctdistance=0.78)
            for at in ats: at.set_fontsize(8); at.set_color("white")
            ax2.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5,-0.18),
                       ncol=3, fontsize=7, frameon=False, labelcolor="#8a8f9e")
            st.pyplot(fig2); plt.close(fig2)
        except Exception as e:
            st.caption(f"Chart unavailable: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2, gap="medium")

    with c3:
        st.markdown('<div class="chart-card"><div class="chart-title">RUNTIME DISTRIBUTION</div>',
                    unsafe_allow_html=True)
        try:
            rt = df["runtime"].astype(float).values
            if len(rt) < 2: raise ValueError("too few")
            fig3, ax3 = dark_fig(5.5, 3.2)
            ax3.hist(rt, bins=min(len(rt),8), color="#c9a84c", alpha=0.8,
                     edgecolor="#10131a", linewidth=1)
            ax3.set_xlabel("Runtime (min)", color="#8a8f9e", fontsize=8)
            ax3.set_ylabel("Count",         color="#8a8f9e", fontsize=8)
            ax3.grid(True, axis="y", color="#1e2230", linewidth=0.6)
            st.pyplot(fig3); plt.close(fig3)
        except Exception as e:
            st.caption(f"Chart unavailable: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-card"><div class="chart-title">RATING OVERVIEW</div>',
                    unsafe_allow_html=True)
        try:
            rb = df.copy().reset_index(drop=True)
            rb["short"] = rb["title"].astype(str).str[:16]
            rb = rb.sort_values("rating", ascending=True)
            fig4, ax4 = dark_fig(5.5, 3.2)
            bcolors = ["#c9a84c" if v >= 7.5 else "#5a6070"
                       for v in rb["rating"].astype(float)]
            ax4.barh(rb["short"], rb["rating"].astype(float),
                     color=bcolors, edgecolor="#10131a", linewidth=0.8, height=0.65)
            ax4.set_xlabel("IMDb Rating", color="#8a8f9e", fontsize=8)
            ax4.axvline(7.5, color="#c9a84c", linewidth=0.8, linestyle="--", alpha=0.5)
            ax4.tick_params(axis="y", labelsize=6.5, colors="#8a8f9e")
            ax4.grid(True, axis="x", color="#1e2230", linewidth=0.6)
            ax4.set_xlim(0, 10.5)
            st.pyplot(fig4); plt.close(fig4)
        except Exception as e:
            st.caption(f"Chart unavailable: {e}")
        st.markdown("</div>", unsafe_allow_html=True)


def render_graph(df: pd.DataFrame):
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Similarity <span>Graph</span></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8a8f9e;font-size:0.85rem;margin-bottom:16px;">'
        'Movies are connected when they share a genre or have similar ratings. '
        'Gold nodes are your top matches.</p>',
        unsafe_allow_html=True
    )
    try:
        path, graph = get_search_results(df, "BFS")
        if not graph or len(graph.nodes) == 0:
            st.info("Not enough results for a graph. Try requesting more movies.")
            return
        fig5, ax5 = plt.subplots(figsize=(12, 5))
        fig5.patch.set_facecolor("#10131a")
        ax5.set_facecolor("#10131a")
        pos  = nx.spring_layout(graph, seed=42, k=2.2)
        ncol = ["#c9a84c" if n in path else "#2a2e3a" for n in graph.nodes()]
        nsz  = [820 if n in path else 500 for n in graph.nodes()]
        nx.draw_networkx_edges(graph, pos, ax=ax5, edge_color="#2a2e3a", alpha=0.7, width=1.0)
        nx.draw_networkx_nodes(graph, pos, ax=ax5, node_color=ncol, node_size=nsz, alpha=0.95)
        nx.draw_networkx_labels(graph, pos, ax=ax5, font_size=6, font_color="#fff", font_weight="bold")
        ax5.axis("off")
        ax5.legend(
            handles=[mpatches.Patch(color="#c9a84c", label="Top Match"),
                     mpatches.Patch(color="#2a2e3a", label="Related Movie")],
            loc="lower right", frameon=False, labelcolor="#8a8f9e", fontsize=8
        )
        st.pyplot(fig5); plt.close(fig5)
    except Exception as ge:
        st.info(f"Graph could not be rendered: {ge}")


# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 8px;'>
      <div style='font-family:"Bebas Neue",sans-serif;font-size:1.8rem;
                  letter-spacing:0.12em;color:#c9a84c;'>CINE AI</div>
      <div style='font-size:0.7rem;letter-spacing:0.2em;color:#8a8f9e;
                  text-transform:uppercase;margin-top:2px;'>Smart Recommender</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Preferences</div>', unsafe_allow_html=True)

    all_genres    = safe_load_genres()
    all_countries = safe_load_countries()

    genre   = st.selectbox("Genre 🎭",   all_genres,
                           help="Filter by genre. 'Any' includes all genres.")
    country = st.selectbox("Country 🌍", all_countries,
                           help="Filter by primary production country.")

    min_rating  = st.slider("Minimum Rating ⭐",  1.0, 10.0, 6.0, 0.1)
    max_runtime = st.slider("Max Runtime ⏱️",      60,  338,  180,  5)
    min_year    = st.slider("From Year 📅",        1916, 2017, 1990,  1)
    top_n       = st.slider("Results 🎬",            3,   20,    8,   1)

    st.markdown('<div class="sidebar-section">Active Filters</div>', unsafe_allow_html=True)
    if not MODULES_OK:
        st.warning("Modules failed to load. Check your installation.")
    else:
        try:
            chips = get_constraint_summary(genre, min_rating, max_runtime, min_year, country)
            st.markdown(
                "".join(f'<span class="constraint-chip">{c}</span>' for c in chips),
                unsafe_allow_html=True,
            )
        except Exception:
            st.caption("Filters apply on next search.")

    st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  HERO
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">AI-Powered Discovery</div>
  <div class="hero-title">Find Your Next<br><span>Favourite Film</span></div>
  <div class="hero-sub">
    Set your preferences on the left, then hit Discover.
    Our AI scans thousands of movies to surface the ones you will love most.
  </div>
</div>
""", unsafe_allow_html=True)

if not MODULES_OK:
    st.markdown(f"""
    <div class="err-banner">
      <b>⚠️ Application failed to start.</b><br>
      <code style="font-size:0.8rem;">{html.escape(_import_err_msg)}</code><br><br>
      Run <code>pip install -r requirements.txt</code> and ensure
      <code>modules/</code> exists next to <code>app.py</code>.
    </div>""", unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════════
#  DISCOVER BUTTON  +  SORT CONTROLS  (always shown if results exist)
#
#  HOW SORTING WORKS:
#  ─────────────────
#  1. User presses "Discover Movies" → pipeline runs → raw results stored
#     in st.session_state["results"].  The button press sets run=True once.
#
#  2. User changes a sort widget → Streamlit reruns the script automatically.
#     run is now False (button not pressed), but st.session_state["results"]
#     still holds the previous results.  apply_sort() re-orders them and
#     render_cards() draws them instantly — no pipeline call is made.
#
#  3. User changes a sidebar filter → they press Discover again → fresh
#     pipeline run, new results stored, sort resets to current widget values.
# ════════════════════════════════════════════════════════════════

left_space, center_btn, right_space = st.columns([2, 2, 2])

with center_btn:
    run = st.button(
        "🎬  Discover Movies",
        use_container_width=True
    )

# ── Run pipeline only when button is pressed ─────────────────────────────────
if run:
    with st.spinner("Scanning the library and ranking your matches…"):
        try:
            res, err = get_recommendations(
                genre, min_rating, max_runtime, min_year, top_n, country
            )
            st.session_state["results"]        = res
            st.session_state["pipeline_error"] = err
        except MemoryError:
            st.session_state["results"]        = pd.DataFrame()
            st.session_state["pipeline_error"] = "Dataset too large. Try narrowing your filters."
        except FileNotFoundError as e:
            st.session_state["results"]        = pd.DataFrame()
            st.session_state["pipeline_error"] = f"Dataset not found — {e}"
        except KeyError as e:
            st.session_state["results"]        = pd.DataFrame()
            st.session_state["pipeline_error"] = f"Missing column in movies.csv: {e}"
        except Exception as e:
            st.session_state["results"]        = pd.DataFrame()
            st.session_state["pipeline_error"] = f"Unexpected error — {e}"

# ── Read current state ───────────────────────────────────────────────────────
stored_results = st.session_state.get("results", None)
stored_error   = st.session_state.get("pipeline_error", None)

# ── Nothing searched yet ─────────────────────────────────────────────────────
if stored_results is None:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">🎞️</div>
      <div class="empty-title">Ready When You Are</div>
      <div class="empty-sub">
        Adjust your preferences in the sidebar,<br>
        then click <b>Discover Movies</b> to get your personalised picks.
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="footer">CineAI · Hybrid AI Recommendation System</div>',
                unsafe_allow_html=True)
    st.stop()

# ── Pipeline error ───────────────────────────────────────────────────────────
if stored_error:
    st.markdown(
        f'<div class="err-banner"><b>⚠️ Could not generate recommendations</b>'
        f'<br>{html.escape(str(stored_error))}</div>',
        unsafe_allow_html=True,
    )

if stored_results is None or stored_results.empty:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">🔍</div>
      <div class="empty-title">No Matches Found</div>
      <div class="empty-sub">
        Try lowering the minimum rating, raising the max runtime,<br>
        choosing an earlier year, or selecting <b>Any</b> for genre or country.
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="footer">CineAI · Hybrid AI Recommendation System</div>',
                unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════════
#  SORT CONTROLS
#  ─────────────────────────────────────────────────────────────
#  Widgets are bound to session_state keys ("sort_by", "sort_dir").
#  Changing either widget triggers an automatic Streamlit rerun.
#  The pipeline is NOT called again — only apply_sort() runs.
# ════════════════════════════════════════════════════════════════

SORT_OPTIONS = ["AI Score", "Rating", "Runtime", "Year", "Popularity"]

st.markdown('<div class="section-heading">Sort &amp; <span>Results</span></div>',
            unsafe_allow_html=True)

sc1, sc2, sc3 = st.columns([3, 2, 1])

with sc1:
    st.radio(
        "Sort by",
        SORT_OPTIONS,
        horizontal=True,
        key="sort_by",          # ← bound to session_state["sort_by"]
        help=(
            "AI Score — overall match rank (default)\n"
            "Rating   — IMDb rating\n"
            "Runtime  — length in minutes\n"
            "Year     — release year\n"
            "Popularity — audience popularity index"
        ),
    )

with sc2:
    st.radio(
        "Direction",
        ["↓ High → Low", "↑ Low → High"],
        horizontal=True,
        key="sort_dir",         # ← bound to session_state["sort_dir"]
    )

with sc3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↺  Reset"):
        st.session_state["sort_by"]  = "AI Score"
        st.session_state["sort_dir"] = "↓ High → Low"
        st.rerun()              # ← immediately re-render with reset values

# Apply sort to the stored results using current widget values
sorted_results = apply_sort(
    stored_results,
    st.session_state["sort_by"],
    st.session_state["sort_dir"],
)

st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  RESULTS  — stats · cards · charts · graph · table
# ════════════════════════════════════════════════════════════════

render_stats(sorted_results)

st.markdown('<div class="section-heading">Your <span>Recommendations</span></div>',
            unsafe_allow_html=True)
st.markdown(
    f'<div class="sort-info">Sorted by: <b>{st.session_state["sort_by"]}</b>'
    f' &nbsp;·&nbsp; {st.session_state["sort_dir"]}</div>',
    unsafe_allow_html=True,
)

render_cards(sorted_results)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-heading">Visual <span>Insights</span></div>',
            unsafe_allow_html=True)
render_charts(sorted_results)

render_graph(sorted_results)

with st.expander("📋  Full Results Table"):
    try:
        show_cols = ["title","genre","year","rating","runtime","popularity",
                     "country","heuristic_score","cluster","ann_predicted_rating"]
        disp = sorted_results[[c for c in show_cols if c in sorted_results.columns]].copy()
        disp.index = range(1, len(disp) + 1)
        disp.index.name = "#"
        st.dataframe(disp, use_container_width=True)
    except Exception as te:
        st.warning(f"Table could not be rendered: {te}")

st.markdown('<div class="footer">CineAI · Hybrid AI Recommendation System</div>',
            unsafe_allow_html=True)
