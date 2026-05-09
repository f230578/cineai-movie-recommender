import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import sys
import os
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modules"))

try:
    from recommender import get_recommendations, load_movies
    from csp_filter import get_constraint_summary
    from search import get_search_results
    from clustering import get_cluster_label
    MODULES_OK = True
    _import_err_msg = ""
except Exception as _import_err:
    MODULES_OK = False
    _import_err_msg = str(_import_err)


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
#  GLOBAL STYLES
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --bg-deep:    #080a0f;
    --bg-card:    #10131a;
    --bg-card2:   #161a24;
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

.hero {
    text-align: center;
    padding: 52px 24px 36px;
    background: radial-gradient(ellipse at 50% 0%, rgba(201,168,76,0.10) 0%, transparent 65%);
    border-bottom: 1px solid var(--border);
    margin-bottom: 36px;
}
.hero-eyebrow {
    font-family: 'Outfit', sans-serif;
    font-weight: 500;
    font-size: 0.78rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3rem, 7vw, 5.5rem);
    letter-spacing: 0.06em;
    line-height: 1;
    color: #ffffff;
    margin: 0 0 16px;
}
.hero-title span { color: var(--gold); }
.hero-sub {
    color: var(--text-muted);
    font-size: 1rem;
    font-weight: 300;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
}

.sidebar-section {
    font-family: 'Outfit', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    margin: 24px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}
.constraint-chip {
    display: inline-block;
    background: rgba(201,168,76,0.10);
    border: 1px solid rgba(201,168,76,0.25);
    color: var(--gold-light);
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 0.75rem;
    margin: 3px 2px;
}

.stat-strip { display: flex; gap: 12px; margin: 0 0 32px; }
.stat-box {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    text-align: center;
}
.stat-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: var(--gold);
    line-height: 1;
}
.stat-label {
    color: var(--text-muted);
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 4px;
}

.section-heading {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    letter-spacing: 0.1em;
    color: #fff;
    margin: 0 0 20px;
}
.section-heading span { color: var(--gold); }

.movie-card {
    position: relative;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--gold);
    border-radius: var(--radius);
    padding: 22px 22px 18px 72px;
    margin-bottom: 16px;
}
.movie-rank {
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 56px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: rgba(201,168,76,0.28);
    border-right: 1px solid var(--border);
    letter-spacing: 0.05em;
}
.movie-title-text {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: #fff;
    margin-bottom: 4px;
    line-height: 1.3;
}
.movie-year { font-size:0.85rem; color:var(--gold); font-weight:500; margin-left:6px; }
.meta-pills { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0 12px; }
.pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 2px 11px;
    font-size: 0.76rem;
    color: var(--text-muted);
}
.pill b { color: var(--text-main); }
.pill-gold { background:rgba(201,168,76,0.12); border-color:rgba(201,168,76,0.3); color:var(--gold-light); }
.pill-green { background:rgba(46,213,115,0.08); border-color:rgba(46,213,115,0.2); color:#6ee7a0; }

.score-bar-wrap { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.score-bar-track { flex:1; height:4px; background:rgba(255,255,255,0.07); border-radius:4px; overflow:hidden; }
.score-bar-fill { height:100%; border-radius:4px; background:linear-gradient(90deg,var(--gold),var(--gold-light)); }
.score-label { font-family:'Bebas Neue',sans-serif; font-size:1rem; color:var(--gold); letter-spacing:0.05em; white-space:nowrap; }

.explain {
    color: var(--text-muted);
    font-size: 0.83rem;
    font-style: italic;
    line-height: 1.5;
    border-top: 1px solid rgba(255,255,255,0.05);
    padding-top: 10px;
    margin-top: 2px;
}
.explain::before { content: "💡  "; }

.chart-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 16px;
}
.chart-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 14px;
}

.empty-state {
    text-align: center;
    padding: 60px 20px;
    border: 1px dashed rgba(201,168,76,0.2);
    border-radius: var(--radius);
    margin: 20px 0;
}
.empty-icon { font-size:3rem; margin-bottom:12px; }
.empty-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    letter-spacing: 0.08em;
    color: var(--text-main);
    margin-bottom: 8px;
}
.empty-sub { color:var(--text-muted); font-size:0.92rem; line-height:1.6; }

.err-banner {
    background: rgba(192,57,43,0.12);
    border: 1px solid rgba(192,57,43,0.35);
    border-left: 4px solid var(--red);
    border-radius: var(--radius);
    padding: 16px 20px;
    color: #f0a89e;
    font-size: 0.9rem;
    line-height: 1.6;
    margin-bottom: 20px;
}
.err-banner b { color: #ff8a80; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--gold) 0%, #a07830 100%) !important;
    color: #0a0c10 !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.06em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88 !important; }
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.04em !important;
}
div[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
div[data-testid="stSpinner"] p { color:var(--gold) !important; }

::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:var(--bg-deep); }
::-webkit-scrollbar-thumb { background:#2a2e3a; border-radius:4px; }

.footer {
    text-align: center;
    padding: 32px 0 16px;
    color: rgba(138,143,158,0.4);
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-top: 1px solid rgba(255,255,255,0.04);
    margin-top: 48px;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════

def safe_load_genres():
    if not MODULES_OK:
        return ["Any", "Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller"]
    try:
        df = load_movies()
        genres = sorted(df["genre"].dropna().astype(str).unique().tolist())
        return ["Any"] + [g for g in genres if g.strip()]
    except Exception:
        return ["Any", "Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller"]


def star_string(rating):
    try:
        filled = max(0, min(5, round(float(rating) / 2)))
        return "★" * filled + "☆" * (5 - filled)
    except Exception:
        return "—"


def score_pct(score, max_score=100.0):
    try:
        return min(100.0, max(0.0, float(score) / max_score * 100))
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


# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 8px;'>
        <div style='font-family:"Bebas Neue",sans-serif; font-size:1.8rem;
                    letter-spacing:0.12em; color:#c9a84c;'>CINE AI</div>
        <div style='font-size:0.7rem; letter-spacing:0.2em; color:#8a8f9e;
                    text-transform:uppercase; margin-top:2px;'>Smart Recommender</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Your Preferences</div>', unsafe_allow_html=True)

    all_genres = safe_load_genres()
    genre = st.selectbox("Genre", all_genres,
                         help="Pick a genre, or choose Any to include all.")

    min_rating = st.slider("Minimum Rating  ⭐", 1.0, 10.0, 6.0, 0.1,
                           help="Show only movies rated at or above this.")

    max_runtime = st.slider("Max Runtime  ⏱️", 60, 338, 180, 5,
                            help="Exclude movies longer than this (in minutes).")

    min_year = st.slider("From Year  📅", 1916, 2017, 1990, 1,
                         help="Show movies released from this year onward.")

    top_n = st.slider("Results  🎬", 3, 20, 8, 1,
                      help="Number of recommendations to display.")

    st.markdown('<div class="sidebar-section">Active Filters</div>', unsafe_allow_html=True)
    if not MODULES_OK:
        st.warning("Modules failed to load.")
    else:
        try:
            chips = get_constraint_summary(genre, min_rating, max_runtime, min_year)
            html_chips = "".join(f'<span class="constraint-chip">{c}</span>' for c in chips)
            st.markdown(html_chips, unsafe_allow_html=True)
        except Exception:
            st.caption("Filters will apply on search.")

    st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  HERO
# ════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered Discovery</div>
    <div class="hero-title">Find Your Next<br><span>Favourite Film</span></div>
    <div class="hero-sub">
        Set your preferences on the left, then hit the button below.
        Our AI scans thousands of movies to surface the ones you will love most.
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  FATAL IMPORT ERROR GUARD
# ════════════════════════════════════════════════════════════════

if not MODULES_OK:
    st.markdown(f"""
    <div class="err-banner">
        <b>⚠️ Application failed to start.</b><br>
        One or more backend modules could not be imported.<br>
        <code style="font-size:0.8rem;">{_import_err_msg}</code><br><br>
        Make sure you ran <code>pip install -r requirements.txt</code>
        and that the <code>modules/</code> folder exists next to <code>app.py</code>.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════════
#  DISCOVER BUTTON
# ════════════════════════════════════════════════════════════════

left_space, center_btn, right_space = st.columns([2, 2, 2])

with center_btn:
    run = st.button(
        "🎬  Discover Movies",
        use_container_width=True
    )

if not run:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🎞️</div>
        <div class="empty-title">Ready When You Are</div>
        <div class="empty-sub">
            Adjust your preferences in the sidebar,<br>
            then click <b>Discover Movies</b> to get your picks.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="footer">CineAI · Hybrid AI Recommendation System</div>',
                unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════════
#  RUN PIPELINE WITH FULL ERROR HANDLING
# ════════════════════════════════════════════════════════════════

pipeline_error = None
results = pd.DataFrame()

with st.spinner("Scanning the library and ranking your matches…"):
    try:
        results, pipeline_error = get_recommendations(
            genre, min_rating, max_runtime, min_year, top_n
        )
    except MemoryError:
        pipeline_error = "The dataset is too large to process. Try narrowing your filters."
    except FileNotFoundError as e:
        pipeline_error = f"Dataset file not found — {e}"
    except KeyError as e:
        pipeline_error = (
            f"A required column is missing in the dataset: {e}. "
            "Check that movies.csv has columns: title, genre, year, rating, runtime, popularity."
        )
    except Exception as e:
        pipeline_error = f"Unexpected error: {e}"


# ─── No results / error ──────────────────────────────────────────────────────

def show_no_results(msg=""):
    if msg:
        st.markdown(f'<div class="err-banner"><b>⚠️ Could not generate recommendations</b><br>{msg}</div>',
                    unsafe_allow_html=True)
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🔍</div>
        <div class="empty-title">No Matches Found</div>
        <div class="empty-sub">
            Try lowering the minimum rating, raising the max runtime,<br>
            choosing an earlier year, or selecting <b>Any</b> as genre.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="footer">CineAI · Hybrid AI Recommendation System</div>',
                unsafe_allow_html=True)
    st.stop()


if pipeline_error:
    show_no_results(pipeline_error)

if results is None or (isinstance(results, pd.DataFrame) and results.empty):
    show_no_results()


# ════════════════════════════════════════════════════════════════
#  STATS STRIP
# ════════════════════════════════════════════════════════════════

try:
    avg_r      = round(float(results["rating"].astype(float).mean()), 1)
    avg_pop    = int(results["popularity"].astype(float).mean())
    yr_min     = int(results["year"].min())
    yr_max     = int(results["year"].max())
    yr_span    = f"{yr_min} – {yr_max}" if yr_min != yr_max else str(yr_min)
    n_genres   = results["genre"].nunique()
except Exception:
    avg_r, avg_pop, yr_span, n_genres = "—", "—", "—", "—"

st.markdown(f"""
<div class="stat-strip">
    <div class="stat-box">
        <div class="stat-number">{len(results)}</div>
        <div class="stat-label">Picks for You</div>
    </div>
    <div class="stat-box">
        <div class="stat-number">{avg_r}</div>
        <div class="stat-label">Avg Rating</div>
    </div>
    <div class="stat-box">
        <div class="stat-number">{avg_pop}</div>
        <div class="stat-label">Avg Popularity</div>
    </div>
    <div class="stat-box">
        <div class="stat-number" style="font-size:1.3rem;padding-top:8px;">{yr_span}</div>
        <div class="stat-label">Year Range</div>
    </div>
    <div class="stat-box">
        <div class="stat-number">{n_genres}</div>
        <div class="stat-label">Genres</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  MOVIE CARDS  (2-column grid)
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="section-heading">Your <span>Recommendations</span></div>',
            unsafe_allow_html=True)

left_col, right_col = st.columns(2, gap="medium")

for i, row in results.reset_index(drop=True).iterrows():
    target = left_col if i % 2 == 0 else right_col

    try:
        rank        = i + 1
        title       = str(row.get("title", "Unknown"))
        year        = int(row.get("year", 0))
        genre_v     = str(row.get("genre", "—"))
        rating      = float(row.get("rating", 0))
        runtime     = int(row.get("runtime", 0))
        popularity  = int(row.get("popularity", 0))
        h_score     = float(row.get("heuristic_score", 0))
        cluster_id  = int(row.get("cluster", 0))
        ann_pred    = row.get("ann_predicted_rating", None)
        explanation = str(row.get("explanation", ""))
        cluster_nm  = get_cluster_label(cluster_id)
        stars       = star_string(rating)
        bar_pct     = score_pct(h_score)
        ann_str     = f"{float(ann_pred):.1f}" if ann_pred is not None else "—"
    except Exception as card_err:
        target.warning(f"Card #{i+1} could not be displayed: {card_err}")
        continue

    with target:
        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-rank">{rank:02d}</div>
            <div class="movie-title-text">
                {title}<span class="movie-year">{year}</span>
            </div>
            <div style="color:#c9a84c;font-size:0.85rem;margin-bottom:10px;
                        letter-spacing:0.06em;">{stars}</div>
            <div class="meta-pills">
                <span class="pill">🎭 <b>{genre_v}</b></span>
                <span class="pill">⭐ <b>{rating}</b> / 10</span>
                <span class="pill">⏱️ <b>{runtime} min</b></span>
                <span class="pill">🔥 <b>{popularity}</b></span>
                <span class="pill pill-gold">🤖 {ann_str}</span>
                <span class="pill pill-green">📂 {cluster_nm}</span>
            </div>
            <div class="score-bar-wrap">
                <div class="score-bar-track">
                    <div class="score-bar-fill" style="width:{bar_pct:.1f}%;"></div>
                </div>
                <div class="score-label">AI {h_score:.0f}</div>
            </div>
            <div class="explain">{explanation}</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  CHARTS
# ════════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-heading">Visual <span>Insights</span></div>',
            unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="medium")

# Chart 1 — AI Score vs Rating
with c1:
    st.markdown('<div class="chart-card"><div class="chart-title">AI SCORE vs IMDB RATING</div>',
                unsafe_allow_html=True)
    try:
        fig, ax = dark_fig(5.5, 3.8)
        xs = results["rating"].astype(float).values
        ys = results["heuristic_score"].astype(float).values
        ax.scatter(xs, ys, c=ys, cmap="YlOrRd", s=130, zorder=5,
                   edgecolors="#2a2e3a", linewidths=0.8,
                   vmin=ys.min() - 5, vmax=ys.max() + 5)
        for _, r in results.iterrows():
            ax.annotate(str(r["title"])[:13],
                        (float(r["rating"]), float(r["heuristic_score"])),
                        fontsize=6.2, color="#8a8f9e",
                        textcoords="offset points", xytext=(5, 3))
        ax.set_xlabel("IMDb Rating", color="#8a8f9e", fontsize=8)
        ax.set_ylabel("AI Score",    color="#8a8f9e", fontsize=8)
        ax.grid(True, color="#1e2230", linewidth=0.6, zorder=0)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.caption(f"Chart unavailable: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# Chart 2 — Cluster donut
with c2:
    st.markdown('<div class="chart-card"><div class="chart-title">CLUSTER BREAKDOWN</div>',
                unsafe_allow_html=True)
    try:
        cc = results["cluster"].value_counts()
        if cc.empty:
            raise ValueError("No cluster data.")
        pie_labels = [get_cluster_label(int(c)) for c in cc.index]
        palette = ["#c9a84c","#e8c97a","#2d6a4f","#1a73e8","#9c27b0","#c0392b","#16a085"]
        fig2, ax2 = dark_fig(5.5, 3.8)
        wedges, _, autotexts = ax2.pie(
            cc.values, labels=None, autopct="%1.0f%%",
            colors=palette[:len(cc)], startangle=90,
            wedgeprops={"linewidth": 2, "edgecolor": "#10131a"}, pctdistance=0.78)
        for at in autotexts:
            at.set_fontsize(8); at.set_color("white")
        ax2.legend(wedges, pie_labels, loc="lower center",
                   bbox_to_anchor=(0.5, -0.18), ncol=3,
                   fontsize=7, frameon=False, labelcolor="#8a8f9e")
        st.pyplot(fig2)
        plt.close(fig2)
    except Exception as e:
        st.caption(f"Chart unavailable: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

c3, c4 = st.columns(2, gap="medium")

# Chart 3 — Runtime histogram
with c3:
    st.markdown('<div class="chart-card"><div class="chart-title">RUNTIME DISTRIBUTION</div>',
                unsafe_allow_html=True)
    try:
        rt = results["runtime"].astype(float).values
        if len(rt) < 2:
            raise ValueError("Too few data points.")
        fig3, ax3 = dark_fig(5.5, 3.2)
        ax3.hist(rt, bins=min(len(rt), 8), color="#c9a84c", alpha=0.8,
                 edgecolor="#10131a", linewidth=1)
        ax3.set_xlabel("Runtime (min)", color="#8a8f9e", fontsize=8)
        ax3.set_ylabel("Count",         color="#8a8f9e", fontsize=8)
        ax3.grid(True, axis="y", color="#1e2230", linewidth=0.6)
        st.pyplot(fig3)
        plt.close(fig3)
    except Exception as e:
        st.caption(f"Chart unavailable: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# Chart 4 — Rating bars
with c4:
    st.markdown('<div class="chart-card"><div class="chart-title">RATING OVERVIEW</div>',
                unsafe_allow_html=True)
    try:
        rb = results.copy().reset_index(drop=True)
        rb["short_title"] = rb["title"].astype(str).str[:16]
        rb = rb.sort_values("rating", ascending=True)
        fig4, ax4 = dark_fig(5.5, 3.2)
        bar_colors = ["#c9a84c" if v >= 7.5 else "#5a6070"
                      for v in rb["rating"].astype(float)]
        ax4.barh(rb["short_title"], rb["rating"].astype(float),
                 color=bar_colors, edgecolor="#10131a", linewidth=0.8, height=0.65)
        ax4.set_xlabel("IMDb Rating", color="#8a8f9e", fontsize=8)
        ax4.axvline(7.5, color="#c9a84c", linewidth=0.8, linestyle="--", alpha=0.5)
        ax4.tick_params(axis="y", labelsize=6.5, colors="#8a8f9e")
        ax4.grid(True, axis="x", color="#1e2230", linewidth=0.6)
        ax4.set_xlim(0, 10.5)
        st.pyplot(fig4)
        plt.close(fig4)
    except Exception as e:
        st.caption(f"Chart unavailable: {e}")
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  SIMILARITY GRAPH  (BFS, no user-facing control)
# ════════════════════════════════════════════════════════════════

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-heading">Similarity <span>Graph</span></div>',
            unsafe_allow_html=True)
st.markdown("""
<div style="color:#8a8f9e;font-size:0.85rem;margin-bottom:16px;">
    Movies are connected when they share a genre or have similar ratings.
    Gold nodes were identified as your top matches.
</div>
""", unsafe_allow_html=True)

try:
    path, graph = get_search_results(results, "BFS")

    if graph is None or len(graph.nodes) == 0:
        st.info("Not enough results to build a similarity graph. Try requesting more movies.")
    else:
        fig5, ax5 = plt.subplots(figsize=(12, 5))
        fig5.patch.set_facecolor("#10131a")
        ax5.set_facecolor("#10131a")
        pos = nx.spring_layout(graph, seed=42, k=2.2)
        node_colors = ["#c9a84c" if n in path else "#2a2e3a" for n in graph.nodes()]
        node_sizes  = [820 if n in path else 500 for n in graph.nodes()]
        nx.draw_networkx_edges(graph, pos, ax=ax5,
                               edge_color="#2a2e3a", alpha=0.7, width=1.0)
        nx.draw_networkx_nodes(graph, pos, ax=ax5,
                               node_color=node_colors,
                               node_size=node_sizes, alpha=0.95)
        nx.draw_networkx_labels(graph, pos, ax=ax5,
                                font_size=6, font_color="#ffffff", font_weight="bold")
        ax5.axis("off")
        gold_p = mpatches.Patch(color="#c9a84c", label="Top Match")
        grey_p = mpatches.Patch(color="#2a2e3a", label="Related Movie")
        ax5.legend(handles=[gold_p, grey_p], loc="lower right",
                   frameon=False, labelcolor="#8a8f9e", fontsize=8)
        st.pyplot(fig5)
        plt.close(fig5)
except Exception as ge:
    st.info(f"Similarity graph could not be rendered: {ge}")


# ════════════════════════════════════════════════════════════════
#  FULL DATA TABLE
# ════════════════════════════════════════════════════════════════

with st.expander("📋  Full Results Table"):
    try:
        show_cols = ["title","genre","year","rating","runtime",
                     "popularity","heuristic_score","cluster","ann_predicted_rating"]
        display_df = results[[c for c in show_cols if c in results.columns]].copy()
        display_df.index = range(1, len(display_df) + 1)
        display_df.index.name = "#"
        st.dataframe(display_df, use_container_width=True)
    except Exception as te:
        st.warning(f"Table could not be rendered: {te}")


# ════════════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════════════

st.markdown('<div class="footer">CineAI · Hybrid AI Recommendation System</div>',
            unsafe_allow_html=True)
