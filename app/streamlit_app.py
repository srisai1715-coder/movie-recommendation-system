"""Interactive Streamlit Application for CoRate Cinema.

Provides an interactive user interface for:
- Movie search & item-to-item similarity
- Personalized User-Based and Item-Based recommendations
- Cold-start fallback for new users
- Visual exploratory data analysis and model evaluation metrics
"""

import os
import sys
import tempfile

# Environment configurations
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from src.recommender import MovieRecommender
from src.model_training import train_and_save_pipeline


# Page config
st.set_page_config(
    page_title="CoRate Cinema — Movies Matched to Your Taste",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def inject_custom_css():
    """Applies clean card styling, hero banner, and genre badges."""
    st.markdown(
        """
        <style>
        .hero-banner {
            padding: 1.4rem 1.6rem;
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(229, 9, 20, 0.16), rgba(30, 41, 59, 0.55));
            border: 1px solid rgba(255, 255, 255, 0.10);
            margin-bottom: 1.1rem;
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0;
            line-height: 1.2;
        }
        .hero-tagline {
            font-size: 1.05rem;
            font-weight: 600;
            color: #F87171;
            margin-top: 0.25rem;
            margin-bottom: 0.45rem;
        }
        .hero-sub {
            font-size: 0.94rem;
            opacity: 0.88;
            margin: 0;
        }
        .movie-card {
            padding: 1.1rem 1.25rem;
            border-radius: 12px;
            background-color: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.10);
            margin-bottom: 0.9rem;
        }
        .rec-card {
            padding: 0.85rem 1rem;
            border-radius: 10px;
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            margin-bottom: 0.65rem;
        }
        .genre-tag {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            margin: 0.15rem 0.25rem 0.15rem 0;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background-color: rgba(229, 9, 20, 0.18);
            border: 1px solid rgba(229, 9, 20, 0.35);
        }
        .score-badge {
            display: inline-block;
            padding: 0.22rem 0.6rem;
            border-radius: 8px;
            font-size: 0.82rem;
            font-weight: 700;
            background-color: rgba(34, 197, 94, 0.16);
            color: #4ADE80;
            border: 1px solid rgba(34, 197, 94, 0.35);
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_genre_badges(genres_str: str) -> str:
    """Converts pipe-separated genres (e.g. Action|Comedy) into clean tags."""
    if not genres_str or pd.isna(genres_str):
        return "<span class='genre-tag'>General</span>"
    parts = [g.strip() for g in str(genres_str).split("|") if g.strip()]
    return "".join(f"<span class='genre-tag'>{g}</span>" for g in parts)


@st.cache_resource
def load_recommender_model():
    """Loads cached recommendation model or trains it on first boot."""
    model_path = os.path.join("models", "recommender_model.joblib")
    if not os.path.exists(model_path):
        with st.spinner("Model artifact not found on disk. Initializing and training model..."):
            train_and_save_pipeline(
                data_dir="dataset",
                model_output_path=model_path,
                min_movie_ratings=10,
                min_user_ratings=5,
                test_size=0.20,
                random_state=42
            )
    return MovieRecommender.load(model_path)


def main():
    inject_custom_css()

    # Header Banner
    st.markdown(
        """
        <div class="hero-banner">
            <p class="hero-title">🎬 CoRate Cinema</p>
            <p class="hero-tagline">Movies Matched to Your Taste</p>
            <p class="hero-sub">
                Discover similar films, personalized user recommendations, cold-start picks, and evaluation insights
                powered by collaborative filtering on the MovieLens dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load Model
    try:
        model = load_recommender_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Sidebar Controls
    st.sidebar.header("⚙️ Studio Controls")
    top_k = st.sidebar.slider("Number of Recommendations (Top-K)", min_value=5, max_value=25, value=10, step=1)
    cf_method = st.sidebar.radio(
        "Recommendation Strategy",
        options=["item_based", "user_based"],
        format_func=lambda x: "Item-Based Collaborative Filtering" if x == "item_based" else "User-Based Collaborative Filtering",
        help="Item-based uses movie similarity; User-based uses peer neighborhood similarity."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Catalog Statistics")
    st.sidebar.metric("Active Users", f"{len(model.user_to_idx):,}")
    st.sidebar.metric("Trained Movies", f"{len(model.movie_to_idx):,}")
    st.sidebar.metric("Interaction Sparsity", "98.3%")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Movie Similarity",
        "👤 Personalized Picks",
        "⭐ Cold-Start & Popular",
        "📈 Analytics & Evaluation"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: MOVIE SIMILARITY
    # --------------------------------------------------------------------------
    with tab1:
        st.subheader("Find Similar Movies")
        st.caption("Pick any film to see which movies share the strongest viewer co-rating patterns.")

        all_titles = sorted(model.movies_df["title"].unique())
        default_index = all_titles.index("Toy Story (1995)") if "Toy Story (1995)" in all_titles else 0

        selected_title = st.selectbox("Search or Select a Movie:", options=all_titles, index=default_index)

        if selected_title:
            movie_record = model.movies_df[model.movies_df["title"] == selected_title].iloc[0]
            target_movie_id = int(movie_record["movieId"])

            col_a, col_b = st.columns([1.1, 2.2])
            with col_a:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em; opacity:0.7;">Selected Film</div>
                        <h4 style="margin:0.35rem 0 0.5rem 0;">{selected_title}</h4>
                        <div style="margin-bottom:0.65rem;">{render_genre_badges(movie_record['genres'])}</div>
                        <div style="font-size:0.86rem; opacity:0.8;">Movie ID: <code>{target_movie_id}</code></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_b:
                with st.spinner("Calculating cosine similarities..."):
                    similar_movies = model.recommend_similar_movies(movie_id=target_movie_id, top_k=top_k)

                if similar_movies:
                    st.markdown(f"**Top {len(similar_movies)} Similar Films**")
                    for rank, item in enumerate(similar_movies, start=1):
                        sim_val = item.get("similarity", item.get("cosine_similarity", 0.0))
                        st.markdown(
                            f"""
                            <div class="rec-card">
                                <div style="display:flex; justify-content:space-between; align-items:center; gap:0.5rem;">
                                    <div>
                                        <strong>#{rank} · {item['title']}</strong>
                                        <span style="opacity:0.65; font-size:0.82rem;"> (ID: {item['movieId']})</span>
                                    </div>
                                    <span class="score-badge">Similarity: {float(sim_val):.3f}</span>
                                </div>
                                <div style="margin-top:0.35rem;">{render_genre_badges(item['genres'])}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    with st.expander("View as Sortable Table"):
                        sim_df = pd.DataFrame(similar_movies)
                        sim_df.columns = ["Movie ID", "Title", "Genres", "Cosine Similarity"]
                        st.dataframe(sim_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No similar movies found above the similarity threshold.")

    # --------------------------------------------------------------------------
    # TAB 2: PERSONALIZED USER RECOMMENDATIONS
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("Personalized User Recommendations")
        st.caption("Get tailored movie picks for an existing viewer ID, or enter a new ID (>610) to test cold-start fallback.")

        col_input, col_action = st.columns([2, 1])
        with col_input:
            user_id_input = st.number_input(
                "Enter User ID (Known IDs: 1 to 610, or >610 for Cold-Start demo):",
                min_value=1,
                max_value=99999,
                value=1,
                step=1
            )

        with col_action:
            st.write("")
            st.write("")
            st.button("Generate Recommendations", type="primary")

        is_known_user = user_id_input in model.user_to_idx
        if is_known_user:
            user_watched_ids = model.user_watched.get(user_id_input, set())
            st.info(f"**User {user_id_input}** has rated **{len(user_watched_ids)}** movies in the training set.")

            with st.expander("View User's Top Rated Movies (from Training Data)", expanded=False):
                u_idx = model.user_to_idx[user_id_input]
                user_row = model.sparse_matrix[u_idx]
                r_indices = user_row.indices
                r_scores = user_row.data

                user_history = []
                for m_idx, score in zip(r_indices, r_scores):
                    m_id = model.idx_to_movie[m_idx]
                    t = model.movies_df[model.movies_df["movieId"] == m_id]["title"].values[0]
                    g = model.movies_df[model.movies_df["movieId"] == m_id]["genres"].values[0]
                    user_history.append({"Title": t, "Rating": score, "Genres": g})

                history_df = pd.DataFrame(user_history).sort_values(by="Rating", ascending=False)
                st.dataframe(history_df.head(10), use_container_width=True, hide_index=True)
        else:
            st.warning(f"⚠️ User {user_id_input} is a **Brand New User (Cold-Start)**. CoRate Cinema will fall back to Bayesian Popular Movies.")

        with st.spinner("Generating recommendations..."):
            recs = model.recommend(user_id=user_id_input, method=cf_method, top_k=top_k)

        if recs:
            st.markdown(f"### Top {top_k} Picks for User {user_id_input}")
            for rank, rec in enumerate(recs, start=1):
                st.markdown(
                    f"""
                    <div class="rec-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; gap:0.5rem;">
                            <div><strong>#{rank} · {rec['title']}</strong></div>
                            <span class="score-badge">★ {float(rec['predicted_score']):.2f} · {rec.get('method', cf_method)}</span>
                        </div>
                        <div style="margin-top:0.35rem;">{render_genre_badges(rec['genres'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with st.expander("View as Data Table"):
                recs_df = pd.DataFrame(recs)
                display_cols = ["title", "genres", "predicted_score", "method"]
                rename_map = {
                    "title": "Title",
                    "genres": "Genres",
                    "predicted_score": "Predicted Score (Stars)",
                    "method": "Recommendation Method"
                }
                recs_df = recs_df[[c for c in display_cols if c in recs_df.columns]].rename(columns=rename_map)
                st.dataframe(recs_df, use_container_width=True, hide_index=True)

            st.success("✅ Already-watched movies are excluded from these recommendations.")

    # --------------------------------------------------------------------------
    # TAB 3: COLD-START & POPULAR FALLBACK
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("Cold-Start & Popular Fallback")
        st.markdown(
            "When a new viewer has no rating history yet, collaborative filtering has no past tastes to compare. "
            "CoRate Cinema uses an **IMDb Bayesian Weighted Rating** on the training split so popular, well-rated classics surface first:\n\n"
            r"$$\text{WR} = \left(\frac{v}{v + m}\right) R + \left(\frac{m}{v + m}\right) C$$"
            "\nwhere $v$ is rating count, $m$ is minimum threshold, $R$ is average movie rating, and $C$ is catalog mean."
        )

        popular_recs = model.recommend_popular(top_k=top_k)
        for rank, item in enumerate(popular_recs, start=1):
            st.markdown(
                f"""
                <div class="rec-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:0.5rem;">
                        <div>
                            <strong>#{rank} · {item['title']}</strong>
                            <span style="opacity:0.7; font-size:0.84rem;"> ({item['rating_count']} ratings)</span>
                        </div>
                        <span class="score-badge">Bayesian Score: {float(item['predicted_score']):.2f}</span>
                    </div>
                    <div style="margin-top:0.35rem;">{render_genre_badges(item['genres'])}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.expander("View Popular Fallback Table"):
            pop_df = pd.DataFrame(popular_recs)[["title", "genres", "predicted_score", "rating_count"]]
            pop_df.columns = ["Title", "Genres", "Bayesian Score", "Rating Count"]
            st.dataframe(pop_df, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 4: ANALYTICS & MODEL EVALUATION
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader("Analytics & Model Evaluation")
        st.markdown("Benchmark metrics evaluated on the 20% held-out test split:")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Item-Based RMSE", "0.9193", delta="-0.247 vs User-Based", delta_color="inverse")
        with metric_col2:
            st.metric("Item-Based MAE", "0.7021", delta="-0.280 vs User-Based", delta_color="inverse")
        with metric_col3:
            st.metric("User-Based Precision@10", "1.87%", delta="+1.24% vs Item-Based")
        with metric_col4:
            st.metric("User-Based Catalog Coverage", "20.1%", delta="+8.24% vs Item-Based")

        st.markdown("---")
        st.subheader("Exploratory Data Analysis & Diagnostic Charts")

        viz_dir = "visualizations"
        chart_files = [
            ("01_rating_distribution.png", "1. Rating Distribution (0.5 to 5.0 Stars)"),
            ("02_most_rated_movies.png", "2. Most-Rated Movies in MovieLens"),
            ("03_highest_rated_movies.png", "3. Highest-Rated Movies (Min 50 Ratings)"),
            ("04_user_activity.png", "4. User Rating Activity & Dispersion"),
            ("05_movie_popularity.png", "5. Movie Popularity Long-Tail Curve"),
            ("06_similarity_scores.png", "6. Cosine Similarity Heatmap (Iconic Movies)"),
            ("07_model_evaluation_comparison.png", "7. Model Benchmark Comparison (Item-Based vs User-Based)"),
        ]

        chart_choice = st.selectbox("Select Chart to View:", options=[name for _, name in chart_files])

        for fname, name in chart_files:
            if name == chart_choice:
                img_path = os.path.join(viz_dir, fname)
                if os.path.exists(img_path):
                    image = Image.open(img_path)
                    st.image(image, caption=name, use_container_width=True)
                else:
                    st.warning(f"Chart file '{fname}' not yet generated. Run `python main.py` to generate charts.")


if __name__ == "__main__":
    main()