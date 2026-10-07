"""Streamlit dashboard built on FilmAura's existing modules and data files."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data" / "gold" / "data_gold.json"
MODEL_PATH = ROOT / "models" / "best_model" / "best_model.joblib"


def load_movies() -> pd.DataFrame:
    with GOLD_PATH.open(encoding="utf-8") as file:
        movies = pd.DataFrame(json.load(file))
    movies["release_date"] = pd.to_datetime(movies.get("release_date"), errors="coerce")
    movies["genre_names"] = movies.get("genres", pd.Series(index=movies.index)).apply(
        lambda values: ", ".join(item.get("name", "") for item in values if isinstance(item, dict))
        if isinstance(values, list) else ""
    )
    return movies


def cluster_movies(movies: pd.DataFrame) -> pd.DataFrame:
    frame = movies.copy()
    columns = [column for column in ["popularity", "vote_average", "vote_count", "runtime"] if column in frame]
    values = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    frame["cluster"] = KMeans(n_clusters=min(5, max(1, len(frame))), random_state=42, n_init=10).fit_predict(values)
    return frame


def recommendations(title: str, movies: pd.DataFrame) -> pd.DataFrame:
    frame = movies.reset_index(drop=True)
    text = frame.get("overview", pd.Series("", index=frame.index)).fillna("")
    matrix = TfidfVectorizer(stop_words="english").fit_transform(text)
    selected = frame.index[frame["title"].eq(title)].tolist()
    if not selected:
        return frame.head(0)
    scores = cosine_similarity(matrix[selected[0]], matrix).ravel()
    return frame.assign(score=scores).drop(index=selected[0]).sort_values("score", ascending=False).head(5)


st.set_page_config(page_title="FilmAura AI", page_icon="🎬", layout="wide")
st.title("FilmAura AI")

if not GOLD_PATH.exists():
    st.error("Le fichier gold est introuvable. Lancez d'abord le pipeline Airflow.")
    st.stop()

movies = load_movies()
dashboard, classification, clusters, recommendations_tab = st.tabs(
    ["Dashboard", "Classification", "Clusters", "Recommandations"]
)

with dashboard:
    first, second, third = st.columns(3)
    first.metric("Films", len(movies))
    second.metric("Note moyenne", f"{movies['vote_average'].mean():.1f}/10")
    third.metric("Popularité médiane", f"{movies['popularity'].median():.1f}")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.histogram(movies, x="vote_average", title="Distribution des notes"), use_container_width=True)
    with right:
        releases = movies.dropna(subset=["release_date"]).assign(year=lambda data: data["release_date"].dt.year)
        st.plotly_chart(px.histogram(releases, x="year", title="Films par année"), use_container_width=True)
    st.dataframe(movies[["title", "vote_average", "popularity", "genre_names"]], use_container_width=True)

with classification:
    if MODEL_PATH.exists():
        st.success("Modèle de classification disponible")
        st.dataframe(movies[["title", "vote_count", "vote_average"]].sort_values("vote_count", ascending=False).head(20), use_container_width=True)
    else:
        st.warning("Aucun modèle disponible. Lancez la tâche ML du DAG.")

with clusters:
    clustered = cluster_movies(movies)
    st.plotly_chart(px.scatter(clustered, x="popularity", y="vote_average", color="cluster", hover_name="title"), use_container_width=True)
    st.dataframe(clustered.groupby("cluster").agg(films=("id", "count"), note_moyenne=("vote_average", "mean")).reset_index(), use_container_width=True)

with recommendations_tab:
    title = st.selectbox("Choisir un film", movies["title"].dropna().sort_values().tolist())
    result = recommendations(title, movies)
    st.dataframe(result[["title", "vote_average", "popularity", "genre_names"]], use_container_width=True)
