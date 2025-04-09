import streamlit as st
from model import get_recommendations
from load_data import load_movie_data

@st.cache_data
def load_movies():
    return load_movie_data()

if 'movies' not in st.session_state:
    st.session_state.movies = load_movies()

df = st.session_state.movies

st.title("🎯 Рекомендации")

movie_list = sorted(df['title'].dropna().unique())
selected_movie = st.selectbox("Выберите фильм", movie_list)

if st.button("Показать рекомендации"):
    recs = get_recommendations(selected_movie)
    st.write("### Вам может понравиться:")
    for i, title in enumerate(recs, 1):
        st.write(f"{i}. {title}")
