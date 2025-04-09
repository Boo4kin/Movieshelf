import pandas as pd

def load_movie_data(path="."):
    df_movies = pd.read_csv(f"{path}/movies_metadata.csv", low_memory=False)
    df_credits = pd.read_csv(f"{path}/credits.csv")
    df_keywords = pd.read_csv(f"{path}/keywords.csv")

    df_movies['id'] = pd.to_numeric(df_movies['id'], errors='coerce')
    df_credits['id'] = pd.to_numeric(df_credits['id'], errors='coerce')
    df_keywords['id'] = pd.to_numeric(df_keywords['id'], errors='coerce')

    df = df_movies.merge(df_credits, on='id').merge(df_keywords, on='id')
    df = df[['title', 'genres', 'keywords', 'crew', 'cast']]
    df = df.dropna(subset=['title'])

    return df
