import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

class RecSysContentBased5:
    def __init__(self, ratings=None, movie_file=None):
        self.movie_file = movie_file
        if ratings is not None:
            self.set_ratings(ratings)

    def set_ratings(self, ratings):
        self.ratings = pd.DataFrame(ratings)
        self.num_of_known_ratings_per_user = (~self.ratings.isnull()).sum(axis=1)
        self.num_of_known_ratings_per_item = (~self.ratings.isnull()).sum(axis=0)

    def load_movie_genres(self):
        movies = pd.read_csv(
            self.movie_file,
            sep='::',
            header=None,
            names=['Title', 'Genres'],
            engine='python',
            encoding='latin1'
        )
        movies['Genres'] = movies['Genres'].str.split('|')
        genres = list(set(genre for sublist in movies['Genres'] for genre in sublist))
        genre_matrix = pd.DataFrame(0, index=movies['Title'], columns=genres)
        for _, row in movies.iterrows():
            genre_matrix.loc[row['Title'], row['Genres']] = 1
        self.genre_matrix = genre_matrix

    def get_similarity_matrix(self):
        if self.genre_matrix is None:
            raise ValueError("A matriz de gêneros não foi carregada.")
        
        genre_matrix_filled = self.genre_matrix.fillna(0)
        similarity = pd.DataFrame(
            cosine_similarity(genre_matrix_filled),
            index=self.genre_matrix.index,
            columns=self.genre_matrix.index
        )
        return similarity

    def fit_model(self):
        if self.movie_file is None:
            raise ValueError("O arquivo de filmes não foi fornecido.")

        self.load_movie_genres()
        similarity = self.get_similarity_matrix()
        self.predictions = pd.DataFrame(index=self.ratings.index, columns=self.ratings.columns).fillna(0)

        for user in self.ratings.index:
            known_items = self.ratings.loc[user][~self.ratings.loc[user].isnull()].index
            if not known_items.empty:
                item_similarities = similarity.loc[known_items]

                X = item_similarities.values.T  # Similaridades como características
                y = self.ratings.loc[user, known_items].values  # Avaliações conhecidas do usuário

                if X.shape[1] != y.size:
                    self.predictions.loc[user] = self.ratings.mean(axis=1)  # Fallback
                    continue

                model = LinearRegression()
                model.fit(X, y)

                unknown_items = self.ratings.columns[~self.ratings.columns.isin(known_items)]
                for item in unknown_items:
                    if item in similarity.columns:
                        prediction = model.predict(item_similarities.loc[item:item].values.reshape(1, -1))
                        self.predictions.loc[user, item] = prediction

        self.predictions = self.predictions.clip(lower=1, upper=5)
        return self.predictions