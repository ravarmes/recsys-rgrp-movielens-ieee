import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
import re

class RecSysContentBased2:
    def __init__(self, k, ratings=None, movies_path=None, user_based=False):
        self.k = k
        self.user_based = user_based  
        self.movies_path = movies_path
        if ratings is not None:
            self.set_ratings(ratings)
        if movies_path is not None:
            self.set_movie_genres(movies_path)

    def set_ratings(self, ratings):
        # Aqui assumimos que os índices da matriz ratings são os IDs de filme
        self.ratings = ratings
    
    def normalize_title(self, title):
        title = title.strip().lower()
        return re.sub(r'[^\w\s]', '', title)

    def is_title_match(self, title, movie_df):
        normalized_title = self.normalize_title(title)
        for index, row in movie_df.iterrows():
            if normalized_title in self.normalize_title(row['Title']):
                return row['MovieID']
        return None

    def set_movie_genres(self, movies_path):
        movies = pd.read_csv(
            movies_path,
            sep="::",
            engine="python",
            header=None,
            names=["MovieID", "Title", "Genres"],
            encoding="latin1"
        )

        # Criar um mapeamento de títulos para IDs de filmes
        movie_ids = {}
        for title in self.ratings.columns:
            movie_id = self.is_title_match(title, movies)
            if movie_id is not None:
                movie_ids[title] = movie_id

        print(f"Matched movie IDs: {movie_ids}")  # Debug: imprime IDs correspondidos

        # Filtra os filmes apenas para aqueles encontrados nas classificações
        filtered_movies = movies[movies['MovieID'].isin(movie_ids.values())]

        if filtered_movies.empty:
            raise ValueError("Nenhum filme correspondente encontrado em 'movies.dat'.")

        # Cria a matriz de gêneros
        genres = filtered_movies['Genres'].str.get_dummies('|')
        self.movie_genres = genres.set_index(filtered_movies['MovieID']).reindex(movie_ids.values(), fill_value=0)

        # Define as colunas de ratings para usar ID do filme
        self.ratings.columns = [movie_ids.get(title, title) for title in self.ratings.columns]
    
    def get_similarity_matrix(self):
        if not hasattr(self, 'movie_genres'):
            raise ValueError("A variável 'movie_genres' não foi definida.")
        # Calcula e retorna a matriz de similaridade
        similarity = cosine_similarity(self.movie_genres)
        return pd.DataFrame(similarity, index=self.movie_genres.index, columns=self.movie_genres.index)

    def knn_filtering(self, similarity):
        for i in similarity.index:
            sorted_neighbors = similarity.loc[i].sort_values(ascending=False)
            keep = sorted_neighbors.iloc[:self.k + 1].index
            similarity.loc[i, ~similarity.columns.isin(keep)] = 0
        return similarity.div(similarity.sum(axis=1), axis=0)
    

    def fit_model(self, max_iter=200, threshold=1e-5):
        similarity = self.get_similarity_matrix()
        knn_similarity = self.knn_filtering(similarity)

        # Debug: imprimir formas para verificar alinhamento
        print(f"Ratings shape: {self.ratings.shape}")
        print(f"KNN Similarity shape: {knn_similarity.shape}")

        try:
            # Gerar predições
            pred = self.ratings.fillna(0).dot(knn_similarity)
            pred = pred.div(knn_similarity.sum(axis=0), axis=1)

            # Remover colunas com todos os zeros
            pred = pred.loc[:, (pred != 0).any(axis=0)]

            # Verifica novamente as formas depois da filtragem
            print(f"Predictions post-filtering shape: {pred.shape}")

            # Alinhar as predições com os IDs de filmes
            pred = pred.reindex(columns=self.ratings.columns, fill_value=0)

            if pred.empty or pred.shape[1] == 0:
                raise ValueError("Não há colunas válidas para a imputação após filtragem.")

            # Imputação de valores ausentes
            imputer = SimpleImputer(strategy='mean')
            pred_imputed = imputer.fit_transform(pred)

            # Criar DataFrame a partir da matriz de predições imputadas
            self.pred = pd.DataFrame(pred_imputed, index=self.ratings.index, columns=pred.columns)

            # Atribuir matrizes de avaliações e similaridade
            self.U = self.ratings
            self.V = knn_similarity

            return self.pred

        except ValueError as e:
            print(f"Erro ao calcular predições: {e}")
            print("Certifique-se de que as colunas de ratings e a matriz de similaridade estão corretamente alinhadas.")
            print("Ratings columns:", self.ratings.columns)
            print("KNN Similarity index:", knn_similarity.index)
            print("Filtered and aligned pred columns:", pred.columns)
            raise