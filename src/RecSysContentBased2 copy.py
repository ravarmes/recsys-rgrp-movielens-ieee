import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from abc import ABCMeta, abstractmethod

class RecSysContentBased2:
    __metaclass__ = ABCMeta
    
    def __init__(self, k, ratings=None, movies_path=None, user_based=False):
        self.k = k
        self.user_based = user_based  
        self.movies_path = movies_path
        if ratings is not None:
            self.set_ratings(ratings)
        if movies_path is not None:
            self.set_movie_genres(movies_path)
    
    def set_ratings(self, ratings):
        self.ratings = ratings
        self.num_of_known_ratings_per_user = (~self.ratings.isnull()).sum(axis=1)
        self.num_of_known_ratings_per_item = (~self.ratings.isnull()).sum(axis=0)
    
    def set_movie_genres(self, movies_path):
        movies = pd.read_csv(
            movies_path, 
            sep="::", 
            engine="python", 
            header=None, 
            names=["MovieID", "Title", "Genres"], 
            encoding="latin1"  
        )
        
        filtered_movies = movies[movies["Title"].isin(self.ratings.columns)]
        genres = filtered_movies['Genres'].str.get_dummies('|')
        self.movie_genres = genres.set_index(filtered_movies['Title']).reindex(self.ratings.columns, fill_value=0)
    
    def get_similarity_matrix(self):
        similarity = pd.DataFrame(
            cosine_similarity(self.movie_genres),
            index=self.movie_genres.index,
            columns=self.movie_genres.index
        )
        return similarity
    
    def knn_filtering(self, similarity):
        for i in similarity.index:
            sorted_neighbors = similarity.loc[i].sort_values(ascending=False)
            keep = sorted_neighbors.iloc[:self.k+1].index
            similarity.loc[i, ~similarity.columns.isin(keep)] = 0
        similarity = similarity.div(similarity.sum(axis=1), axis=0)
        return similarity
    
    def fit_model(self, max_iter=200, threshold=1e-5):
        similarity = self.get_similarity_matrix()
        knn_similarity = self.knn_filtering(similarity)

        # Ajustar o alinhamento dos vizinhos mais próximos
        knn_similarity = knn_similarity.reindex(index=self.ratings.columns, columns=self.ratings.columns, fill_value=0)

        # Gerar predições
        pred = self.ratings.fillna(0).dot(knn_similarity)  # Produz predições

        # Normalizar as predições
        pred = pred.div(knn_similarity.sum(axis=0), axis=1)

        # Remover colunas vazias após a normalização
        pred = pred.loc[:, (pred != 0).any(axis=0)]
        
        # Verifique se ainda existem colunas válidas após filtragem
        if pred.empty or pred.shape[1] == 0:
            raise ValueError("Após filtragem, não há colunas válidas para a imputação.")

        # Impor um DataFrame sem colunas vazias para o imputador
        imputer = SimpleImputer(strategy='mean')
        # Filtros para evitar erros de imputação
        pred_imputed = imputer.fit_transform(pred)

        # Criar um DataFrame a partir da matriz imputada
        self.pred = pd.DataFrame(pred_imputed, index=self.ratings.index, columns=pred.columns)

        self.U = self.ratings
        self.V = knn_similarity

        return self.pred