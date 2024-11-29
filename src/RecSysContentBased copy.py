import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from abc import ABCMeta
from sklearn.impute import SimpleImputer

class RecSysContentBased2:
    __metaclass__ = ABCMeta
    
    def __init__(self, k, ratings=None, movies_file='Data/MovieLens-1M/movies.dat'):
        self.k = k  # Adiciona o atributo k
        self.ratings = ratings
        self.movies_file = movies_file
        self.movies = None
        self.movie_features = None
        self.set_movies()
    
    def set_movies(self):
        # Leitura do arquivo de filmes
        movies_list = []
        with open(self.movies_file, 'r') as f:
            for line in f:
                parts = line.strip().split('::')
                movie_id = int(parts[0])
                movie_title = parts[1]
                genres = parts[2].split('|')
                movies_list.append((movie_id, movie_title, genres))
        
        self.movies = pd.DataFrame(movies_list, columns=['movie_id', 'title', 'genres'])
        self.movies['genres'] = self.movies['genres'].apply(lambda x: ' '.join(x))
        
        # Criar matriz de recursos baseado nos gêneros
        self.movie_features = self.movies.set_index('title')['genres'].str.get_dummies(sep=' ')
        
        # Reindexar movie_features para incluir apenas os filmes que estão em self.ratings
        self.movie_features = self.movie_features.reindex(columns=self.ratings.columns, fill_value=0)
    
    def get_similarity_matrix(self):
        # Calcular a matriz de similaridade
        similarity = pd.DataFrame(cosine_similarity(self.movie_features), index=self.movie_features.index, columns=self.movie_features.index)
        return similarity
    
    def knn_filtering(self, similarity):
        # Filtragem KNN
        for i in similarity.index:
            sorted_neighbors = similarity.loc[i].sort_values(ascending=False)
            keep = sorted_neighbors.iloc[:self.k + 1].index
            similarity.loc[i, ~similarity.columns.isin(keep)] = 0
        similarity = similarity.div(similarity.sum(axis=1), axis=0)
        return similarity
    
    def fit_model(self, max_iter=200, threshold=1e-5):
        similarity = self.get_similarity_matrix()
        knn_similarity = self.knn_filtering(similarity)

        # Certifique-se de que as matrizes estão alinhadas
        if not self.ratings.index.equals(knn_similarity.index):
            knn_similarity = knn_similarity.reindex(self.ratings.index)

        # Predição
        pred = knn_similarity.dot(self.ratings.fillna(0)).div(knn_similarity.sum(axis=1), axis=0)
        
        imputer = SimpleImputer(strategy='mean')
        self.pred = pd.DataFrame(imputer.fit_transform(pred), index=self.ratings.index, columns=self.ratings.columns)
        
        return self.pred