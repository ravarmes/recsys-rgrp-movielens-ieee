import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from abc import ABCMeta, abstractmethod

class RecSysContentBased3:
    __metaclass__ = ABCMeta
    
    def __init__(self, k, ratings=None, user_based=True, movie_file=None):
        self.k = k
        self.user_based = user_based
        self.movie_file = movie_file
        if ratings is not None:
            self.set_ratings(ratings)
    
    def set_ratings(self, ratings):
        """
        Define a matriz de avaliações e realiza imputação inteligente.
        """
        # Estratégia 3: Imputação inteligente
        imputer = SimpleImputer(strategy='mean')  # Substituir valores ausentes pela média dos filmes
        self.ratings = pd.DataFrame(
            imputer.fit_transform(ratings),
            index=ratings.index,
            columns=ratings.columns
        )
        self.num_of_known_ratings_per_user = (~ratings.isnull()).sum(axis=1)
        self.num_of_known_ratings_per_item = (~ratings.isnull()).sum(axis=0)
    
    def load_movie_genres(self):
        """
        Carrega os gêneros dos filmes do arquivo fornecido e cria uma matriz de características binárias.
        """
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
        """
        Calcula a matriz de similaridade baseada nos gêneros dos filmes.
        """
        if self.genre_matrix is None:
            raise ValueError("A matriz de gêneros não foi carregada.")
        similarity = pd.DataFrame(
            cosine_similarity(self.genre_matrix),
            index=self.genre_matrix.index,
            columns=self.genre_matrix.index
        )
        return similarity
    
    def knn_filtering(self, similarity):
        """
        Realiza o filtro de k-vizinhos mais próximos, zerando os filmes menos similares.
        """
        for i in similarity.index:
            sorted_neighbors = similarity.loc[i].sort_values(ascending=False)
            keep = sorted_neighbors.iloc[:self.k+1].index
            similarity.loc[i, ~similarity.columns.isin(keep)] = 0
        similarity = similarity.div(similarity.sum(axis=1) + 1e-5, axis=0)  # Estratégia 5: Regularização na normalização
        return similarity
    
    def fit_model(self):
        """
        Ajusta o modelo para gerar predições.
        """
        if self.movie_file is None:
            raise ValueError("O arquivo de filmes não foi fornecido.")

        self.load_movie_genres()
        similarity = self.get_similarity_matrix()
        knn_similarity = self.knn_filtering(similarity)

        knn_similarity = knn_similarity.reindex(columns=self.ratings.columns, index=self.ratings.columns)
        pred_raw = self.ratings.fillna(0).dot(knn_similarity)  # Predição sem normalização
        pred = pred_raw.copy()

        # Estratégia 5: Regularização na normalização da similaridade
        sum_similarities = knn_similarity.sum(axis=0)
        for i in range(len(pred.columns)):
            if sum_similarities[i] > 0:
                pred.iloc[:, i] = pred.iloc[:, i] / (sum_similarities[i] + 1e-5)

        # Estratégia 9: Ajustar com base em tendências globais
        global_mean = self.ratings.stack().mean()  # Média global
        user_bias = self.ratings.mean(axis=1) - global_mean  # Viés de cada usuário
        item_bias = self.ratings.mean(axis=0) - global_mean  # Viés de cada filme
        pred = pred.add(user_bias, axis=0).add(item_bias, axis=1).fillna(global_mean)

        # Clipping para garantir que os valores estejam entre 1 e 5
        pred = pred.clip(lower=1, upper=5)
        self.pred = pred
        self.U = self.ratings
        self.V = knn_similarity

        return self.pred
