import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import KNNImputer


class RecSysContentBased4:
    def __init__(self, k, ratings=None, user_based=True, movie_file=None, regularization=0.5):
        self.k = k
        self.user_based = user_based
        self.movie_file = movie_file
        self.regularization = regularization
        if ratings is not None:
            self.set_ratings(ratings)
    
    def set_ratings(self, ratings):
        # Imputação baseada em KNN para valores ausentes
        imputer = KNNImputer(n_neighbors=self.k)
        self.ratings = pd.DataFrame(
            imputer.fit_transform(ratings),
            index=ratings.index,
            columns=ratings.columns
        )
        self.num_of_known_ratings_per_user = (~ratings.isnull()).sum(axis=1)
        self.num_of_known_ratings_per_item = (~ratings.isnull()).sum(axis=0)
    
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
        
        # Regularização aprimorada
        similarity = pd.DataFrame(
            cosine_similarity(self.genre_matrix) * (1 - self.regularization) + self.regularization,
            index=self.genre_matrix.index,
            columns=self.genre_matrix.index
        )
        return similarity
    
    def knn_filtering(self, similarity):
        weights = np.linspace(1, 0.1, self.k + 1)  # Pesos lineares decrescentes
        for i in similarity.index:
            sorted_neighbors = similarity.loc[i].sort_values(ascending=False)
            keep = sorted_neighbors.iloc[:self.k + 1].index
            similarity.loc[i, ~similarity.columns.isin(keep)] = 0
            similarity.loc[i, keep] = similarity.loc[i, keep] * weights[:len(keep)]
        
        similarity = similarity.div(similarity.sum(axis=1), axis=0)
        return similarity
    
    def fit_model(self):
        if self.movie_file is None:
            raise ValueError("O arquivo de filmes não foi fornecido.")
        
        self.load_movie_genres()
        similarity = self.get_similarity_matrix()
        knn_similarity = self.knn_filtering(similarity)
        knn_similarity = knn_similarity.reindex(columns=self.ratings.columns, index=self.ratings.columns)
        
        pred_raw = self.ratings.fillna(0).dot(knn_similarity)
        sum_similarities = knn_similarity.sum(axis=0)
        pred = pred_raw.copy()
        
        for i in range(len(pred.columns)):
            if sum_similarities[i] > 0:
                pred.iloc[:, i] = pred.iloc[:, i] / sum_similarities[i]
        
        # Clipping entre os valores 1 e 5
        pred = pred.clip(lower=1, upper=5)
        
        # Pós-processamento: normalização com base na média e desvio padrão
        avg_ratings = self.ratings.mean(axis=0, skipna=True)
        std_ratings = self.ratings.std(axis=0, skipna=True)
        for i in range(len(pred.columns)):
            if avg_ratings[i] > 0:
                pred.iloc[:, i] = (pred.iloc[:, i] - avg_ratings[i]) / (std_ratings[i] + 1e-8)
                pred.iloc[:, i] = ((pred.iloc[:, i] - pred.iloc[:, i].min()) /
                                   (pred.iloc[:, i].max() - pred.iloc[:, i].min())) * 4 + 1
        
        # Penalização para valores extremos
        pred = pred.applymap(lambda x: x if 1 <= x <= 5 else (1 if x < 1 else 5))
        
        self.pred = pred.fillna(0)
        self.U = self.ratings
        self.V = knn_similarity
        
        return self.pred
