import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from abc import ABCMeta, abstractmethod

class RecSysKNN2:
    __metaclass__ = ABCMeta
    
    def __init__(self, k, ratings=None, user_based=True, min_similarity=0.1, normalize=True, hybrid_factor=0.5):
        """
        Parâmetros:
        - k: Número de vizinhos mais próximos.
        - ratings: Matriz de classificações.
        - user_based: True para filtragem baseada em usuários, False para itens.
        - min_similarity: Limite mínimo para similaridade.
        - normalize: Normalizar as classificações para média zero.
        - hybrid_factor: Peso para combinar similaridade de usuários e itens (0 a 1).
        """
        self.k = k
        self.user_based = user_based
        self.min_similarity = min_similarity
        self.normalize = normalize
        self.hybrid_factor = hybrid_factor
        if ratings is not None:
            self.set_ratings(ratings)
    
    def set_ratings(self, ratings):
        self.ratings = ratings
        self.num_of_known_ratings_per_user = (~self.ratings.isnull()).sum(axis=1)
        self.num_of_known_ratings_per_item = (~self.ratings.isnull()).sum(axis=0)
    
    def normalize_ratings(self, ratings):
        if self.user_based:
            mean_ratings = ratings.mean(axis=1)
            return ratings.sub(mean_ratings, axis=0).fillna(0), mean_ratings
        else:
            mean_ratings = ratings.mean(axis=0)
            return ratings.sub(mean_ratings, axis=1).fillna(0), mean_ratings
    
    def get_similarity_matrix(self):
        ratings = self.ratings.copy()
        if self.normalize:
            ratings, _ = self.normalize_ratings(ratings)
        
        if self.user_based:
            similarity = pd.DataFrame(cosine_similarity(ratings.fillna(0)), index=ratings.index, columns=ratings.index)
        else:
            similarity = pd.DataFrame(cosine_similarity(ratings.fillna(0).T), index=ratings.columns, columns=ratings.columns)
        return similarity
    
    def knn_filtering(self, similarity):
        for i in similarity.index:
            sorted_neighbors = similarity.loc[i].sort_values(ascending=False)
            keep = sorted_neighbors.iloc[:self.k+1].index
            similarity.loc[i, ~similarity.columns.isin(keep)] = 0
        
        similarity = similarity[similarity >= self.min_similarity].fillna(0)
        similarity = similarity.div(similarity.sum(axis=1), axis=0)
        return similarity
    
    def fit_model(self, max_iter=200, threshold=1e-5):
        ratings = self.ratings.copy()
        if self.normalize:
            normalized_ratings, mean_ratings = self.normalize_ratings(ratings)
        else:
            normalized_ratings, mean_ratings = ratings, None
        
        similarity = self.get_similarity_matrix()
        knn_similarity = self.knn_filtering(similarity)
        
        if self.user_based:
            pred = knn_similarity.dot(normalized_ratings.fillna(0)).div(knn_similarity.sum(axis=1), axis=0)
        else:
            pred = normalized_ratings.fillna(0).dot(knn_similarity).div(knn_similarity.sum(axis=0), axis=1)
        
        if self.normalize:
            if self.user_based:
                pred = pred.add(mean_ratings, axis=0)
            else:
                pred = pred.add(mean_ratings, axis=1)
        
        imputer = SimpleImputer(strategy='mean')
        self.pred = pd.DataFrame(imputer.fit_transform(pred), index=self.ratings.index, columns=self.ratings.columns)
        
        self.U = knn_similarity if self.user_based else self.ratings
        self.V = self.ratings.T if self.user_based else knn_similarity.T
        
        return self.pred
