import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from abc import ABCMeta, abstractmethod

class RecSysKNN3:
    __metaclass__ = ABCMeta
    
    def __init__(self, k, ratings=None, user_based=True):
        self.k = k
        self.user_based = user_based
        if ratings is not None:
            self.set_ratings(ratings)
    
    def set_ratings(self, ratings):
        self.ratings = ratings
        self.num_of_known_ratings_per_user = (~self.ratings.isnull()).sum(axis=1)
        self.num_of_known_ratings_per_item = (~self.ratings.isnull()).sum(axis=0)

    def normalize_ratings(self):
        # Normaliza as avaliações, subtraindo a média de cada usuário ou item
        if self.user_based:
            mean_values = self.ratings.mean(axis=1)
            normalized = self.ratings.sub(mean_values, axis=0)
        else:
            mean_values = self.ratings.mean(axis=0)
            normalized = self.ratings.sub(mean_values, axis=1)
        return normalized.fillna(0), mean_values

    def get_similarity_matrix(self):
        # Normaliza antes de calcular a similaridade
        normalized_ratings, _ = self.normalize_ratings()
        if self.user_based:
            similarity = pd.DataFrame(cosine_similarity(normalized_ratings), 
                                      index=self.ratings.index, 
                                      columns=self.ratings.index)
        else:
            similarity = pd.DataFrame(cosine_similarity(normalized_ratings.T), 
                                      index=self.ratings.columns, 
                                      columns=self.ratings.columns)
        return similarity
    
    def knn_filtering(self, similarity):
        for i in similarity.index:
            sorted_neighbors = similarity.loc[i].sort_values(ascending=False)
            keep = sorted_neighbors.iloc[:self.k+1].index
            similarity.loc[i, ~similarity.columns.isin(keep)] = 0
        similarity = similarity.div(similarity.sum(axis=1), axis=0)
        return similarity
    
    def fit_model(self, max_iter=200, threshold=1e-5):
        # Obtém similaridade e aplica KNN
        similarity = self.get_similarity_matrix()
        knn_similarity = self.knn_filtering(similarity)
        
        # Normaliza as classificações e ajusta as predições
        normalized_ratings, mean_values = self.normalize_ratings()
        
        if self.user_based:
            pred = knn_similarity.dot(normalized_ratings).div(knn_similarity.sum(axis=1), axis=0)
            pred = pred.add(mean_values, axis=0)  # Reverte normalização
        else:
            pred = normalized_ratings.dot(knn_similarity).div(knn_similarity.sum(axis=0), axis=1)
            pred = pred.add(mean_values, axis=1)  # Reverte normalização
        
        # Imputa valores faltantes para manter coerência
        imputer = SimpleImputer(strategy='mean')
        self.pred = pd.DataFrame(imputer.fit_transform(pred), index=self.ratings.index, columns=self.ratings.columns)
        
        # Salva matrizes finais para análise
        self.U = knn_similarity if self.user_based else self.ratings
        self.V = self.ratings.T if self.user_based else knn_similarity.T
        
        return self.pred
