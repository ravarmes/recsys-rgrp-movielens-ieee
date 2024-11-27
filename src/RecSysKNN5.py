import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
from abc import ABCMeta

class RecSysKNN5:
    __metaclass__ = ABCMeta
    
    def __init__(self, k, ratings=None, user_based=True, min_similarity=0.01, normalize=True, pearson_weight=0.8):
        """
        Parâmetros:
        - k: Número de vizinhos mais próximos.
        - ratings: Matriz de classificações.
        - user_based: True para filtragem baseada em usuários, False para itens.
        - min_similarity: Limite mínimo para similaridade.
        - normalize: Normalizar as classificações para média zero.
        - pearson_weight: Peso para combinar entre similaridade de cosseno e Pearson.
        """
        self.k = k
        self.user_based = user_based
        self.min_similarity = min_similarity
        self.normalize = normalize
        self.pearson_weight = pearson_weight
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
    
    def get_cosine_similarity(self, ratings):
        return pd.DataFrame(cosine_similarity(ratings.fillna(0)), index=ratings.index, columns=ratings.index)

    def get_pearson_similarity(self, ratings):
        if self.user_based:
            return ratings.T.corr().fillna(0)
        else:
            return ratings.corr().fillna(0)
    
    def get_similarity_matrix(self):
        ratings = self.ratings.copy()
        
        if self.normalize:
            ratings, _ = self.normalize_ratings(ratings)
        
        cosine_sim = self.get_cosine_similarity(ratings)
        pearson_sim = self.get_pearson_similarity(ratings)
        
        # Hibridização das similaridades
        similarity = self.pearson_weight * pearson_sim + (1 - self.pearson_weight) * cosine_sim
        return similarity.where(similarity.notnull(), 0)  # Certifica-se de que não há NaNs
    
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