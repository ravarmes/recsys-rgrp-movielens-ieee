import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
import numpy.ma as ma

class RecSysKNN6():
    __metaclass__ = ABCMeta
    
    def __init__(self, k, ratings=None):
        self.k = k
        if ratings is not None:
            self.set_ratings(ratings)
    
    def set_ratings(self, ratings):
        self.ratings = ratings
        self.num_of_known_ratings_per_user = (~self.ratings.isnull()).sum(axis=1)
        self.num_of_known_ratings_per_movie = (~self.ratings.isnull()).sum(axis=0)

    def get_U(self):
        return None  # Not applicable for KNN
    
    def get_V(self):
        return None  # Not applicable for KNN
    
    @abstractmethod
    def fit_model(self):
        pass

    def knn(self, X):
        # Calcular a matriz de similaridade
        similarity_matrix = cosine_similarity(X.fillna(0))
        return similarity_matrix

    def recommend(self, user_index):
        # Obter a similaridade dos usuários
        similarity_matrix = self.knn(self.ratings)
        user_similarities = similarity_matrix[user_index]
        
        # Obter as classificações conhecidas do usuário
        user_ratings = self.ratings.iloc[user_index]
        
        # Considerar apenas os usuários mais semelhantes
        similar_users_indices = user_similarities.argsort()[-self.k-1:-1][::-1]
        
        # Calcular recomendações ponderadas
        weighted_scores = np.zeros(len(user_ratings))
        total_similarity = 0
        
        for idx in similar_users_indices:
            if idx != user_index:  # não considerar o próprio usuário
                similar_user_ratings = self.ratings.iloc[idx]
                non_nan_indices = user_ratings.index[~user_ratings.isna() & ~similar_user_ratings.isna()]
                weights = user_similarities[idx]  # Similaridade para o usuário similar
                
                # Somar pontuações ponderadas
                weighted_scores[non_nan_indices] += weights * similar_user_ratings[non_nan_indices]
                total_similarity += weights * len(non_nan_indices)  # Contar as contribuições
                
        # Se total_similarity for maior que 0, normalizar as recomendações
        if total_similarity > 0:
            weighted_scores /= total_similarity
        
        return pd.Series(weighted_scores, index=user_ratings.index)

class knn_RecSysKNN(RecSysKNN6):
    def fit_model(self, ratings=None):
        X = self.ratings if ratings is None else ratings
        self.recommendations = self.recommend(user_index=0)  # Recomendação para o primeiro usuário como exemplo
        self.error = ma.power(ma.masked_invalid(X.fillna(0) - self.recommendations), 2).sum()
        return self.recommendations, self.error