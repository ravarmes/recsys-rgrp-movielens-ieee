import numpy as np
import pandas as pd
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

class RecSysRBM:
    def __init__(self, ratings, n_components=5, learning_rate=0.1, n_iter=1000):
        self.ratings = ratings
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.model = BernoulliRBM(n_components=self.n_components, learning_rate=self.learning_rate, n_iter=self.n_iter)
        self.n_users, self.n_items = ratings.shape
        self.user_ids, self.item_ids = np.where(~self.ratings.isnull())  # Obter índices de avaliações conhecidas

    def set_ratings(self, ratings):
        """Atualizar a matriz de ratings"""
        self.ratings = ratings

    def fit_model(self):
        """Treinar a RBM e ajustar aos dados de entrada"""
        # Substitui NaNs por 0 para o treinamento
        self.ratings.fillna(0, inplace=True)
        self.model.fit(self.ratings)

    def predict(self):
        """Fazer previsões a partir do modelo treinado"""
        # Obtenha as predições como dot product entre o input da RBM e os pesos aprendidos
        hidden_features = self.model.transform(self.ratings)
        predictions = np.dot(hidden_features, self.model.components_)
        return predictions

    def recommend(self, user_id, num_recommendations=5):
        """Recomende itens para um usuário específico"""
        predicted_ratings = self.predict()
        
        # Obter as classificações do usuário
        user_ratings = self.ratings.iloc[user_id].to_numpy()
        unrecommended_items = np.where(user_ratings == 0)[0]  # Itens não avaliados

        # Obter as previsões para itens não avaliados
        recommendations = predicted_ratings[user_id, unrecommended_items]
        
        # Classificar as previsões e obter os índices dos melhores itens
        recommended_indices = np.argsort(recommendations)[::-1][:num_recommendations]

        return unrecommended_items[recommended_indices]

    def evaluate_model(self):
        """Avaliar o modelo usando RMSE"""
        # Gerar previsões
        predictions = self.predict()

        # Calcular RMSE
        known_ratings_values = self.ratings.values[self.user_ids, self.item_ids]
        predicted_ratings_values = predictions[self.user_ids, self.item_ids]
        error = sqrt(mean_squared_error(known_ratings_values, predicted_ratings_values))

        return error